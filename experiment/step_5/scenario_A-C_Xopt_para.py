"""
Multi-core driver for the Xopt / Bayesian-optimisation version of Scenarios A, B, C
(the companion of `scenario_A-C_para.py`, which does the same for the scipy methods).

Same problems, same real FELsim objective, same raw-MSE target as `scenario_A-C_Xopt.ipynb`
— only the scheduling changes.

WHY BO NEEDS A DIFFERENT PARALLEL STRATEGY
------------------------------------------
A scipy run is a black box you can simply replicate. A BO run is not: one `X.step()` is
*fit GP -> maximise the acquisition -> evaluate -> append to X.data*, and step k+1 needs the
result of step k. The loop is inherently sequential, so the two useful axes are:

  AXIS 1 — across random restarts ("--mode restarts", the default)
      `n_runs` independent Xopt objects, each with its own random initialisation and its own
      GP. Embarrassingly parallel, exactly like the random starts in `scenario_A-C_para.py`:
      one process per restart, no communication. Sample-efficiency per restart is unchanged.

  AXIS 2 — inside one BO run, via Xopt itself ("--mode batch")
      Xopt's `Evaluator` owns a process pool, and `Xopt.step()` asks the generator for
      exactly `evaluator.max_workers` candidates:

          n_generate = self.evaluator.max_workers          # xopt/base.py, Xopt.step()
          new_samples = self.generator.suggest(n_generate)

      So `Evaluator(function=..., max_workers=q)` turns every step into batched (q-EI)
      Bayesian optimisation: `ExpectedImprovementGenerator` proposes q points at once
      (`supports_batch_generation = True`) and the pool evaluates all q simulations
      concurrently. This shortens a *single* BO run — the price is sample efficiency, since
      the q points in a batch are chosen without seeing each other's results.

  "--mode both" nests the two: `outer` restart processes, each running a batch of `inner`
      evaluations. Total processes = outer * inner, so keep `outer * inner <= max_cores`.

  Scenario B is 11 sequential stages (stage k+1 starts from the currents stage k wrote back),
  so a pipeline is never split: axis 1 parallelises whole pipelines, axis 2 parallelises the
  evaluations inside each stage's BO loop.

PICKLABILITY (why the objective is a class, not a closure)
----------------------------------------------------------
Both axes ship the objective to another process, and `Evaluator` calls `executor.map(...)`
on it. The notebook's `make_felsim_evaluator()` returns a *closure*, which cannot be pickled.
`FELsimObjective` below is a module-level callable holding only picklable config; it builds
its `beamOptimizer` lazily on first call in whatever process it lands in, and `__getstate__`
drops that cache so the object stays picklable on the way back.

HOW MUCH SPEEDUP TO EXPECT
--------------------------
With `P` processes:

  restarts : near-linear until `P` exceeds `n_runs` — speedup ~ min(P, n_runs). This is the
             one to use when you want restart statistics (conv_rate over many random inits).
  batch    : bounded by q = max_workers per step, and *less* than q in practice because each
             step still pays one serial GP fit + acquisition optimisation. Fitting the GP
             costs ~0.3-3 s and grows with the number of points, so with a ~0.6 s FELsim
             evaluation (Scenario A) the GP can dominate and batching buys little; with the
             ~5 s evaluation of Scenario C (118 elements) the simulation dominates and q-EI
             pays off. Rule of thumb: batch when t_eval >> t_GP, restart otherwise.

  Amdahl for one batched run:  t_step ~= t_GP + t_eval  (instead of t_GP + q * t_eval),
                               so speedup_per_run ~= (t_GP + q*t_eval) / (t_GP + t_eval).

Every driver prints `sum(per-task time) / elapsed` as the achieved speedup, and `--benchmark`
re-runs the same workload serially for a measured number.

USAGE
-----
    python scenario_A-C_Xopt_para.py --scenario A --n_runs 8 --max_cores 8
    python scenario_A-C_Xopt_para.py --scenario C --n_runs 4 --mode batch --q 4
    python scenario_A-C_Xopt_para.py --scenario B --n_runs 4 --n_steps 25
    python scenario_A-C_Xopt_para.py --scenario A --n_runs 4 --mode both --outer 2 --q 2
    python scenario_A-C_Xopt_para.py --scenario A --n_runs 4 --benchmark

Run it from `experiment/step_5/` (or anywhere — it chdir's to its own directory, which is
what `experiments_utils` needs to resolve `../../backend` and `../../beam_excel`).
"""

# ── 0. Thread pinning ────────────────────────────────────────────────
# Before NumPy / PyTorch are imported, or their thread pools are already sized. BoTorch's GP
# fit is itself multi-threaded, so without this every worker would fight for all the cores.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

# ── 1. Path setup & imports ──────────────────────────────────────────
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # experiments_utils resolves the backend relative to os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../../backend")))

import argparse
import copy
import pickle
import time
import warnings
from functools import partial
import concurrent.futures

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from experiments_utils import (
    PARTICLES,
    EXCEL_PATH,
    ExcelElements,
    beamOptimizer,
    ALPHA_XM,
    ALPHA_YM,
    BETA_XM,
    BETA_YM,
)
from configs import FELSIM_S1_CURRENTS

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator

torch.set_num_threads(1)  # one BLAS/GP thread per process — the pool provides the parallelism

# ── 2. Shared configuration (identical to scenario_A-C_Xopt.ipynb) ───
CURRENT_BOUNDS = (0.01, 1.5)
EPSILON = 1e-3
SEED = 42
XOPT_TAG = "Xopt-EI"

try:
    DEFAULT_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - 1
except TypeError:
    DEFAULT_CORES = 4
DEFAULT_CORES = max(1, DEFAULT_CORES)


def identity_func(x):
    """Top-level (picklable) stand-in for `lambda x: x` — closures cannot cross processes."""
    return x


def _v(name):
    """Variable spec understood by beamOptimizer: [name, attribute, transform]."""
    return [name, "current", identity_func]


# ── 3. The picklable FELsim objective ────────────────────────────────
class FELsimObjective:
    """`inputs dict -> {"MSE": ..., "<elem>_<quantity>_<plane>": ...}` for Xopt.

    Equivalent to `make_felsim_evaluator()` in the notebook, but as a module-level callable
    so it can be pickled into Xopt's evaluator pool and into restart workers.

    * `MSE` is the raw weighted MSE — the same quantity scipy minimises, no log transform.
    * A diverging simulation returns NaN instead of raising: Xopt then drops that row when
      fitting the GP rather than aborting the run.
    * The `beamOptimizer` is built on first use in each process and cached; `__getstate__`
      drops the cache (it holds bound methods and beamline objects) so pickling still works.
    * The particle bunch is carried as an attribute, NOT read from `experiments_utils`.
      `PARTICLES` is drawn at import time from the unseeded global torch RNG, so every
      spawned process (restart worker or Xopt evaluator worker) would otherwise materialise
      its own beam and optimise a subtly different objective. Carrying it pins one bunch for
      the whole study — ~48 KB per pickle, which is nothing next to a FELsim evaluation.
    """

    def __init__(self, vars_spec, obj_spec, beamline_len, bounds=CURRENT_BOUNDS,
                 particles=None):
        self.vars_spec = vars_spec
        self.obj_spec = obj_spec
        self.beamline_len = beamline_len
        self.bounds = tuple(bounds)
        self.particles = PARTICLES if particles is None else particles
        self.var_names = list(dict.fromkeys(v[0] for v in vars_spec.values()))
        self.out_names = [
            f"{elem}_{t['measure'][1]}_{t['measure'][0]}"
            for elem, targets in obj_spec.items()
            for t in targets
        ]
        self._opti = None
        self._order = None
        self._flat_goals = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_opti"] = state["_order"] = state["_flat_goals"] = None
        return state

    def _ensure_built(self, beamline=None):
        if self._opti is not None:
            return
        bl = beamline if beamline is not None else ExcelElements(EXCEL_PATH).create_beamline()
        self._opti = beamOptimizer(bl[: self.beamline_len], self.particles.clone())
        start_point = {n: {"bounds": self.bounds, "start": 1.0} for n in self.var_names}
        # _prepare() wires up the objectives and the internal variable order without optimising
        self._opti._prepare(self.vars_spec, start_point, copy.deepcopy(self.obj_spec))
        self._order = list(self._opti.variablesToOptimize)  # built from a set -> arbitrary order
        self._flat_goals = [g for e in self._opti.objectives for g in self._opti.objectives[e]]

    def __call__(self, inputs: dict) -> dict:
        out = {"MSE": np.nan, **{n: np.nan for n in self.out_names}}
        try:
            self._ensure_built()
            x = np.array([float(inputs[n]) for n in self._order])  # map names -> internal order
            mse = float(self._opti._optiSpeed(x))
            if np.isfinite(mse):
                out["MSE"] = mse
                for name, g in zip(self.out_names, self._flat_goals):
                    val = g["measured"]
                    out[name] = float(val.item() if hasattr(val, "item") else val)
        except Exception:  # noqa: BLE001 - a failed point is NaN, never a dead run
            pass
        return out


# ── 4. Scenario definitions (verbatim from scenario_A-C_Xopt.ipynb) ──
A_BEAMLINE_LEN = 10
A_VARS = {1: _v("I"), 3: _v("I2")}
A_OBJ = {
    8: [{"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0}],
    9: [{"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}],
}

C_BEAMLINE_LEN = 118
C_VARS = {
    56: _v("I_56"), 58: _v("I_58"),
    61: _v("I_61"), 63: _v("I_63"),
    76: _v("I_76"), 78: _v("I_78"), 80: _v("I_80"),
    87: _v("I_87"), 93: _v("I_93"), 95: _v("I_95"), 97: _v("I_97"),
}
C_OBJ = {
    59: [
        {"measure": ["x", "envelope"], "goal": 0.0, "weight": 1.0},
        {"measure": ["y", "envelope"], "goal": 0.0, "weight": 1.0},
    ],
    117: [
        {"measure": ["x", "alpha"], "goal": ALPHA_XM, "weight": 1.0},
        {"measure": ["y", "alpha"], "goal": ALPHA_YM, "weight": 1.0},
        {"measure": ["x", "beta"], "goal": BETA_XM, "weight": 1.0},
        {"measure": ["y", "beta"], "goal": BETA_YM, "weight": 1.0},
    ],
}

# (label, segment variables, objectives, beamline slice length)
STAGES_B = [
    (
        "Stage 1 Doublet",
        {1: _v("I"), 3: _v("I2")},
        {
            8: [
                {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.0},
            ],
            9: [
                {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
            ],
        },
        10,
    ),
    (
        "Stage 2 Chrom.1",
        {10: _v("I")},
        {15: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        16,
    ),
    (
        "Stage 3 Triplet1",
        {16: _v("I"), 18: _v("I2"), 20: _v("I3")},
        {
            25: [
                {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.5},
            ],
            26: [
                {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
            ],
        },
        27,
    ),
    (
        "Stage 4 Chrom.2",
        {27: _v("I")},
        {32: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        33,
    ),
    (
        "Stage 5 DblTriplet",
        {37: _v("I"), 35: _v("I2"), 33: _v("I3")},
        {
            37: [
                {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["x", "envelope"], "goal": 2.0, "weight": 1},
                {"measure": ["y", "envelope"], "goal": 2.0, "weight": 1},
            ]
        },
        38,
    ),
    (
        "Stage 6 Chrom.3",
        {50: _v("I")},
        {55: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        56,
    ),
    (
        "Stage 7 IP",
        {56: _v("I"), 58: _v("I2")},
        {
            59: [
                {"measure": ["x", "envelope"], "goal": 0.0, "weight": 1},
                {"measure": ["y", "envelope"], "goal": 0.0, "weight": 1},
            ]
        },
        60,
    ),
    (
        "Stage 8 Doublet2",
        {61: _v("I"), 63: _v("I2")},
        {
            68: [
                {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.5},
            ],
            69: [
                {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
            ],
        },
        70,
    ),
    (
        "Stage 9 Chrom.4",
        {70: _v("I")},
        {75: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        76,
    ),
    (
        "Stage 10 Triplet3",
        {76: _v("I"), 78: _v("I2"), 80: _v("I3")},
        {
            85: [
                {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.5},
            ],
            86: [
                {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
            ],
        },
        87,
    ),
    (
        "Stage 11 UND Match",
        {87: _v("Ic"), 93: _v("I"), 95: _v("I2"), 97: _v("I3")},
        {
            92: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 0.5}],
            117: [
                {"measure": ["x", "alpha"], "goal": ALPHA_XM, "weight": 1},
                {"measure": ["y", "alpha"], "goal": ALPHA_YM, "weight": 1},
                {"measure": ["x", "beta"], "goal": BETA_XM, "weight": 1},
                {"measure": ["y", "beta"], "goal": BETA_YM, "weight": 1},
            ],
        },
        118,
    ),
]

SCENARIOS = {
    "A": {"vars": A_VARS, "obj": A_OBJ, "beamline_len": A_BEAMLINE_LEN},
    "C": {"vars": C_VARS, "obj": C_OBJ, "beamline_len": C_BEAMLINE_LEN},
}

# default BO budgets, mirroring the notebook
BUDGETS = {
    "A": {"n_init": 1, "n_steps": 30},
    "B": {"n_init": 2, "n_steps": 25},   # per stage
    "C": {"n_init": 11, "n_steps": 60},
}


# ── 5. The BO loop (one restart) ─────────────────────────────────────
def run_bo_once(objective, n_init, n_steps, seed=SEED, bounds=CURRENT_BOUNDS,
                max_workers=1, executor=None, beamline=None, verbose=False, label=""):
    """One Xopt Bayesian-optimisation run on the raw MSE.

    `max_workers > 1` (optionally with a pre-built `executor`) makes this a *batched* run:
    Xopt.step() requests `evaluator.max_workers` candidates from the EI generator and the
    evaluator's pool simulates them concurrently.

    Returns (X, stats).
    """
    if beamline is not None:  # reuse the caller's beamline (Scenario B hands one over)
        objective._ensure_built(beamline)

    vocs = VOCS(variables={n: list(bounds) for n in objective.var_names},
                objectives={"MSE": "MINIMIZE"})
    ev_kwargs = {"function": objective, "max_workers": max_workers}
    if executor is not None:
        ev_kwargs["executor"] = executor
    evaluator = Evaluator(**ev_kwargs)
    X = Xopt(evaluator=evaluator,
             generator=ExpectedImprovementGenerator(vocs=vocs),
             vocs=vocs)

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    X.random_evaluate(n_init, seed=seed)
    n_fallback = 0
    steps = int(np.ceil(n_steps / max(1, max_workers)))  # keep the evaluation budget constant
    for i in range(steps):
        try:
            X.step()                      # fit GP -> suggest max_workers points -> evaluate
        except Exception:                 # noqa: BLE001 - degenerate GP fit: fall back to random
            n_fallback += 1
            X.random_evaluate(max(1, max_workers), seed=seed + 1000 + i)
        if verbose and (i + 1) % 5 == 0:
            print(f"      {label:18s} step {i + 1:3d}/{steps}   best MSE = {X.data['MSE'].min():.3e}")
    wall = time.perf_counter() - t0

    mse = X.data["MSE"].to_numpy(dtype=float)
    n_ok = int(np.isfinite(mse).sum())
    if n_ok:
        best_pos = int(np.nanargmin(mse))
        best = X.data.iloc[best_pos]
        hist = np.minimum.accumulate(np.nan_to_num(mse, nan=np.inf))
        n_to_eps = int(np.argmax(hist < EPSILON)) + 1 if np.any(hist < EPSILON) else None
        best_x = {n: float(best[n]) for n in objective.var_names}
        measured = {n: float(best[n]) for n in objective.out_names}
        final_mse = float(best["MSE"])
    else:
        best_pos, n_to_eps = -1, None
        best_x = {n: np.nan for n in objective.var_names}
        measured = {n: np.nan for n in objective.out_names}
        final_mse = np.nan

    stats = {
        "final_mse": final_mse,
        "best_x": best_x,
        "measured": measured,
        "best_eval": best_pos + 1,
        "nfev": int(len(X.data)),
        "n_failed": int(len(X.data) - n_ok),
        "n_fallback": n_fallback,
        "n_to_eps": n_to_eps,
        "wall_time": round(wall, 3),
        "success": bool(np.isfinite(final_mse) and final_mse < EPSILON),
        "mse_curve": [float(v) for v in mse],
        "best_mse_curve": [float(v) for v in np.minimum.accumulate(np.nan_to_num(mse, nan=np.inf))],
    }
    return X, stats


# ── 6. Workers (top-level so they survive pickling to a spawned process) ──
def _worker_bo_restart(task, particles, inner_workers=1):
    """AXIS 1 for scenario A/C: one independent BO restart in its own process.

    The scenario config and the particle bunch ride along in/with the task — a spawned worker
    re-imports this module from disk, so module-level state (a caller's edited A_OBJ, and in
    particular `PARTICLES`, which is redrawn on every import) would never reach it otherwise.
    """
    scenario, cfg, run_idx, n_init, n_steps = task
    try:
        objective = FELsimObjective(cfg["vars"], cfg["obj"], cfg["beamline_len"],
                                    particles=particles)
        _, st = run_bo_once(objective, n_init, n_steps, seed=SEED + run_idx * 997,
                            max_workers=inner_workers, label=f"{scenario} run{run_idx + 1}")
        rec = {
            "run": run_idx + 1, "scenario": scenario, "method_tag": XOPT_TAG,
            "final_mse": st["final_mse"], "nfev": st["nfev"], "nit": st["nfev"],
            "wall_time": st["wall_time"], "success": st["success"],
            "n_to_eps": st["n_to_eps"], "n_failed": st["n_failed"],
            "n_fallback": st["n_fallback"], "best_eval": st["best_eval"],
            "mse_curve": st["mse_curve"], "best_mse_curve": st["best_mse_curve"],
            "result_x": st["best_x"], **st["best_x"], **st["measured"],
        }
        return rec, True
    except Exception as exc:  # noqa: BLE001
        return {"run": run_idx + 1, "scenario": scenario, "method_tag": XOPT_TAG,
                "final_mse": np.nan, "error": str(exc)}, False


def _worker_bo_B_pipeline(task, particles, inner_workers=1):
    """AXIS 1 for scenario B: one full 11-stage BO pipeline in its own process.

    Stages stay sequential — each stage's best currents are written back into this task's
    private beamline before the next stage's BO run starts (and stage 5 mirrors its currents
    onto elements 39/41/43, as in scenario_B_utils).
    """
    stages, run_idx, n_init, n_steps = task
    stage_records = []
    executor = None
    try:
        bl_run = ExcelElements(EXCEL_PATH).create_beamline()
        if inner_workers > 1:
            # one pool for the whole pipeline instead of one per stage (11 pools would pay
            # the ~25 s interpreter start-up eleven times over)
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=inner_workers)

        for s_i, (lbl, seg_var, obj, bl_len) in enumerate(stages):
            objective = FELsimObjective(seg_var, obj, bl_len, particles=particles)
            _, st = run_bo_once(objective, n_init, n_steps,
                                seed=SEED + run_idx * 997 + s_i,
                                max_workers=inner_workers, executor=executor,
                                beamline=bl_run, label=lbl)

            if all(np.isfinite(v) for v in st["best_x"].values()):
                for elem_idx, var_info in seg_var.items():
                    bl_run[elem_idx].current = float(st["best_x"][var_info[0]])
                if s_i == 4:  # stage 5 symmetry
                    bl_run[43].current = bl_run[33].current
                    bl_run[41].current = bl_run[35].current
                    bl_run[39].current = bl_run[37].current

            stage_records.append({
                "run": run_idx + 1, "stage": s_i + 1, "stage_name": lbl,
                "method_tag": XOPT_TAG, "final_mse": st["final_mse"],
                "nfev": st["nfev"], "nit": st["nfev"], "wall_time": st["wall_time"],
                "success": st["success"], "n_to_eps": st["n_to_eps"],
                "n_failed": st["n_failed"], "n_fallback": st["n_fallback"],
                **st["best_x"],
            })

        finals = {
            elem_idx: float(bl_run[elem_idx].current)
            for _, seg_var, _, _ in stages
            for elem_idx in seg_var
        }
        if len(stages) > 4:
            finals.update({i: float(bl_run[i].current) for i in (39, 41, 43)})
        return stage_records, finals, True
    except Exception as exc:  # noqa: BLE001
        return stage_records, {"error": str(exc)}, False
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


# ── 7. Reporting helpers ─────────────────────────────────────────────
def _report_speedup(scenario, records, elapsed, n_workers, mode, derivable=True):
    """Speedup estimated as (summed per-task BO time) / (elapsed wall time).

    That estimate is only meaningful when the tasks ran *concurrently* (axis 1). Inside a
    single batched run (axis 2) the parallelism happens below `wall_time`, so summing it
    would just return the elapsed time and print a meaningless 1.00x — in that case report
    throughput instead and point at `--benchmark` for a real comparison.
    """
    n_eval = int(np.nansum([r.get("nfev", np.nan) for r in records]))
    if not derivable:
        print(
            f"\n⚡ Scenario {scenario} [{mode}]: {n_eval} FELsim evaluations, {n_workers} processes\n"
            f"   elapsed wall time         : {elapsed:8.1f} s\n"
            f"   throughput                : {n_eval / elapsed:8.2f} evaluations/s\n"
            f"   (speedup cannot be derived from one batched run — use --benchmark to measure it)"
        )
        return {"serial_equivalent_s": np.nan, "elapsed_s": elapsed, "speedup": np.nan,
                "evals_per_s": n_eval / elapsed if elapsed else np.nan,
                "n_evaluations": n_eval, "n_workers": n_workers, "mode": mode}

    serial = float(np.nansum([r.get("wall_time", np.nan) for r in records]))
    speedup = serial / elapsed if elapsed > 0 else float("nan")
    print(
        f"\n⚡ Scenario {scenario} [{mode}]: {len(records)} tasks, {n_workers} processes\n"
        f"   BO time summed over tasks : {serial:8.1f} s  (serial equivalent)\n"
        f"   elapsed wall time         : {elapsed:8.1f} s\n"
        f"   speedup ~= {speedup:.2f}x   (parallel efficiency "
        f"{100 * speedup / max(1, n_workers):.0f}% of {n_workers} processes)"
    )
    return {"serial_equivalent_s": serial, "elapsed_s": elapsed, "speedup": speedup,
            "n_evaluations": n_eval, "n_workers": n_workers, "mode": mode}


def _summarise_restarts(scenario, records, VARS, OBJ, verbose=True):
    """One row per restart + a per-scenario summary in the shape of the scipy tables."""
    df = pd.DataFrame(records)
    if df.empty or "final_mse" not in df.columns:
        print(f"⚠️  Scenario {scenario}: no usable result to summarise")
        return df, pd.DataFrame()

    current_name = list(dict.fromkeys(info[0] for info in VARS.values()))
    evalPos = [f"{e}_{t['measure'][1]}_{t['measure'][0]}" for e, ts in OBJ.items() for t in ts]
    for col in (*current_name, *evalPos):
        if col not in df.columns:
            df[col] = np.nan
    df["is_converged"] = df["final_mse"] < EPSILON

    conv = df.loc[df["final_mse"] < EPSILON, "final_mse"]
    best_row = df.loc[df["final_mse"].fillna(np.inf).idxmin()]
    summary = pd.DataFrame([{
        "method_tag": XOPT_TAG,
        "n_runs": len(df),
        "conv_rate": float(df["is_converged"].mean()),
        "nfev_mean": float(df["nfev"].mean()) if "nfev" in df else np.nan,
        "wall_time_mean": float(df["wall_time"].mean()) if "wall_time" in df else np.nan,
        "final_mse_geom_conv": float(10 ** np.log10(conv).mean()) if len(conv) else np.nan,
        "final_mse_best": float(best_row["final_mse"]),
        **{n: float(best_row[n]) for n in current_name},
        **{f"{info[0]}_ref": FELSIM_S1_CURRENTS[k] for k, info in VARS.items()},
        **{n: float(best_row[n]) for n in evalPos},
    }])

    if verbose:
        print("\n" + "=" * 110)
        print(f" 🎯 SCENARIO {scenario} ({XOPT_TAG}): {len(df)} restarts, Tol < {EPSILON}")
        print("=" * 110)
        cols = ["run", "final_mse", "nfev", "wall_time", "success", "n_to_eps", *current_name]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        print(f"\n conv_rate = {summary.iloc[0]['conv_rate']:.0%}   |   best MSE = "
              f"{summary.iloc[0]['final_mse_best']:.3e}   |   mean wall/run = "
              f"{summary.iloc[0]['wall_time_mean']:.1f} s")
        ref_line = "   ".join(
            f"{n}={float(best_row[n]):.4f} (ref {FELSIM_S1_CURRENTS[k]:.4f})"
            for k, info in VARS.items() for n in [info[0]]
        )
        print(f" champion currents: {ref_line}")
        print("=" * 110)
    return df, summary


# ── 8. Parallel drivers ──────────────────────────────────────────────
def run_restarts_parallel(scenario, n_runs, max_cores=DEFAULT_CORES, cfg=None,
                          n_init=None, n_steps=None, inner_workers=1, verbose=True):
    """AXIS 1 (A/C): `n_runs` independent BO restarts spread over processes."""
    cfg = cfg if cfg is not None else SCENARIOS[scenario]
    n_init = BUDGETS[scenario]["n_init"] if n_init is None else n_init
    n_steps = BUDGETS[scenario]["n_steps"] if n_steps is None else n_steps

    tasks = [(scenario, cfg, i, n_init, n_steps) for i in range(n_runs)]
    outer = max(1, min(max_cores, n_runs))
    mode = "restarts" if inner_workers == 1 else f"both(outer={outer},q={inner_workers})"
    if verbose:
        print(f"\n▶ 🚀 Scenario {scenario} [{mode}] — {n_runs} BO restarts "
              f"({n_init} random + {n_steps} evaluations each) on {outer} processes"
              + (f", each batching {inner_workers} evaluations" if inner_workers > 1 else ""))

    worker = partial(_worker_bo_restart, particles=PARTICLES, inner_workers=inner_workers)
    records, n_failed = [], 0
    t0 = time.perf_counter()
    if outer == 1:  # no pool: keeps --benchmark honest and avoids a pointless process
        results_iter = map(worker, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=outer)
        results_iter = executor.map(worker, tasks)
    try:
        for done, (rec, ok) in enumerate(results_iter, start=1):
            records.append(rec)
            if not ok:
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ run {rec.get('run')} failed: {rec.get('error')}")
            elif verbose:
                print(f"   [{done}/{n_runs}] run {rec['run']:3d}  MSE={rec['final_mse']:.3e}  "
                      f"nfev={rec['nfev']:<4} t={rec['wall_time']:.1f}s")
    finally:
        if outer > 1:
            executor.shutdown(wait=True)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"✓ Scenario {scenario} finished in {elapsed:.2f}s ({n_failed} failed)")
    df, summary = _summarise_restarts(scenario, records, cfg["vars"], cfg["obj"], verbose=verbose)
    perf = _report_speedup(scenario, records, elapsed, outer * max(1, inner_workers), mode)
    return records, df, summary, perf


def run_scenario_B_parallel(n_runs, max_cores=DEFAULT_CORES, stages=None,
                            n_init=None, n_steps=None, inner_workers=1, verbose=True):
    """AXIS 1 (B): whole 11-stage BO pipelines in parallel; stages sequential inside."""
    stages = stages if stages is not None else STAGES_B
    n_init = BUDGETS["B"]["n_init"] if n_init is None else n_init
    n_steps = BUDGETS["B"]["n_steps"] if n_steps is None else n_steps

    tasks = [(stages, i, n_init, n_steps) for i in range(n_runs)]
    outer = max(1, min(max_cores, n_runs))
    mode = "restarts" if inner_workers == 1 else f"both(outer={outer},q={inner_workers})"
    if verbose:
        print(f"\n▶ 🚀 Scenario B [{mode}] — {n_runs} pipelines x {len(stages)} stages "
              f"({n_init} random + {n_steps} evaluations per stage) on {outer} processes")
        print("   stages inside a pipeline stay sequential (each hands its currents on)")

    worker = partial(_worker_bo_B_pipeline, particles=PARTICLES, inner_workers=inner_workers)
    all_records, finals, n_failed = [], [], 0
    t0 = time.perf_counter()
    if outer == 1:
        results_iter = map(worker, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=outer)
        results_iter = executor.map(worker, tasks)
    try:
        for done, (recs, fin, ok) in enumerate(results_iter, start=1):
            all_records.extend(recs)
            if ok:
                finals.append(fin)
            else:
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ pipeline {done} failed after {len(recs)} stages: {fin.get('error')}")
            if verbose:
                n_conv = sum(1 for r in recs if r["final_mse"] < EPSILON)
                print(f"   [{done}/{n_runs}] {len(recs)}/{len(stages)} stages, "
                      f"{n_conv} below EPSILON")
    finally:
        if outer > 1:
            executor.shutdown(wait=True)
    elapsed = time.perf_counter() - t0

    df_B = pd.DataFrame(all_records)
    if verbose and not df_B.empty:
        print(f"\n✓ Scenario B finished in {elapsed:.2f}s ({n_failed} failed pipelines)")
        print(f"  stages with MSE < {EPSILON:g}: "
              f"{int((df_B['final_mse'] < EPSILON).sum())}/{len(df_B)}")
        per_stage = df_B.groupby(["stage", "stage_name"])["final_mse"].min()
        print("  best MSE per stage across pipelines:")
        for (stage, name), val in per_stage.items():
            flag = "✓" if val < EPSILON else " "
            print(f"    {flag} {stage:2d} {name:22s} {val:.3e}")
    perf = _report_speedup("B", all_records, elapsed, outer * max(1, inner_workers), mode)
    return all_records, df_B, finals, perf


def run_batch_single(scenario, q, max_cores=DEFAULT_CORES, cfg=None, n_init=None,
                     n_steps=None, run_idx=0, verbose=True):
    """AXIS 2: ONE BO run whose every step evaluates `q` candidates concurrently (q-EI).

    This is Xopt's own parallelism: `Evaluator(max_workers=q)` owns the pool and
    `Xopt.step()` requests exactly `q` candidates per iteration.
    """
    cfg = cfg if cfg is not None else SCENARIOS[scenario]
    n_init = BUDGETS[scenario]["n_init"] if n_init is None else n_init
    n_steps = BUDGETS[scenario]["n_steps"] if n_steps is None else n_steps
    q = max(1, min(q, max_cores))

    if verbose:
        print(f"\n▶ 🚀 Scenario {scenario} [batch q={q}] — one BO run, "
              f"{n_init} random + ~{n_steps} evaluations in batches of {q}")
    objective = FELsimObjective(cfg["vars"], cfg["obj"], cfg["beamline_len"])
    t0 = time.perf_counter()
    X, st = run_bo_once(objective, n_init, n_steps, seed=SEED + run_idx * 997,
                        max_workers=q, verbose=verbose, label=f"{scenario} batch")
    elapsed = time.perf_counter() - t0

    rec = {"run": run_idx + 1, "scenario": scenario, "method_tag": XOPT_TAG,
           "final_mse": st["final_mse"], "nfev": st["nfev"], "nit": st["nfev"],
           "wall_time": st["wall_time"], "success": st["success"],
           "n_to_eps": st["n_to_eps"], "n_failed": st["n_failed"],
           "n_fallback": st["n_fallback"], "mse_curve": st["mse_curve"],
           "best_mse_curve": st["best_mse_curve"], "result_x": st["best_x"],
           **st["best_x"], **st["measured"]}
    if verbose:
        print(f"✓ best MSE = {st['final_mse']:.3e} after {st['nfev']} evaluations "
              f"in {elapsed:.1f}s")
    df, summary = _summarise_restarts(scenario, [rec], cfg["vars"], cfg["obj"], verbose=verbose)
    perf = _report_speedup(scenario, [rec], elapsed, q, f"batch(q={q})", derivable=False)
    return [rec], df, summary, perf


# ── 9. Entry point ───────────────────────────────────────────────────
def _save(obj, save_dir, filename):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✓ saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xopt Bayesian optimisation of scenarios A/B/C across CPU cores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", choices=["A", "B", "C", "all"], default="A")
    parser.add_argument("--mode", choices=["restarts", "batch", "both"], default="restarts",
                        help="restarts: independent BO runs in parallel | "
                             "batch: one BO run, q-EI evaluated in parallel by Xopt | "
                             "both: nested (outer restarts x q evaluations)")
    parser.add_argument("--n_runs", type=int, default=4, help="BO restarts (axis 1)")
    parser.add_argument("--q", type=int, default=4,
                        help="candidates per BO step = Evaluator.max_workers (axis 2)")
    parser.add_argument("--outer", type=int, default=0,
                        help="mode=both: restart processes (0 -> max_cores // q)")
    parser.add_argument("--max_cores", type=int, default=DEFAULT_CORES)
    parser.add_argument("--n_init", type=int, default=0, help="0 -> scenario default")
    parser.add_argument("--n_steps", type=int, default=0,
                        help="evaluation budget after the random init (0 -> scenario default)")
    parser.add_argument("--save_dir", type=str, default="../../results/scenario_A-C_Xopt_para",
                        help="where to pickle the results ('' to skip)")
    parser.add_argument("--benchmark", action="store_true",
                        help="also run the same restarts serially and print the measured speedup")
    args = parser.parse_args()

    n_init = args.n_init or None
    n_steps = args.n_steps or None
    print(f"host cores: {os.cpu_count()}   |   using: {args.max_cores}   |   mode: {args.mode}"
          f"   |   n_runs: {args.n_runs}   |   scenario: {args.scenario}")

    t_all = time.perf_counter()
    for scen in (["A", "B", "C"] if args.scenario == "all" else [args.scenario]):
        if args.mode == "batch":
            if scen == "B":
                out = run_scenario_B_parallel(n_runs=1, max_cores=args.max_cores,
                                              n_init=n_init, n_steps=n_steps,
                                              inner_workers=min(args.q, args.max_cores))
                _save({"records": out[0], "df": out[1], "final_currents": out[2], "perf": out[3]},
                      args.save_dir, "scenario_B_xopt_batch.pkl")
            else:
                out = run_batch_single(scen, q=args.q, max_cores=args.max_cores,
                                       n_init=n_init, n_steps=n_steps)
                _save({"records": out[0], "df": out[1], "summary": out[2], "perf": out[3]},
                      args.save_dir, f"scenario_{scen}_xopt_batch.pkl")
            continue

        inner = 1
        outer_cap = args.max_cores
        if args.mode == "both":
            inner = max(1, min(args.q, args.max_cores))
            outer_cap = args.outer if args.outer > 0 else max(1, args.max_cores // inner)

        if scen == "B":
            recs, df, finals, perf = run_scenario_B_parallel(
                n_runs=args.n_runs, max_cores=outer_cap, n_init=n_init, n_steps=n_steps,
                inner_workers=inner)
            _save({"records": recs, "df": df, "final_currents": finals, "perf": perf},
                  args.save_dir, "scenario_B_xopt_para.pkl")
        else:
            recs, df, summary, perf = run_restarts_parallel(
                scen, n_runs=args.n_runs, max_cores=outer_cap, n_init=n_init,
                n_steps=n_steps, inner_workers=inner)
            _save({"records": recs, "df": df, "summary": summary, "perf": perf},
                  args.save_dir, f"scenario_{scen}_xopt_para.pkl")

            if args.benchmark:
                print(f"\n⏱  benchmark: repeating Scenario {scen} serially (1 process)...")
                t0 = time.perf_counter()
                run_restarts_parallel(scen, n_runs=args.n_runs, max_cores=1, n_init=n_init,
                                      n_steps=n_steps, inner_workers=1, verbose=False)
                serial_elapsed = time.perf_counter() - t0
                print(f"   serial   : {serial_elapsed:8.1f} s\n"
                      f"   parallel : {perf['elapsed_s']:8.1f} s on {perf['n_workers']} processes\n"
                      f"   measured speedup: {serial_elapsed / perf['elapsed_s']:.2f}x")

    print(f"\n🏁 total wall time: {time.perf_counter() - t_all:.2f} s")
