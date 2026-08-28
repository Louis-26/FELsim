"""
Multiprocessing runners for the Xopt / Bayesian-optimisation version of Scenarios A, B, C —
the `--use_multi 1` path of `step_5/scenario_A-C_Xopt_seq.py`.

This is to `utils/Xopt_utils.py` what `experiment/parallel_utils.py` is to
`experiments_utils` / `scenario_B_utils`: same problems, same FELsim objective, same raw-MSE
target and the same record layout — only the scheduling changes.

WHY BO NEEDS A DIFFERENT PARALLEL STRATEGY THAN SCIPY
-----------------------------------------------------
A scipy run is a black box you can simply replicate. A BO run is not: one `X.step()` is
*fit GP -> maximise the acquisition -> evaluate -> append to X.data*, and step k+1 needs the
result of step k. The loop is inherently sequential, so the two useful axes are:

  AXIS 1 — across random restarts (`mode="restarts"`, the default)
      `N_RUNS` independent Xopt objects, each with its own random initialisation and its own
      GP. Embarrassingly parallel, exactly like the random starts in `parallel_utils.py`:
      one process per restart, no communication. Sample efficiency per restart is unchanged.

  AXIS 2 — inside one BO run, via Xopt itself (`q > 1`)
      Xopt's `Evaluator` owns a process pool, and `Xopt.step()` asks the generator for
      exactly `evaluator.max_workers` candidates:

          n_generate = self.evaluator.max_workers          # xopt/base.py, Xopt.step()
          new_samples = self.generator.suggest(n_generate)

      So `Evaluator(function=..., max_workers=q)` turns every step into batched (q-EI)
      Bayesian optimisation: `ExpectedImprovementGenerator` proposes q points at once
      (`supports_batch_generation = True`) and the pool evaluates all q simulations
      concurrently. This shortens a *single* BO run — the price is sample efficiency, since
      the q points of a batch are chosen without seeing each other's results.

  `mode="both"` nests the two: `outer` restart processes, each batching `q` evaluations.
  Total processes = outer * q, so keep `outer * q <= max_cores`.

  Scenario B is 11 sequential stages (stage k+1 starts from the currents stage k wrote back),
  so a pipeline is never split: axis 1 parallelises whole pipelines, axis 2 parallelises the
  evaluations inside each stage's BO loop.

Rule of thumb for axis 2: one step costs `t_GP + t_eval` instead of `t_GP + q * t_eval`, so
it only pays when `t_eval >> t_GP`. The GP fit is ~0.3-3 s and grows with the number of
points; a Scenario A evaluation is ~1 s (GP dominates -> batch buys little), a Scenario C
evaluation is ~5 s (simulation dominates -> q-EI pays off).

THREE THINGS THAT SILENTLY BREAK MULTIPROCESSING HERE
-----------------------------------------------------
1. `configs.PARTICLES` is drawn at import time from the *unseeded* global torch RNG. Under
   spawn each worker re-imports the module and would materialise its OWN 1000-particle bunch,
   i.e. every restart would optimise a slightly different objective. Every worker here
   therefore takes `particles` as an argument and the bunch rides along in the pickle
   (~48 KB, nothing next to a FELsim evaluation).
2. `Xopt_utils.make_felsim_evaluator()` returns a **closure**, which cannot be pickled — and
   both axes ship the objective to another process (`Evaluator` calls `executor.map` on it).
   `FELsimObjective` below is the same objective as a module-level callable holding only
   picklable config; it builds its `beamOptimizer` lazily in whatever process it lands in and
   `__getstate__` drops that cache so it stays picklable on the way back.
3. `beamOptimizer.variablesToOptimize` is built from a `set`, so its order follows the
   per-interpreter hash randomisation. `FELsimObjective` maps Xopt's named inputs into that
   order by name, so it is correct in any process; `PYTHONHASHSEED=0` is pinned for the
   children anyway so a given run index means the same thing across invocations.

Also: one persistent pool per driver, never a pool per task — importing the backend costs
~25 s per process. Scenario B builds a single pool for the whole 11-stage pipeline for the
same reason.

Every driver returns a `perf` dict and prints `sum(per-task BO time) / elapsed` as the
achieved speedup. That estimate is only meaningful when tasks ran *concurrently* (axis 1):
inside one batched run the parallelism happens below `wall_time`, so throughput is reported
instead.
"""

import concurrent.futures
import copy
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# `configs` lives in `experiment/`, which the entry script puts on sys.path (spawned children
# inherit the parent's sys.path and cwd, so the same import works inside a worker).
from configs import PARTICLES, EXCEL_PATH, ExcelElements, beamOptimizer, XOPT_TAG

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator

CURRENT_BOUNDS = (0.01, 1.5)     # same defaults as Xopt_utils
EPSILON        = 1e-3            # success threshold on the raw MSE (NOT configs.EPSILON = 1)
SEED           = 42

# leave one core for the OS / the launching shell
try:
    DEFAULT_CORES = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - 1)
except TypeError:
    DEFAULT_CORES = 4


# ── Environment hygiene for the children ─────────────────────────────
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin_child_environment():
    """One BLAS/OMP thread per worker + a fixed hash seed, applied to the *children*.

    Called in the parent right before a pool is created, not at import time: NumPy/PyTorch are
    already loaded here so the parent's own thread pools keep their size (the sequential path
    of the entry script must stay multi-threaded), while a spawned child inherits this
    environment block *before* it imports anything. BoTorch's GP fit is itself multi-threaded,
    so without this every worker would fight for all the cores and the pool would be slower
    than the serial run.
    """
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    os.environ["PYTHONHASHSEED"] = "0"


def _enter_worker():
    """Per-task setup inside a worker process."""
    torch.set_num_threads(1)   # belt and braces: also caps PyTorch's intra-op pool


def measured_names(obj):
    """Column names of the measured quantities, e.g. '8_alpha_x' (as in Xopt_utils)."""
    return [f"{elem}_{t['measure'][1]}_{t['measure'][0]}"
            for elem, targets in obj.items() for t in targets]


# ── The picklable FELsim objective ───────────────────────────────────
class FELsimObjective:
    """`inputs dict -> {"MSE": ..., "<elem>_<quantity>_<plane>": ...}` for Xopt.

    Behaviourally identical to `Xopt_utils.make_felsim_evaluator()`, but a module-level
    callable instead of a closure so it can be pickled into Xopt's evaluator pool and into
    restart workers.

    * `MSE` is the raw weighted MSE — the same quantity scipy minimises, no log transform.
    * A diverging simulation returns NaN instead of raising: Xopt then drops that row when
      fitting the GP rather than aborting the run.
    * The `beamOptimizer` is built on first use in each process and cached; `__getstate__`
      drops the cache (bound methods + beamline objects) so pickling still works.
    * The particle bunch is carried as an attribute, never read from the module global —
      see note 1 in the module docstring.
    """

    def __init__(self, seg_var, obj, beamline_len, bounds=CURRENT_BOUNDS, particles=None):
        self.seg_var = seg_var
        self.obj = obj
        self.beamline_len = beamline_len
        self.bounds = tuple(bounds)
        self.particles = PARTICLES if particles is None else particles
        self.var_names = list(dict.fromkeys(v[0] for v in seg_var.values()))
        self.out_names = measured_names(obj)
        self._opti = None
        self._order = None
        self._flat_goals = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_opti"] = state["_order"] = state["_flat_goals"] = None
        return state

    def _ensure_built(self, beamline=None):
        """Build the beamOptimizer once per process (or bind it to a caller's beamline)."""
        if self._opti is not None:
            return
        bl = beamline if beamline is not None else ExcelElements(EXCEL_PATH).create_beamline()
        self._opti = beamOptimizer(bl[: self.beamline_len], self.particles.clone())
        start_point = {n: {"bounds": self.bounds, "start": 1.0} for n in self.var_names}
        # _prepare() wires up the objectives and the internal variable order without optimising
        self._opti._prepare(self.seg_var, start_point, copy.deepcopy(self.obj))
        self._order = list(self._opti.variablesToOptimize)   # set-derived -> arbitrary order
        self._flat_goals = [g for e in self._opti.objectives for g in self._opti.objectives[e]]

    def __call__(self, inputs: dict) -> dict:
        out = {"MSE": np.nan, **{n: np.nan for n in self.out_names}}
        try:
            self._ensure_built()
            x = np.array([float(inputs[n]) for n in self._order])   # names -> internal order
            mse = float(self._opti._optiSpeed(x))
            if np.isfinite(mse):
                out["MSE"] = mse
                for name, g in zip(self.out_names, self._flat_goals):
                    val = g["measured"]
                    out[name] = float(val.item() if hasattr(val, "item") else val)
        except Exception:      # noqa: BLE001 - a failed point is NaN, never a dead run
            pass
        return out


# ── The BO loop (one restart), batch-capable ─────────────────────────
def run_xopt_bo_parallel(objective, n_init, n_steps, seed=SEED, bounds=CURRENT_BOUNDS,
                         EPSILON=EPSILON, q=1, executor=None, beamline=None,
                         verbose=False, print_every=5, label=""):
    """`Xopt_utils.run_xopt_bo` plus AXIS 2: `q > 1` evaluates q candidates per step.

    With `q > 1` (optionally re-using a caller-supplied `executor`) `Xopt.step()` requests
    `evaluator.max_workers` candidates from the EI generator and the evaluator's pool
    simulates them concurrently. The number of steps is divided by q so the *evaluation*
    budget stays `n_init + n_steps` however the work is scheduled.

    Returns (X, stats) — `stats` carries the same keys as `Xopt_utils.run_xopt_bo` plus the
    raw / best-so-far MSE curves.
    """
    if beamline is not None:            # reuse the caller's beamline (Scenario B hands one over)
        objective._ensure_built(beamline)

    vocs = VOCS(variables={n: list(bounds) for n in objective.var_names},
                objectives={"MSE": "MINIMIZE"})
    ev_kwargs = {"function": objective, "max_workers": q}
    if executor is not None:
        ev_kwargs["executor"] = executor
    X = Xopt(evaluator=Evaluator(**ev_kwargs),
             generator=ExpectedImprovementGenerator(vocs=vocs), vocs=vocs)

    torch.manual_seed(seed)             # reproducible GP fits / acquisition optimisation
    t0 = time.perf_counter()
    X.random_evaluate(n_init, seed=seed)                    # phase 1: random exploration
    n_fallback = 0
    steps = int(np.ceil(n_steps / max(1, q)))               # keep the evaluation budget constant
    for i in range(steps):                                  # phase 2: fit GP -> EI -> evaluate
        try:
            X.step()
        except Exception:               # noqa: BLE001 - degenerate GP fit: random point instead
            n_fallback += 1
            X.random_evaluate(max(1, q), seed=seed + 1000 + i)
        if verbose and print_every and (i + 1) % print_every == 0:
            print(f"      {label:18s} BO step {i + 1:3d}/{steps}   "
                  f"best MSE = {X.data['MSE'].min():.3e}")
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
    else:                               # every evaluation failed
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
        "best_mse_curve": [float(v) for v in np.minimum.accumulate(
            np.nan_to_num(mse, nan=np.inf))],
    }
    return X, stats


# ── Workers (module level so they survive pickling into a spawned process) ──
def _worker_bo_restart(task, particles, EPSILON=EPSILON, q=1):
    """AXIS 1 for a single-stage scenario (A or C): one independent BO restart per process.

    The scenario config travels *inside the task*: a spawned worker re-imports this module
    from disk, so anything the caller changed at runtime would otherwise never reach it.

    Returns (record, X.data, ok) — exceptions are returned, never raised, so one bad restart
    cannot take the pool down.
    """
    scenario, cfg, run_idx, n_init, n_steps, seed = task
    try:
        _enter_worker()
        objective = FELsimObjective(cfg["vars"], cfg["obj"], cfg["beamline_len"],
                                    bounds=cfg["bounds"], particles=particles)
        X, st = run_xopt_bo_parallel(objective, n_init, n_steps,
                                     seed=seed + run_idx * 997, bounds=cfg["bounds"],
                                     EPSILON=EPSILON, q=q,
                                     label=f"{scenario} run{run_idx + 1}")
        rec = {
            "run": run_idx + 1, "method": XOPT_TAG, "scenario": scenario,
            "method_tag": XOPT_TAG, "final_mse": st["final_mse"],
            "nfev": st["nfev"], "nit": st["nfev"], "wall_time": st["wall_time"],
            "success": st["success"], "n_to_eps": st["n_to_eps"],
            "n_failed": st["n_failed"], "n_fallback": st["n_fallback"],
            "best_eval": st["best_eval"],
            "mse_curve": st["mse_curve"], "best_mse_curve": st["best_mse_curve"],
            "result_x": st["best_x"], **st["best_x"], **st["measured"],
        }
        return rec, X.data.copy(), True
    except Exception as exc:  # noqa: BLE001
        return ({"run": run_idx + 1, "method": XOPT_TAG, "scenario": scenario,
                 "method_tag": XOPT_TAG, "final_mse": np.nan, "error": str(exc)},
                pd.DataFrame(), False)


def _worker_bo_B_pipeline(task, particles, EPSILON=EPSILON, q=1):
    """AXIS 1 for Scenario B: one full 11-stage BO pipeline in its own process.

    Stages stay sequential — each stage's best currents are written back into this task's
    private beamline before the next stage's BO run starts (and stage 5 mirrors its currents
    onto elements 39/41/43, exactly as `Xopt_utils.run_scenario_B_xopt` does).

    Returns (stage_records, {stage index: X.data}, final_currents, ok).
    """
    stages, run_idx, n_init, n_steps, seed, bounds = task
    stage_records, stage_data = [], {}
    executor = None
    try:
        _enter_worker()
        bl_run = ExcelElements(EXCEL_PATH).create_beamline()   # private beamline per task
        if q > 1:
            # one pool for the whole pipeline instead of one per stage — 11 pools would pay
            # the ~25 s interpreter start-up eleven times over
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=q)

        for s_i, stage in enumerate(stages):
            lbl, seg_var, obj, bl_len = stage[0], stage[1], stage[2], stage[3]
            objective = FELsimObjective(seg_var, obj, bl_len, bounds=bounds,
                                        particles=particles)
            X, st = run_xopt_bo_parallel(objective, n_init, n_steps,
                                         seed=seed + run_idx * 997 + s_i, bounds=bounds,
                                         EPSILON=EPSILON, q=q, executor=executor,
                                         beamline=bl_run, label=lbl)
            stage_data[s_i + 1] = X.data.copy()

            # 🎯 hand-off: the best currents become the physical state for the next stage
            if all(np.isfinite(v) for v in st["best_x"].values()):
                for elem_idx, var_info in seg_var.items():
                    bl_run[elem_idx].current = float(st["best_x"][var_info[0]])
                if s_i == 4:                                   # Stage 5 symmetry
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

        # only the currents matter downstream; beamline objects are heavy to ship back
        finals = {elem_idx: float(bl_run[elem_idx].current)
                  for stage in stages for elem_idx in stage[1]}
        if len(stages) > 4:            # the mirrored elements only exist once stage 5 has run
            finals.update({i: float(bl_run[i].current) for i in (39, 41, 43)})
        return stage_records, stage_data, finals, True
    except Exception as exc:  # noqa: BLE001
        return stage_records, stage_data, {"error": str(exc)}, False
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


# ── Reporting helpers ────────────────────────────────────────────────
def report_speedup(scenario, records, elapsed, n_workers, mode="restarts", derivable=True):
    """Speedup estimated as (summed per-task BO time) / (elapsed wall time).

    Only meaningful when the tasks ran *concurrently* (axis 1). Inside a single batched run
    (axis 2) the parallelism happens below `wall_time`, so summing it would just return the
    elapsed time and print a meaningless 1.00x — report throughput instead.

    Like `parallel_utils.report_speedup` it is a rough estimate, biased both ways: it ignores
    the ~25 s worker start-up (understating the gain) while the per-task timings were measured
    inside the loaded pool and already contain CPU contention (overstating it). This host has
    6 physical cores behind 12 logical ones, so "efficiency" past ~6 workers is optimistic.
    """
    n_eval = int(np.nansum([r.get("nfev", np.nan) for r in records]))
    if not derivable:
        print(f"\n⚡ Scenario {scenario} [{mode}]: {n_eval} FELsim evaluations, "
              f"{n_workers} processes\n"
              f"   elapsed wall time         : {elapsed:8.1f} s\n"
              f"   throughput                : {n_eval / elapsed:8.2f} evaluations/s\n"
              f"   (speedup cannot be derived from one batched run)")
        return {"summed_task_time_s": np.nan, "elapsed_s": elapsed, "speedup": np.nan,
                "evals_per_s": n_eval / elapsed if elapsed else np.nan,
                "n_evaluations": n_eval, "n_workers": n_workers, "mode": mode}

    serial = float(np.nansum([r.get("wall_time", np.nan) for r in records]))
    speedup = serial / elapsed if elapsed > 0 else float("nan")
    print(f"\n⚡ Scenario {scenario} [{mode}]: {len(records)} tasks on {n_workers} processes\n"
          f"   BO time summed over tasks : {serial:8.1f} s  (serial equivalent)\n"
          f"   elapsed wall time         : {elapsed:8.1f} s\n"
          f"   speedup ~= {speedup:.2f}x   (rough estimate — "
          f"{100 * speedup / max(1, n_workers):.0f}% of {n_workers} processes)")
    return {"speedup": speedup, "summed_task_time_s": serial, "elapsed_s": elapsed,
            "n_evaluations": n_eval, "n_workers": n_workers, "mode": mode}


def summarise_xopt_restarts(scenario, records, VARS, OBJ, EPSILON, REF_CURRENTS, verbose=True):
    """One row per restart + a summary shaped like `run_scenario_C_xopt`'s `final_summary`.

    The summary keeps exactly the columns `display_scenario_C_xopt_summary` expects, so the
    parallel results can be fed straight into the existing display / plotting helpers.
    """
    df = pd.DataFrame(records)
    if df.empty or "final_mse" not in df.columns:
        print(f"⚠️  Scenario {scenario}: no usable restart to summarise")
        return df, pd.DataFrame()

    current_name = list(dict.fromkeys(info[0] for info in VARS.values()))
    out_names = measured_names(OBJ)
    # a restart that raised comes back with no current / measured columns at all
    for col in (*current_name, *out_names):
        if col not in df.columns:
            df[col] = np.nan
    df["is_converged"] = df["final_mse"] < EPSILON

    conv = df.loc[df["final_mse"] < EPSILON, "final_mse"]
    best_run = df.loc[df["final_mse"].fillna(np.inf).idxmin()]      # all-NaN safe
    final_summary = pd.DataFrame([{
        "method_tag": XOPT_TAG,
        "n_runs": len(df),
        "conv_rate": float(df["is_converged"].mean()),
        "nfev_mean": float(df["nfev"].mean()) if "nfev" in df else np.nan,
        "wall_time_mean": float(df["wall_time"].mean()) if "wall_time" in df else np.nan,
        "final_mse_mean": float(10 ** np.log10(conv).mean()) if len(conv) else np.nan,
        "final_mse_best": float(best_run["final_mse"]),
        **{n: float(best_run[n]) for n in current_name},
        **{f"{info[0]}_ref": REF_CURRENTS[k] for k, info in VARS.items()},
        **{n: float(best_run[n]) for n in out_names},
    }])

    if verbose:
        print("\n" + "=" * 110)
        print(f" 🎯 SCENARIO {scenario} ({XOPT_TAG}): {len(df)} restarts, Tol < {EPSILON}")
        print("=" * 110)
        cols = ["run", "final_mse", "nfev", "wall_time", "success", "n_to_eps", *current_name]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        s = final_summary.iloc[0]
        print(f"\n conv_rate = {s['conv_rate']:.0%}   |   best MSE = {s['final_mse_best']:.3e}"
              f"   |   mean wall/restart = {s['wall_time_mean']:.1f} s")
        print(" champion currents: " + "   ".join(
            f"{n}={float(best_run[n]):.4f} (ref {REF_CURRENTS[k]:.4f})"
            for k, info in VARS.items() for n in [info[0]]))
        print("=" * 110)
    return df, final_summary


class _CurrentHolder:
    """Stand-in for a beamline element that only carries its current."""

    __slots__ = ("current",)

    def __init__(self, current):
        self.current = float(current)


def currents_to_beamline_view(finals):
    """{element index: current} -> a dict usable wherever `beamline[i].current` is read.

    The workers ship currents rather than beamline objects (those are heavy and hold
    unpicklable state), but `Xopt_utils.display_scenario_B_xopt_summary` indexes a beamline.
    This restores that interface.
    """
    return {i: _CurrentHolder(c) for i, c in finals.items()}


def plot_xopt_restarts_convergence(xopt_data, n_init, title, eps=EPSILON, ax=None):
    """Best-so-far MSE of every restart on one log axis, champion highlighted.

    The restart counterpart of `Xopt_utils.plot_scenario_A_xopt_convergence`, which shows the
    individual evaluations of a single run.
    """
    if not xopt_data:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))

    best_run, best_val = None, np.inf
    for run_id, data in sorted(xopt_data.items()):
        mse = np.nan_to_num(data["MSE"].to_numpy(dtype=float), nan=np.inf)
        if not len(mse):
            continue
        curve = np.minimum.accumulate(mse)
        ax.semilogy(np.arange(1, len(curve) + 1), curve, drawstyle="steps-post",
                    color="#9ecae1", lw=1.1, zorder=1)
        if curve[-1] < best_val:
            best_run, best_val = run_id, curve[-1]

    if best_run is not None:
        mse = np.nan_to_num(xopt_data[best_run]["MSE"].to_numpy(dtype=float), nan=np.inf)
        ev = np.arange(1, len(mse) + 1)
        ok = np.isfinite(mse)
        ax.semilogy(ev[ok], mse[ok], "o", ms=4, mfc="#9ecae1", mec="white", mew=0.7,
                    label="evaluation (best restart)", zorder=2)
        ax.semilogy(ev, np.minimum.accumulate(mse), drawstyle="steps-post", color="#1f4e79",
                    lw=2, label=f"best so far (restart {best_run})", zorder=3)
    ax.plot([], [], color="#9ecae1", lw=1.1, label="other restarts")
    ax.axhline(eps, color="#c44e52", lw=1.2, ls="--", label=f"EPSILON = {eps:g}")
    ax.axvline(n_init + 0.5, color="gray", lw=1, ls=":", label="BO starts")
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()
    return ax


# ── Parallel drivers ─────────────────────────────────────────────────
def run_scenario_single_xopt_parallel(scenario, VARS, OBJ, BEAMLINE_LEN, CURRENT_BOUNDS,
                                      EPSILON, N_RUNS, PARTICLES, REF_CURRENTS,
                                      N_INIT, N_STEPS, SEED=SEED, max_cores=DEFAULT_CORES,
                                      q=1, verbose=True):
    """Scenario A or C: `N_RUNS` independent BO restarts spread over a process pool.

    `q > 1` additionally batches each restart's BO steps (q-EI), so the pool holds
    `outer * q` processes — keep that below the core count.

    Returns (results, df, final_summary, xopt_data, perf); `results` / `df` / `final_summary`
    have the same shape as the sequential `Xopt_utils.run_scenario_C_xopt`, and
    `xopt_data[run]` is that restart's full Xopt history.
    """
    cfg = {"vars": VARS, "obj": OBJ, "beamline_len": BEAMLINE_LEN, "bounds": CURRENT_BOUNDS}
    tasks = [(scenario, cfg, i, N_INIT, N_STEPS, SEED) for i in range(N_RUNS)]

    q = max(1, min(q, max_cores))
    outer = max(1, min(max_cores // q, len(tasks)))
    mode = "restarts" if q == 1 else f"both(outer={outer},q={q})"
    if verbose:
        print(f"\n▶ 🚀 Scenario {scenario} [{XOPT_TAG}, {mode}] — {N_RUNS} BO restarts "
              f"({N_INIT} random + {N_STEPS} evaluations each) on {outer} processes"
              + (f", each batching {q} evaluations" if q > 1 else ""))

    pin_child_environment()
    worker = partial(_worker_bo_restart, particles=PARTICLES, EPSILON=EPSILON, q=q)
    records, xopt_data, n_failed = [], {}, 0
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=outer) as executor:
        for done, (rec, data, ok) in enumerate(executor.map(worker, tasks), start=1):
            records.append(rec)
            if ok:
                xopt_data[rec["run"]] = data
                if verbose:
                    print(f"   [{done}/{N_RUNS}] run {rec['run']:3d}  "
                          f"MSE={rec['final_mse']:.3e}  nfev={rec['nfev']:<4} "
                          f"t={rec['wall_time']:.1f}s  (failed evals: {rec['n_failed']}, "
                          f"random fallbacks: {rec['n_fallback']})")
            else:
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ run {rec.get('run')} failed: {rec.get('error')}")
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"✓ Scenario {scenario} finished in {elapsed:.2f}s ({n_failed} failed)")
    df, final_summary = summarise_xopt_restarts(scenario, records, VARS, OBJ, EPSILON,
                                                REF_CURRENTS, verbose=verbose)
    perf = report_speedup(scenario, records, elapsed, outer * q, mode)
    return {XOPT_TAG: records}, df, final_summary, xopt_data, perf


def run_scenario_B_xopt_parallel(STAGES_B, CURRENT_BOUNDS, EPSILON, N_RUNS_B, PARTICLES,
                                 N_INIT, N_STEPS, SEED=SEED, max_cores=DEFAULT_CORES,
                                 q=1, verbose=True):
    """Scenario B: one task = one full 11-stage BO pipeline (stages sequential inside).

    Returns (results, df_B, final_currents, xopt_data, perf).
    `final_currents` is a list of `{element index: current}` dicts, one per pipeline — feed one
    through `currents_to_beamline_view()` for `display_scenario_B_xopt_summary`.
    `xopt_data[(run, stage)]` is that stage's full Xopt history, matching the sequential
    `run_scenario_B_xopt` so `plot_scenario_B_xopt_convergence` works unchanged.
    """
    tasks = [(STAGES_B, i, N_INIT, N_STEPS, SEED, CURRENT_BOUNDS) for i in range(N_RUNS_B)]

    q = max(1, min(q, max_cores))
    outer = max(1, min(max_cores // q, len(tasks)))
    mode = "restarts" if q == 1 else f"both(outer={outer},q={q})"
    if verbose:
        print(f"\n▶ 🚀 Scenario B ({len(STAGES_B)}-stage sequential) [{XOPT_TAG}, {mode}] — "
              f"{N_RUNS_B} pipelines ({N_INIT} random + {N_STEPS} evaluations per stage) "
              f"on {outer} processes")
        print("   stages inside a pipeline stay sequential (current hand-off)"
              + (f"   |   each stage batches {q} evaluations" if q > 1 else ""))

    pin_child_environment()
    worker = partial(_worker_bo_B_pipeline, particles=PARTICLES, EPSILON=EPSILON, q=q)
    all_records, xopt_data, finals, n_failed = [], {}, [], 0
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=outer) as executor:
        for done, (recs, data, fin, ok) in enumerate(executor.map(worker, tasks), start=1):
            all_records.extend(recs)
            for stage_idx, stage_df in data.items():
                xopt_data[(done, stage_idx)] = stage_df
            if ok:
                finals.append(fin)
            else:
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ pipeline {done} failed after {len(recs)} stages: "
                          f"{fin.get('error')}")
            if verbose:
                n_conv = sum(1 for r in recs if r["final_mse"] < EPSILON)
                print(f"   [{done}/{N_RUNS_B}] {len(recs)}/{len(STAGES_B)} stages, "
                      f"{n_conv} below EPSILON")
    elapsed = time.perf_counter() - t0

    df_B = pd.DataFrame(all_records)
    if verbose and not df_B.empty:
        print(f"\n✓ Scenario B finished in {elapsed:.2f}s ({n_failed} failed pipelines)")
        print(f"  stages with MSE < {EPSILON:g}: "
              f"{int((df_B['final_mse'] < EPSILON).sum())}/{len(df_B)}")
        print("  best MSE per stage across pipelines:")
        for (stage, name), val in df_B.groupby(["stage", "stage_name"])["final_mse"].min().items():
            print(f"    {'✓' if val < EPSILON else ' '} {stage:2d} {name:22s} {val:.3e}")
    perf = report_speedup("B", all_records, elapsed, outer * q, mode)
    return {XOPT_TAG: all_records}, df_B, finals, xopt_data, perf


def run_batch_single_xopt(scenario, VARS, OBJ, BEAMLINE_LEN, CURRENT_BOUNDS, EPSILON,
                          PARTICLES, REF_CURRENTS, N_INIT, N_STEPS, SEED=SEED,
                          max_cores=DEFAULT_CORES, q=4, run_idx=0, verbose=True):
    """AXIS 2 only: ONE BO run whose every step evaluates `q` candidates concurrently (q-EI).

    This is Xopt's own parallelism — `Evaluator(max_workers=q)` owns the pool and
    `Xopt.step()` requests exactly `q` candidates per iteration. Use it when a single run is
    what you need (no restart statistics) and the simulation dominates the GP fit.
    """
    q = max(1, min(q, max_cores))
    if verbose:
        print(f"\n▶ 🚀 Scenario {scenario} [{XOPT_TAG}, batch q={q}] — one BO run, "
              f"{N_INIT} random + ~{N_STEPS} evaluations in batches of {q}")

    pin_child_environment()
    objective = FELsimObjective(VARS, OBJ, BEAMLINE_LEN, bounds=CURRENT_BOUNDS,
                                particles=PARTICLES)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=q) as executor:
        X, st = run_xopt_bo_parallel(objective, N_INIT, N_STEPS, seed=SEED + run_idx * 997,
                                     bounds=CURRENT_BOUNDS, EPSILON=EPSILON, q=q,
                                     executor=executor, verbose=verbose,
                                     label=f"{scenario} batch")
    elapsed = time.perf_counter() - t0

    rec = {"run": run_idx + 1, "method": XOPT_TAG, "scenario": scenario,
           "method_tag": XOPT_TAG, "final_mse": st["final_mse"], "nfev": st["nfev"],
           "nit": st["nfev"], "wall_time": st["wall_time"], "success": st["success"],
           "n_to_eps": st["n_to_eps"], "n_failed": st["n_failed"],
           "n_fallback": st["n_fallback"], "best_eval": st["best_eval"],
           "mse_curve": st["mse_curve"], "best_mse_curve": st["best_mse_curve"],
           "result_x": st["best_x"], **st["best_x"], **st["measured"]}
    if verbose:
        print(f"✓ best MSE = {st['final_mse']:.3e} after {st['nfev']} evaluations "
              f"in {elapsed:.1f}s")
    df, final_summary = summarise_xopt_restarts(scenario, [rec], VARS, OBJ, EPSILON,
                                                REF_CURRENTS, verbose=verbose)
    perf = report_speedup(scenario, [rec], elapsed, q, f"batch(q={q})", derivable=False)
    return {XOPT_TAG: [rec]}, df, final_summary, {1: X.data.copy()}, perf
