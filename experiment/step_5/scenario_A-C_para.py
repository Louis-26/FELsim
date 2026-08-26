"""
Multi-core driver for the Scenario A / B / C matching problems of `scenario_A-C.ipynb`.

The optimisation itself is untouched: every task still calls the same
`beamOptimizer.calc()` -> `scipy.optimize.minimize` as the notebook, with the same
variables, objectives, bounds, method options and reference currents. Only the *scheduling*
changes.

WHY THIS IS PARALLELISABLE
--------------------------
Each of the `N_RUNS` random starts is a completely independent optimisation: it builds its
own beamline (`ExcelElements(EXCEL_PATH).create_beamline()`), clones its own particles, and
never reads or writes another run's state. The same is true across methods (Nelder-Mead,
L-BFGS-B, ...). So the work decomposes into `len(METHODS) * N_RUNS` embarrassingly parallel
tasks that can be spread over CPU cores with a process pool:

    Scenario A :  tasks = methods x random starts          (fully parallel)
    Scenario C :  tasks = methods x random starts          (fully parallel)
    Scenario B :  tasks = methods x random starts          (parallel ACROSS pipelines,
                  but the 11 stages INSIDE one pipeline stay sequential — stage k+1 starts
                  from the currents stage k wrote back, so they cannot be split)

Processes, not threads: the objective spends its time in NumPy/PyTorch matrix products that
hold the GIL for long stretches, and each run mutates its own beamline objects, so separate
address spaces are both faster and safer.

HOW MUCH SPEEDUP TO EXPECT
--------------------------
With `P` worker processes and `T` tasks, the wall time is roughly

    t_parallel  ~=  t_startup  +  ceil(T / P) * t_task          (tasks of similar cost)
    speedup     ~=  T * t_task / t_parallel      ->  min(P, T)  when t_task >> t_startup

Two things eat into the ideal `min(P, T)`:

1. **Worker start-up.** Windows/macOS spawn a fresh interpreter per worker, and importing
   `experiments_utils` costs ~25 s (PyTorch + reading the beamline Excel + generating the
   1000-particle bunch). A *pool* pays this once per worker, not once per task, so it is
   amortised over `T/P` tasks — negligible for B and C, visible for a short Scenario A run.
2. **Load imbalance.** Task costs differ a lot per method. Measured in `scenario_A-C.ipynb`
   (1 run each): Scenario A — SLSQP 7 s, L-BFGS-B 9 s, trust-constr 10 s, Nelder-Mead 14 s,
   COBYLA 77 s; Scenario C — SLSQP 46 s, L-BFGS-B 1602 s, COBYLA 3198 s, Nelder-Mead 3443 s.
   The pool hands out tasks dynamically so this mostly self-balances, but the makespan can
   never beat the single longest task (Amdahl): e.g. Scenario C with 4 methods x 1 run on
   >= 4 cores still takes >= 3443 s.

MEASURED on this machine (12 cores, Windows, felsim env) — small runs, so the ~25 s worker
start-up is still a large share of the wall time:

    Scenario A  5 methods x 2 runs  = 10 tasks on  4 cores :  172 s serial-equivalent /  68 s  -> 2.5x (64% eff.)
    Scenario B  1 method  x 2 runs  =  2 pipelines on 2 cores : 229 s / 141 s               -> 1.6x (81% eff.)
    Scenario C  1 method  x 2 runs  =  2 tasks on  2 cores :   79 s /  52 s                 -> 1.5x (76% eff.)

Extrapolating to production-sized runs, where start-up is amortised away:

    Scenario A, 5 methods x 100 runs  (500 tasks, ~23 s avg)  ~3.2 h serial  ->  ~20 min on 11 cores
    Scenario B, 5 methods x 1  run    (5 tasks, ~800 s each)  ~1.1 h serial  ->  ~15 min (limited by T=5)
    Scenario C, 4 methods x 1  run    (4 tasks, up to 3443 s) ~2.3 h serial  ->  ~1 h   (limited by the longest task)

The rule of thumb: **parallelism helps in proportion to how many independent random starts
you ask for.** Raising `N_RUNS` costs almost nothing in wall time until `T` exceeds `P` —
which is exactly what makes the `N_RUNS_A = 1000` robustness study in the notebook affordable.

Each driver prints `sum(per-task optimiser time) / elapsed wall time` as a **CPU-time ratio** —
a rough estimate only. It ignores worker start-up (understating the gain) while its per-task
timings already include pool contention (overstating it), and this host has just 6 physical
cores behind its 12 logical ones. `--benchmark` re-runs the identical workload one task at a
time and prints the measured speedup (a 10-task Scenario A run: ratio 1.97x, measured 2.11x).

Reproducibility: every task runs inside a worker process with `PYTHONHASHSEED=0` and receives
the parent's particle bunch, so a given `run_idx` means the same random start and the same
beam in every process and across invocations (see the notes in section 0 and the workers).

USAGE
-----
    python scenario_A-C_para.py --scenario A --n_runs 100 --max_cores 11
    python scenario_A-C_para.py --scenario B --n_runs 4
    python scenario_A-C_para.py --scenario C --n_runs 8 --save_dir ../../results/para
    python scenario_A-C_para.py --scenario all --n_runs 2 --max_cores 4
    python scenario_A-C_para.py --scenario A --n_runs 8 --benchmark    # serial vs parallel

Run it from `experiment/step_5/` (or anywhere — the script chdir's to its own directory,
which is what `experiments_utils` needs to resolve `../../backend` and `../../beam_excel`).
"""

# ── 0. Thread pinning ────────────────────────────────────────────────
# MUST happen before NumPy / PyTorch are imported, otherwise their thread pools are already
# sized and these variables have no effect. One thread per process: P processes each running
# a multi-threaded BLAS would oversubscribe the cores and run *slower* than the serial code.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

# Spawned children inherit this environment. `run_benchmark` draws its random start vector
# by iterating `list({seg_var[i][0] for i in seg_var})` — a *set of strings*, whose order
# depends on Python's per-interpreter hash randomisation. Pinning the seed makes every worker
# assign the same draw to the same variable, so a given run_idx means the same start point in
# every process and across invocations. (All tasks run inside workers for exactly this reason:
# the parent was started without PYTHONHASHSEED and cannot change its own hash order.)
os.environ["PYTHONHASHSEED"] = "0"

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
from functools import partial
import concurrent.futures

import numpy as np
import pandas as pd
import torch

import experiments_utils
from experiments_utils import (
    PARTICLES,
    EXCEL_PATH,
    ExcelElements,
    beamOptimizer,
    run_benchmark,
    results_to_df,
    method_label,
    ALPHA_XM,
    ALPHA_YM,
    BETA_XM,
    BETA_YM,
)
from configs import FELSIM_S1_CURRENTS, METHOD_OPTIONS

torch.set_num_threads(1)  # belt and braces: also caps PyTorch's intra-op pool per worker

# ── 2. Shared experiment configuration ───────────────────────────────
CURRENT_BOUNDS = (0.01, 1.5)
EPSILON = 1e-3
SEED = 42

# how many cores to use by default: leave one for the OS / the launching shell
try:
    DEFAULT_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - 1
except TypeError:
    DEFAULT_CORES = 4
DEFAULT_CORES = max(1, DEFAULT_CORES)

METHODS_A = [
    ("Nelder-Mead", None),
    ("L-BFGS-B", "2-point"),
    ("SLSQP", "2-point"),
    ("COBYLA", None),
    ("trust-constr", "2-point"),
]
METHODS_B = list(METHODS_A)
METHODS_C = [
    ("Nelder-Mead", None),
    ("L-BFGS-B", "2-point"),
    ("SLSQP", "2-point"),
    ("COBYLA", None),
]


def identity_func(x):
    """Top-level (picklable) replacement for the notebook's `lambda x: x`.

    The variable specs travel to the worker processes, and a lambda / closure cannot be
    pickled — every variable transform in this file must therefore be a module-level def.
    """
    return x


def _v(name):
    """Variable spec understood by beamOptimizer: [name, attribute, transform]."""
    return [name, "current", identity_func]


# ── 3. Scenario definitions (verbatim from scenario_A-C.ipynb) ───────
A_BEAMLINE_LEN = 10
A_VARS = {
    1: _v("I"),
    3: _v("I2"),
}
A_OBJ = {
    8: [{"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0}],
    9: [{"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}],
}

C_BEAMLINE_LEN = 118
C_VARS = {
    56: _v("I_56"),
    58: _v("I_58"),
    61: _v("I_61"),
    63: _v("I_63"),
    76: _v("I_76"),
    78: _v("I_78"),
    80: _v("I_80"),
    87: _v("I_87"),
    93: _v("I_93"),
    95: _v("I_95"),
    97: _v("I_97"),
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

# (label, segment variables, objectives, beamline slice length, fixed start points)
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
        {
            "I": {"bounds": CURRENT_BOUNDS, "start": 1},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 1},
        },
    ),
    (
        "Stage 2 Chrom.1",
        {10: _v("I")},
        {15: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        16,
        {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
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
        {
            "I": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 5},
            "I3": {"bounds": CURRENT_BOUNDS, "start": 3},
        },
    ),
    (
        "Stage 4 Chrom.2",
        {27: _v("I")},
        {32: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        33,
        {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
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
        {
            "I": {"bounds": CURRENT_BOUNDS, "start": 0.28},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 2.65},
            "I3": {"bounds": CURRENT_BOUNDS, "start": 2.69},
        },
    ),
    (
        "Stage 6 Chrom.3",
        {50: _v("I")},
        {55: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        56,
        {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
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
        {
            "I": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
        },
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
        {
            "I": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
        },
    ),
    (
        "Stage 9 Chrom.4",
        {70: _v("I")},
        {75: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
        76,
        {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
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
        {
            "I": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I3": {"bounds": CURRENT_BOUNDS, "start": 2},
        },
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
        {
            "Ic": {"bounds": CURRENT_BOUNDS, "start": 4},
            "I": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
            "I3": {"bounds": CURRENT_BOUNDS, "start": 2},
        },
    ),
]

# Single-stage scenarios, looked up by name inside the workers. Keeping the (unpicklable-ish,
# heavy) specs in module state means each task only has to ship a small tuple.
SCENARIOS = {
    "A": {"vars": A_VARS, "obj": A_OBJ, "beamline_len": A_BEAMLINE_LEN, "methods": METHODS_A},
    "C": {"vars": C_VARS, "obj": C_OBJ, "beamline_len": C_BEAMLINE_LEN, "methods": METHODS_C},
}


# ── 4. Workers (top-level so they survive pickling to a spawned process) ──
def _worker_single_stage(task, particles, use_log=False, use_epsilon=1e-13, noise=False,
                         sigma=None):
    """One (method, random start) task of a single-stage scenario (A or C).

    The scenario config travels *inside the task* rather than being read from module state:
    a spawned worker re-imports this module from disk, so anything a caller changed at
    runtime (a tweaked A_OBJ, a shorter beamline) would otherwise never reach the workers.
    It stays picklable because every variable transform is the module-level `identity_func`.

    `particles` is passed in for the same reason: `experiments_utils.PARTICLES` is drawn at
    import time from the *unseeded* global torch RNG, so a worker that used the module global
    would optimise its own private beam sample and the tasks would no longer be repetitions
    of one problem.

    Returns (method_name, record, ok). Exceptions are returned, never raised, so one bad
    task cannot take the whole pool down.
    """
    method_name, run_idx = "?", -1
    try:
        scenario, cfg, method, jac, method_name, run_idx = task
        bounds = {info[0]: CURRENT_BOUNDS for info in cfg["vars"].values()}
        res = run_benchmark(
            scenario,
            cfg["beamline_len"],
            particles,
            cfg["vars"],
            cfg["obj"],
            bounds,
            method=method,
            n_runs=1,
            SEED=SEED,
            seed_offset=run_idx,  # <- makes every task draw a different random start
            jac=jac,
            options=METHOD_OPTIONS.get(method_name, None),
            method_tag=method_name,
            use_log=use_log,
            use_epsilon=use_epsilon,
            noise=noise,
            sigma=sigma,
            verbose=False,
        )
        rec = res[0]
        rec["method_tag"] = method_name
        rec["run"] = run_idx + 1
        return method_name, rec, True
    except Exception as exc:  # noqa: BLE001 - report, don't kill the pool
        return method_name, {"method_tag": method_name, "run": run_idx + 1,
                             "final_mse": np.nan, "error": str(exc)}, False


def _stage_start_points(fixed_sp, rng, randomize):
    """Start point for one Scenario B stage: the notebook's fixed value, or a random draw.

    NOTE: `scenario_B_utils.run_scenario_B` always uses the fixed `start` values baked into
    STAGES_B, which makes every repetition of the pipeline byte-identical — so repeating it
    only pays off once the starts actually differ. With `randomize=True` each stage draws its
    start uniformly inside the bounds, exactly like `run_benchmark` does for A and C.
    """
    sp = copy.deepcopy(fixed_sp)
    if randomize:
        for var, spec in sp.items():
            lo, hi = spec["bounds"]
            spec["start"] = float(rng.uniform(lo, hi))
    return sp


def _worker_B_pipeline(task, particles, randomize_starts=True):
    """One (method, random start) task of Scenario B: the whole 11-stage pipeline.

    The stages inside run sequentially — each one writes its optimised currents back into
    this task's private beamline before the next stage reads it — so this function is the
    smallest unit that can be handed to a core. As above, the stage list travels inside the
    task so that a caller-supplied `stages` actually reaches the spawned workers.

    Returns (method_name, stage_records, final_currents, ok).
    """
    stages, method, jac, method_name, run_idx = task
    options = METHOD_OPTIONS.get(method_name, None)
    rng = np.random.default_rng(SEED + run_idx * 997)
    stage_records = []
    try:
        bl_run = ExcelElements(EXCEL_PATH).create_beamline()  # private beamline per task
        p_run = particles.clone()  # the parent's bunch, not this worker's own draw

        for s_i, (lbl, seg_var, obj, bl_len, fixed_sp) in enumerate(stages):
            sp = _stage_start_points(fixed_sp, rng, randomize_starts)
            obj_copy = copy.deepcopy(obj)
            opti = beamOptimizer(bl_run[:bl_len], p_run.clone())
            t0 = time.perf_counter()
            res = opti.calc(
                method, seg_var, sp, obj_copy, jac=jac, options=options,
                printResults=False, plotProgress=False,
            )
            wall = time.perf_counter() - t0

            # hand-off: optimised currents become the physical state for the next stage
            for elem_idx, var_info in seg_var.items():
                var_idx = opti.variablesToOptimize.index(var_info[0])
                setattr(bl_run[elem_idx], "current", float(res.x[var_idx]))
            if s_i == 4:  # Stage 5 symmetry, as in scenario_B_utils.run_scenario_B
                bl_run[43].current = bl_run[33].current
                bl_run[41].current = bl_run[35].current
                bl_run[39].current = bl_run[37].current

            nfev = int(getattr(res, "nfev", len(opti.plotMSE)))
            nit = int(getattr(res, "nit", nfev))
            if nit < 0:
                nit = nfev
            stage_records.append({
                "run": run_idx + 1,
                "stage": s_i + 1,
                "stage_name": lbl,
                "method_tag": method_name,
                "final_mse": float(res.fun),
                "nfev": nfev,
                "nit": nit,
                "wall_time": round(wall, 3),
                "success": bool(res.success),
                "start_x": {v: sp[v]["start"] for v in sp},
            })

        # only the currents matter downstream; beamline objects are heavy to ship back
        finals = {
            elem_idx: float(bl_run[elem_idx].current)
            for _, seg_var, _, _, _ in stages
            for elem_idx in seg_var
        }
        if len(stages) > 4:  # the mirrored elements only exist once stage 5 has run
            finals.update({i: float(bl_run[i].current) for i in (39, 41, 43)})
        return method_name, stage_records, finals, True
    except Exception as exc:  # noqa: BLE001
        return method_name, stage_records, {"error": str(exc)}, False


# ── 5. Aggregation & reporting helpers ───────────────────────────────
def _report_speedup(scenario, records, elapsed, n_workers, key="wall_time"):
    """CPU-time ratio: (summed per-task optimiser time) / (elapsed wall time).

    A rough estimate of the speedup, biased in both directions, so do not quote it as the
    speedup itself:
      * it ignores interpreter start-up (~25 s per worker), which *understates* the gain;
      * each `wall_time` was measured inside the loaded pool, so CPU contention and all-core
        clock throttling are baked into it, which *overstates* the gain — and the more
        workers compete, the more the ratio drifts towards `n_workers` on its own. This host
        has 6 physical cores behind 12 logical ones, so past ~6 workers the ratio keeps
        climbing while real throughput does not.
    `--benchmark` re-runs the identical workload one task at a time and reports the measured
    number (for reference, one 10-task Scenario A run: ratio 1.97x, measured 2.11x).
    """
    serial = float(np.nansum([r.get(key, np.nan) for r in records]))
    ratio = serial / elapsed if elapsed > 0 else float("nan")
    print(
        f"\n⚡ Scenario {scenario}: {len(records)} tasks on {n_workers} workers\n"
        f"   optimiser time summed over tasks : {serial:8.1f} s\n"
        f"   elapsed wall time                : {elapsed:8.1f} s\n"
        f"   CPU-time ratio ~= {ratio:.2f}x   (rough estimate only — run --benchmark for a "
        f"measured speedup)"
    )
    return {"cpu_time_ratio": ratio, "summed_task_time_s": serial, "elapsed_s": elapsed,
            "n_workers": n_workers}


def _summarise_single_stage(scenario, results, VARS, OBJ, EPSILON, verbose=True):
    """Per-method summary table — same columns/format as experiments_utils.run_scenario_C."""
    current_name = list(dict.fromkeys([info[0] for info in VARS.values()]))
    ref_I_name = [n + "_ref" for n in current_name]
    evalPos_parameter = [
        str(i) + "_" + "_".join(reversed(j[0]["measure"])) for i, j in OBJ.items()
    ]
    flat = [r for v in results.values() for r in v]
    df = results_to_df(flat, current_name, evalPos_parameter)
    if df.empty or "final_mse" not in df.columns:
        print(f"⚠️  Scenario {scenario}: no usable task result, nothing to summarise")
        return df, pd.DataFrame()

    # A run that raised inside run_benchmark comes back as final_mse=NaN with no current /
    # measured columns at all; backfill them so the champion lookup below cannot KeyError.
    for col in (*current_name, *evalPos_parameter):
        if col not in df.columns:
            df[col] = np.nan

    if "nit" in df.columns and "nfev" in df.columns:
        df.loc[df["nit"] < 0, "nit"] = df.loc[df["nit"] < 0, "nfev"]
    df["is_converged"] = df["final_mse"] < EPSILON

    def _geom_mean_converged(s):
        conv = s[s < EPSILON]
        return 10 ** np.log10(conv).mean() if len(conv) else np.nan

    summary_stats = df.groupby("method_tag").agg(
        conv_rate=("is_converged", "mean"),
        nit_mean=("nit", "mean"),
        nfev_mean=("nfev", "mean"),
        wall_time_mean=("wall_time", "mean"),
    ).join(
        df.groupby("method_tag")["final_mse"].apply(_geom_mean_converged).rename("final_mse_mean")
    )

    idx_best = df.groupby("method_tag")["final_mse"].idxmin().dropna()   # NaN-only methods drop out
    best_cols = ["method_tag", *current_name, *evalPos_parameter]
    best_runs = (df.loc[idx_best, best_cols].set_index("method_tag") if len(idx_best)
                 else pd.DataFrame(columns=best_cols).set_index("method_tag"))
    final_summary = summary_stats.join(best_runs).reset_index()
    for k, var_info in VARS.items():
        final_summary[f"{var_info[0]}_ref"] = FELSIM_S1_CURRENTS[k]

    hybrid = [x for pair in zip(current_name, ref_I_name) for x in pair]
    final_summary = final_summary[[
        "method_tag", "conv_rate", "nit_mean", "nfev_mean", "wall_time_mean",
        "final_mse_mean", *hybrid, *evalPos_parameter,
    ]]

    if verbose:
        fmt = {
            "conv_rate": "{:.0%}", "nit_mean": "{:.1f}", "nfev_mean": "{:.1f}",
            "wall_time_mean": "{:.3f} s", "final_mse_mean": "{:.2e}",
            **{c: "{:.4f} A" for c in hybrid},
            **{e: "{:.4f}" for e in evalPos_parameter},
        }
        shown = final_summary.copy()
        for col, f in fmt.items():
            if col in shown.columns:
                shown[col] = shown[col].apply(lambda x, _f=f: _f.format(x) if pd.notna(x) else "NaN")
        print("\n" + "=" * 125)
        print(f" 🎯 SCENARIO {scenario}: Robustness (Tol < {EPSILON}) & Discovered Physics State vs Reference")
        print("=" * 125)
        print(shown.to_string(index=False, justify="center"))
        print("=" * 125)
    return df, final_summary


# ── 6. Parallel drivers ──────────────────────────────────────────────
def run_single_stage_parallel(scenario, n_runs, max_cores=DEFAULT_CORES, methods=None,
                              cfg=None, use_log=False, use_epsilon=1e-13, noise=False,
                              sigma=None, verbose=True):
    """Scenario A or C: `len(methods) * n_runs` independent optimisations over a process pool.

    `cfg` overrides the built-in scenario definition ({"vars", "obj", "beamline_len"}); it is
    shipped to the workers with each task, so overrides really take effect.
    """
    cfg = cfg if cfg is not None else SCENARIOS[scenario]
    methods = methods if methods is not None else SCENARIOS[scenario]["methods"]

    tasks = []
    for spec in methods:
        method, jac = spec[0], spec[1]
        label = spec[2] if len(spec) == 3 else None
        name = method_label(method, label)
        for run_idx in range(n_runs):
            tasks.append((scenario, cfg, method, jac, name, run_idx))

    n_workers = max(1, min(max_cores, len(tasks)))
    if verbose:
        print(f"\n▶ 🚀 Scenario {scenario} — {len(methods)} methods x {n_runs} random starts "
              f"= {len(tasks)} tasks on {n_workers} cores")

    worker = partial(_worker_single_stage, particles=PARTICLES, use_log=use_log,
                     use_epsilon=use_epsilon, noise=noise, sigma=sigma)
    results, ok_records, n_failed = {}, [], 0
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for done, (name, rec, ok) in enumerate(executor.map(worker, tasks), start=1):
            if ok:
                results.setdefault(name, []).append(rec)
                ok_records.append(rec)
            # run_benchmark swallows its own exceptions and returns a NaN record carrying an
            # "error" key, so count that as a failure too — otherwise an all-broken scenario
            # would still report "0 failed".
            if not ok or rec.get("error"):
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ {name} run {rec.get('run')} failed: {rec.get('error')}")
            if verbose and done % max(1, len(tasks) // 10) == 0:
                print(f"   {done}/{len(tasks)} tasks done ({time.perf_counter() - t0:.0f}s)")
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"✓ Scenario {scenario} finished in {elapsed:.2f}s ({n_failed} failed)")
    df, final_summary = _summarise_single_stage(scenario, results, cfg["vars"], cfg["obj"],
                                                EPSILON, verbose=verbose)
    perf = _report_speedup(scenario, ok_records, elapsed, n_workers)
    return results, df, final_summary, perf


def run_scenario_B_parallel(n_runs, max_cores=DEFAULT_CORES, methods=None, stages=None,
                            randomize_starts=None, verbose=True):
    """Scenario B: one task = one full 11-stage pipeline (stages stay sequential inside)."""
    methods = methods if methods is not None else METHODS_B
    stages = stages if stages is not None else STAGES_B
    if randomize_starts is None:
        # a repeated pipeline from identical fixed starts would reproduce itself exactly
        randomize_starts = n_runs > 1

    tasks = []
    for spec in methods:
        method, jac = spec[0], spec[1]
        label = spec[2] if len(spec) == 3 else None
        name = method_label(method, label)
        for run_idx in range(n_runs):
            tasks.append((stages, method, jac, name, run_idx))

    n_workers = max(1, min(max_cores, len(tasks)))
    if verbose:
        print(f"\n▶ 🚀 Scenario B (11-stage sequential) — {len(methods)} methods x {n_runs} runs "
              f"= {len(tasks)} pipelines on {n_workers} cores")
        print(f"   start points: {'random per run' if randomize_starts else 'fixed (as in the notebook)'}"
              f"   |   stages inside a pipeline stay sequential (current hand-off)")

    worker = partial(_worker_B_pipeline, particles=PARTICLES,
                     randomize_starts=randomize_starts)
    results, all_records, finals, n_failed = {}, [], {}, 0
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for done, (name, recs, fin, ok) in enumerate(executor.map(worker, tasks), start=1):
            results.setdefault(name, []).extend(recs)
            all_records.extend(recs)
            if ok:
                finals.setdefault(name, []).append(fin)
            else:
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ {name} pipeline failed after {len(recs)} stages: {fin.get('error')}")
            if verbose:
                got = [r["final_mse"] for r in recs]
                tail = f"last stage MSE={got[-1]:.3e}" if got else "no stage completed"
                print(f"   [{done}/{len(tasks)}] {name:14s} {len(recs)}/{len(stages)} stages, {tail}")
    elapsed = time.perf_counter() - t0

    df_B = pd.DataFrame(all_records)
    if verbose and not df_B.empty:
        converged = df_B.groupby("method_tag")["final_mse"].apply(lambda s: (s < EPSILON).sum())
        print(f"\n✓ Scenario B finished in {elapsed:.2f}s ({n_failed} failed pipelines)")
        print(f"  stages with MSE < {EPSILON:g} per method (out of {len(stages) * n_runs}):")
        for name, n_ok in converged.items():
            print(f"    {name:16s} {int(n_ok)}")
    perf = _report_speedup("B", all_records, elapsed, n_workers)
    return results, df_B, finals, perf


def run_single_stage_serial(scenario, n_runs, methods=None, cfg=None, verbose=False):
    """Same workload, one task at a time — the baseline `--benchmark` compares against.

    It runs in a *one-worker pool* rather than in this process on purpose: the tasks then see
    the same pinned PYTHONHASHSEED (and therefore the same random starts) as the parallel run,
    which is what makes the two arms comparable.
    """
    cfg = cfg if cfg is not None else SCENARIOS[scenario]
    methods = methods if methods is not None else SCENARIOS[scenario]["methods"]
    tasks = [
        (scenario, cfg, spec[0], spec[1],
         method_label(spec[0], spec[2] if len(spec) == 3 else None), run_idx)
        for spec in methods
        for run_idx in range(n_runs)
    ]
    worker = partial(_worker_single_stage, particles=PARTICLES)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        records = [rec for _, rec, ok in executor.map(worker, tasks) if ok]
    return records, time.perf_counter() - t0


# ── 7. Entry point ───────────────────────────────────────────────────
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
        description="Run scenarios A/B/C with the random starts spread over CPU cores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", choices=["A", "B", "C", "all"], default="A",
                        help="which scenario to run")
    parser.add_argument("--n_runs", type=int, default=4,
                        help="random starts per method (the axis that gets parallelised)")
    parser.add_argument("--max_cores", type=int, default=DEFAULT_CORES,
                        help="worker processes (capped at the number of tasks)")
    parser.add_argument("--save_dir", type=str, default="../../results/scenario_A-C_para",
                        help="where to pickle the results ('' to skip saving)")
    parser.add_argument("--fixed_starts", action="store_true",
                        help="Scenario B: keep the notebook's fixed stage starts "
                             "(every run then produces an identical pipeline)")
    parser.add_argument("--benchmark", action="store_true",
                        help="also run the same workload serially and print the measured speedup")
    args = parser.parse_args()

    print(f"host cores: {os.cpu_count()}   |   using: {args.max_cores}   |   "
          f"n_runs: {args.n_runs}   |   scenario: {args.scenario}")

    t_all = time.perf_counter()
    todo = ["A", "B", "C"] if args.scenario == "all" else [args.scenario]

    for scen in todo:
        if scen == "B":
            results, df, finals, perf = run_scenario_B_parallel(
                n_runs=args.n_runs, max_cores=args.max_cores,
                randomize_starts=(False if args.fixed_starts else None),
            )
            _save({"results": results, "df": df, "final_currents": finals, "perf": perf},
                  args.save_dir, "scenario_B_para.pkl")
        else:
            results, df, final_summary, perf = run_single_stage_parallel(
                scen, n_runs=args.n_runs, max_cores=args.max_cores,
            )
            _save({"results": results, "df": df, "final_summary": final_summary, "perf": perf},
                  args.save_dir, f"scenario_{scen}_para.pkl")

            if args.benchmark:
                print(f"\n⏱  benchmark: repeating Scenario {scen} serially in this process...")
                _, serial_elapsed = run_single_stage_serial(scen, n_runs=args.n_runs)
                print(f"   serial   : {serial_elapsed:8.1f} s\n"
                      f"   parallel : {perf['elapsed_s']:8.1f} s on {perf['n_workers']} cores\n"
                      f"   measured speedup: {serial_elapsed / perf['elapsed_s']:.2f}x")

    print(f"\n🏁 total wall time: {time.perf_counter() - t_all:.2f} s")
