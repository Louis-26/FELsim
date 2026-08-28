"""
Multiprocessing runners for Scenarios A / B / C (the `--use_multi 1` path of
`step_5/scenario_A-C_seq.py`).

The optimisation is untouched — every task still calls the same `beamOptimizer.calc()` ->
`scipy.optimize.minimize` with the same variables, objectives, bounds and method options.
Only the scheduling changes.

WHAT IS PARALLEL
----------------
Each random start is an independent optimisation: it builds its own beamline, clones its own
particles and never touches another run's state. Same across methods. So the work is
`len(METHODS) * N_RUNS` embarrassingly parallel tasks:

    Scenario A / C : tasks = methods x random starts        (fully parallel)
    Scenario B     : tasks = methods x random starts        (parallel ACROSS pipelines only —
                     the 11 stages inside one pipeline are sequential, since stage k+1 starts
                     from the currents stage k wrote back)

Processes, not threads: the objective sits in NumPy/PyTorch matrix products that hold the GIL,
and every run mutates its own beamline objects.

THREE THINGS THAT SILENTLY BREAK MULTIPROCESSING HERE
-----------------------------------------------------
1. `configs.PARTICLES` is drawn at import time from the *unseeded* global torch RNG. Under
   spawn each worker re-imports the module and would materialise its OWN 1000-particle bunch,
   i.e. every task would optimise a slightly different objective (measured: MSE 0.578 / 0.632
   / 0.601 across three workers for identical currents, vs 0.5815 everywhere once the parent's
   bunch is shipped). Every worker here therefore takes `particles` as an argument.
2. `run_benchmark` draws its random start vector by iterating `list({seg_var[i][0] ...})` — a
   set of *strings*, whose order depends on per-interpreter hash randomisation. The entry
   script pins `PYTHONHASHSEED=0` (children inherit it) and every task runs inside a worker,
   so a given `run_idx` means the same start point in every process and across invocations.
3. Importing `experiments_utils` costs ~25 s (PyTorch + the beamline Excel + the bunch). One
   persistent pool pays that once per worker; a pool per task would pay it every time.

MEMORY IS USUALLY THE BINDING CONSTRAINT, NOT CORES
---------------------------------------------------
Every worker imports PyTorch, and on Windows a CUDA-enabled build maps its whole CUDA runtime
at import time (~1.5 GB of commit charge per process, whether or not a GPU is used). Ask for
more workers than the machine has memory for and the pool dies during start-up with
`OSError: [WinError 1455] The paging file is too small for this operation to complete` while
loading `torch\\lib\\*.dll` — several minutes into what may be an hours-long run.
`safe_worker_count()` therefore caps the requested worker count by the memory actually
available. Raise the ceiling by closing memory hogs (IDEs, browsers, idle Jupyter kernels) or
by enlarging the Windows paging file; `MB_PER_WORKER` below is the per-worker estimate.

Variable specs must stay picklable, which is why `configs._v()` / `configs.A_VARS` use the
module-level `identity_func` instead of a lambda.
"""

import concurrent.futures
import copy
import sys
import time
from functools import partial

import numpy as np
import pandas as pd

from concurrent.futures.process import BrokenProcessPool

from experiments_utils import (
    ExcelElements,
    EXCEL_PATH,
    beamOptimizer,
    run_benchmark,
    results_to_df,
    method_label,
    plot_stat_convergence,
)


MB_PER_WORKER = 1500      # rough commit charge of one worker once PyTorch is imported


def memory_headroom_mb():
    """Memory a new process can actually commit, in MB (None if it cannot be determined).

    On Windows the binding limit is the *commit charge*, not free RAM and not free page-file
    space: `psutil.swap_memory().free` reports the page file's spare capacity (tens of GB here)
    while `CommitLimit - CommitTotal` can be under 3 GB — take the page-file number and you
    will happily launch 20 workers and watch them all die loading torch.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class _PerfInfo(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
                            ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                            ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                            ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                            ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                            ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
                            ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]

            info = _PerfInfo()
            info.cb = ctypes.sizeof(_PerfInfo)
            if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
                return None
            return (info.CommitLimit - info.CommitTotal) * info.PageSize / 2 ** 20
        except Exception:  # noqa: BLE001
            return None
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 20
    except Exception:  # noqa: BLE001
        return None


def safe_worker_count(requested, n_tasks=None, mb_per_worker=MB_PER_WORKER, verbose=True):
    """Cap `requested` by the number of tasks and by the memory actually available.

    Each worker maps the whole PyTorch (+CUDA) runtime, so an over-ambitious worker count does
    not merely run slowly — it kills the pool during start-up with WinError 1455. Better to
    lose some parallelism than to lose an hours-long run.
    """
    n = max(1, int(requested))
    if n_tasks:
        n = min(n, int(n_tasks))
    headroom_mb = memory_headroom_mb()
    if headroom_mb is None:
        return n
    affordable = max(1, int(headroom_mb // mb_per_worker))
    if affordable < n:
        if verbose:
            print(f"⚠️  memory-capped: {n} -> {affordable} worker(s) — only "
                  f"{headroom_mb / 1024:.1f} GB can be committed and each worker needs "
                  f"~{mb_per_worker / 1024:.1f} GB for PyTorch.\n"
                  f"    Close memory hogs (IDEs, browsers, idle Jupyter kernels) or enlarge "
                  f"the Windows paging file to use more cores.")
        n = affordable
    return n


def _pool_stream(iterator, n_workers):
    """Yield from a pool's result iterator, turning a dead pool into an actionable message.

    The usual cause here is not a bug but memory: each worker maps the whole PyTorch runtime,
    and when Windows runs out of commit charge the workers die while loading `torch\\lib\\*.dll`
    and the pool comes back as BrokenProcessPool / OSError.
    """
    try:
        yield from iterator
    except (BrokenProcessPool, OSError) as exc:
        raise RuntimeError(
            f"the worker pool died with {n_workers} workers ({type(exc).__name__}: {exc}).\n"
            f"    Most likely the machine ran out of memory: every worker imports PyTorch "
            f"(~{MB_PER_WORKER} MB of commit charge each).\n"
            f"    Retry with a smaller --max_cores, close memory hogs (IDEs, browsers, idle "
            f"Jupyter kernels), or enlarge the Windows paging file."
        ) from exc


# ── Workers (module level so they survive pickling into a spawned process) ──
def _worker_single_stage(task, particles, METHOD_OPTIONS=None, SEED=42,
                         use_log=False, use_epsilon=1e-13, noise=False, sigma=None):
    """One (method, random start) task of a single-stage scenario (A or C).

    The scenario config travels inside the task: a spawned worker re-imports this module from
    disk, so anything the caller changed at runtime would otherwise never reach it.

    Returns (method_name, record, ok) — exceptions are returned, never raised, so one bad task
    cannot take down the pool.
    """
    METHOD_OPTIONS = METHOD_OPTIONS or {}
    method_name, run_idx = "?", -1
    try:
        scenario, cfg, method, jac, method_name, run_idx = task
        bounds = {info[0]: cfg["bounds"] for info in cfg["vars"].values()}
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
            seed_offset=run_idx,        # <- what makes each task a different random start
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
    except Exception as exc:  # noqa: BLE001
        return method_name, {"method_tag": method_name, "run": run_idx + 1,
                             "final_mse": np.nan, "error": str(exc)}, False


def _stage_start_points(fixed_sp, rng, randomize):
    """Start point for one Scenario B stage: the fixed value from STAGES_B, or a random draw.

    `scenario_B_utils.run_scenario_B` always uses the fixed `start` values, which makes every
    repetition of the pipeline byte-identical — so repeating it only buys something once the
    starts differ. With `randomize=True` each stage draws uniformly inside its bounds, exactly
    like `run_benchmark` does for A and C.
    """
    sp = copy.deepcopy(fixed_sp)
    if randomize:
        for spec in sp.values():
            lo, hi = spec["bounds"]
            spec["start"] = float(rng.uniform(lo, hi))
    return sp


def _worker_B_pipeline(task, particles, METHOD_OPTIONS=None, SEED=42, randomize_starts=True):
    """One (method, random start) task of Scenario B: a whole 11-stage pipeline.

    The stages run sequentially inside — each writes its optimised currents into this task's
    private beamline before the next one reads it — so a pipeline is the smallest unit that
    can be handed to a core.

    Returns (method_name, stage_records, final_currents, ok).
    """
    METHOD_OPTIONS = METHOD_OPTIONS or {}
    stages, method, jac, method_name, run_idx = task
    options = METHOD_OPTIONS.get(method_name, None)
    rng = np.random.default_rng(SEED + run_idx * 997)
    stage_records = []
    try:
        bl_run = ExcelElements(EXCEL_PATH).create_beamline()   # private beamline per task
        p_run = particles.clone()                              # the parent's bunch

        for s_i, (lbl, seg_var, obj, bl_len, fixed_sp) in enumerate(stages):
            sp = _stage_start_points(fixed_sp, rng, randomize_starts)
            opti = beamOptimizer(bl_run[:bl_len], p_run.clone())
            t0 = time.perf_counter()
            res = opti.calc(method, seg_var, sp, copy.deepcopy(obj), jac=jac, options=options,
                            printResults=False, plotProgress=False)
            wall = time.perf_counter() - t0

            # hand-off: the optimised currents become the physical state for the next stage
            for elem_idx, var_info in seg_var.items():
                var_idx = opti.variablesToOptimize.index(var_info[0])
                setattr(bl_run[elem_idx], "current", float(res.x[var_idx]))
            if s_i == 4:      # Stage 5 symmetry, as in scenario_B_utils.run_scenario_B
                bl_run[43].current = bl_run[33].current
                bl_run[41].current = bl_run[35].current
                bl_run[39].current = bl_run[37].current

            nfev = int(getattr(res, "nfev", len(opti.plotMSE)))
            nit = int(getattr(res, "nit", nfev))
            if nit < 0:
                nit = nfev
            stage_records.append({
                "run": run_idx + 1, "stage": s_i + 1, "stage_name": lbl,
                "method_tag": method_name, "final_mse": float(res.fun),
                "nfev": nfev, "nit": nit, "wall_time": round(wall, 3),
                "success": bool(res.success),
                "start_x": {v: sp[v]["start"] for v in sp},
            })

        finals = {elem_idx: float(bl_run[elem_idx].current)
                  for _, seg_var, _, _, _ in stages for elem_idx in seg_var}
        if len(stages) > 4:                    # mirrored elements only exist once stage 5 ran
            finals.update({i: float(bl_run[i].current) for i in (39, 41, 43)})
        return method_name, stage_records, finals, True
    except Exception as exc:  # noqa: BLE001
        return method_name, stage_records, {"error": str(exc)}, False


# ── Reporting helpers ────────────────────────────────────────────────
def report_speedup(scenario, records, elapsed, n_workers, key="wall_time"):
    """CPU-time ratio: (summed per-task optimiser time) / (elapsed wall time).

    A rough estimate, biased both ways — it ignores the ~25 s worker start-up (understating
    the gain) while the per-task timings were measured inside the loaded pool and so already
    contain CPU contention (overstating it). Quote a measured number instead when it matters.
    """
    serial = float(np.nansum([r.get(key, np.nan) for r in records]))
    ratio = serial / elapsed if elapsed > 0 else float("nan")
    print(f"\n⚡ Scenario {scenario}: {len(records)} tasks on {n_workers} workers\n"
          f"   optimiser time summed over tasks : {serial:8.1f} s\n"
          f"   elapsed wall time                : {elapsed:8.1f} s\n"
          f"   CPU-time ratio ~= {ratio:.2f}x   (rough estimate)")
    return {"cpu_time_ratio": ratio, "summed_task_time_s": serial,
            "elapsed_s": elapsed, "n_workers": n_workers}


def summarise_single_stage(scenario, results, VARS, OBJ, EPSILON, REF_CURRENTS, verbose=True):
    """Per-method summary table — same columns and formatting as run_scenario_A/C."""
    current_name = list(dict.fromkeys([info[0] for info in VARS.values()]))
    ref_I_name = [n + "_ref" for n in current_name]
    evalPos_parameter = [str(i) + "_" + "_".join(reversed(j[0]["measure"])) for i, j in OBJ.items()]

    flat = [r for v in results.values() for r in v]
    df = results_to_df(flat, current_name, evalPos_parameter)
    if df.empty or "final_mse" not in df.columns:
        print(f"⚠️  Scenario {scenario}: no usable task result to summarise")
        return df, pd.DataFrame()

    # a run that raised inside run_benchmark comes back with no current / measured columns
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
    ).join(df.groupby("method_tag")["final_mse"].apply(_geom_mean_converged).rename("final_mse_mean"))

    idx_best = df.groupby("method_tag")["final_mse"].idxmin().dropna()
    best_cols = ["method_tag", *current_name, *evalPos_parameter]
    best_runs = (df.loc[idx_best, best_cols].set_index("method_tag") if len(idx_best)
                 else pd.DataFrame(columns=best_cols).set_index("method_tag"))
    final_summary = summary_stats.join(best_runs).reset_index()
    for k, var_info in VARS.items():
        final_summary[f"{var_info[0]}_ref"] = REF_CURRENTS[k]

    hybrid = [x for pair in zip(current_name, ref_I_name) for x in pair]
    final_summary = final_summary[[
        "method_tag", "conv_rate", "nit_mean", "nfev_mean", "wall_time_mean",
        "final_mse_mean", *hybrid, *evalPos_parameter,
    ]]

    if verbose:
        fmt = {"conv_rate": "{:.0%}", "nit_mean": "{:.1f}", "nfev_mean": "{:.1f}",
               "wall_time_mean": "{:.3f} s", "final_mse_mean": "{:.2e}",
               **{c: "{:.4f} A" for c in hybrid},
               **{e: "{:.4f}" for e in evalPos_parameter}}
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


# ── Parallel drivers ─────────────────────────────────────────────────
def run_scenario_single_parallel(scenario, VARS, OBJ, BEAMLINE_LEN, CURRENT_BOUNDS, EPSILON,
                                 N_RUNS, METHODS, PARTICLES, REF_CURRENTS, METHOD_OPTIONS=None,
                                 SEED=42, max_cores=6, use_log=False, use_epsilon=1e-13,
                                 noise=False, sigma=None, plot_curve=False, scale="log",
                                 verbose=True):
    """Scenario A or C: `len(METHODS) * N_RUNS` independent optimisations over a process pool.

    Returns (results, df, final_summary, perf) — `results`/`df`/`final_summary` have the same
    shape as the sequential `run_scenario_A` / `run_scenario_C`.
    """
    METHOD_OPTIONS = METHOD_OPTIONS or {}
    cfg = {"vars": VARS, "obj": OBJ, "beamline_len": BEAMLINE_LEN, "bounds": CURRENT_BOUNDS}

    tasks = []
    for spec in METHODS:
        method, jac = spec[0], spec[1]
        name = method_label(method, spec[2] if len(spec) == 3 else None)
        for run_idx in range(N_RUNS):
            tasks.append((scenario, cfg, method, jac, name, run_idx))

    n_workers = safe_worker_count(max_cores, len(tasks), verbose=verbose)
    if verbose:
        print(f"\n▶ 🚀 Scenario {scenario} — {len(METHODS)} methods x {N_RUNS} random starts "
              f"= {len(tasks)} tasks on {n_workers} cores")

    worker = partial(_worker_single_stage, particles=PARTICLES, METHOD_OPTIONS=METHOD_OPTIONS,
                     SEED=SEED, use_log=use_log, use_epsilon=use_epsilon, noise=noise, sigma=sigma)
    results, ok_records, n_failed = {}, [], 0
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        stream = _pool_stream(executor.map(worker, tasks), n_workers)
        for done, (name, rec, ok) in enumerate(stream, start=1):
            if ok:
                results.setdefault(name, []).append(rec)
                ok_records.append(rec)
            # run_benchmark swallows its own exceptions and returns a NaN record with an
            # "error" key, so count that as a failure too
            if not ok or rec.get("error"):
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ {name} run {rec.get('run')} failed: {rec.get('error')}")
            if verbose and done % max(1, len(tasks) // 10) == 0:
                print(f"   {done}/{len(tasks)} tasks done ({time.perf_counter() - t0:.0f}s)")
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"✓ Scenario {scenario} finished in {elapsed:.2f}s ({n_failed} failed)")
    df, final_summary = summarise_single_stage(scenario, results, VARS, OBJ, EPSILON,
                                               REF_CURRENTS, verbose=verbose)
    if plot_curve and results:
        plot_stat_convergence(results, title=f"Scenario {scenario}: (Target MSE < {EPSILON})",
                              convergence_epsilon=EPSILON, scale=scale)
    perf = report_speedup(scenario, ok_records, elapsed, n_workers)
    return results, df, final_summary, perf


def run_scenario_B_parallel(STAGES_B, PARTICLES, METHODS_B, N_RUNS_B=1, METHOD_OPTIONS=None,
                            SEED=42, max_cores=6, EPSILON=1e-3, randomize_starts=None,
                            verbose=True):
    """Scenario B: one task = one full 11-stage pipeline (stages stay sequential inside).

    Returns (results, df_B, final_currents, perf); `final_currents[method]` is a list of
    {element index: current} dicts, one per run — the beamline objects themselves are heavy
    to ship back, and only the currents matter downstream.
    """
    METHOD_OPTIONS = METHOD_OPTIONS or {}
    if randomize_starts is None:
        # repeating a pipeline from identical fixed starts would just reproduce itself
        randomize_starts = N_RUNS_B > 1

    tasks = []
    for spec in METHODS_B:
        method, jac = spec[0], spec[1]
        name = method_label(method, spec[2] if len(spec) == 3 else None)
        for run_idx in range(N_RUNS_B):
            tasks.append((STAGES_B, method, jac, name, run_idx))

    n_workers = safe_worker_count(max_cores, len(tasks), verbose=verbose)
    if verbose:
        print(f"\n▶ 🚀 Scenario B (11-stage sequential) — {len(METHODS_B)} methods x {N_RUNS_B} runs "
              f"= {len(tasks)} pipelines on {n_workers} cores")
        print(f"   start points: {'random per run' if randomize_starts else 'fixed (as in the notebook)'}"
              f"   |   stages inside a pipeline stay sequential (current hand-off)")

    worker = partial(_worker_B_pipeline, particles=PARTICLES, METHOD_OPTIONS=METHOD_OPTIONS,
                     SEED=SEED, randomize_starts=randomize_starts)
    results, all_records, finals, n_failed = {}, [], {}, 0
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        stream = _pool_stream(executor.map(worker, tasks), n_workers)
        for done, (name, recs, fin, ok) in enumerate(stream, start=1):
            results.setdefault(name, []).extend(recs)
            all_records.extend(recs)
            if ok:
                finals.setdefault(name, []).append(fin)
            else:
                n_failed += 1
                if verbose:
                    print(f"  ⚠️ {name} pipeline failed after {len(recs)} stages: {fin.get('error')}")
            if verbose:
                worst = max((r["final_mse"] for r in recs), default=float("nan"))
                print(f"   [{done}/{len(tasks)}] {name:14s} {len(recs)}/{len(STAGES_B)} stages, "
                      f"worst stage MSE={worst:.3e}")
    elapsed = time.perf_counter() - t0

    df_B = pd.DataFrame(all_records)
    if verbose and not df_B.empty:
        print(f"\n✓ Scenario B finished in {elapsed:.2f}s ({n_failed} failed pipelines)")
        print(f"  stages with MSE < {EPSILON:g}: {int((df_B['final_mse'] < EPSILON).sum())}/{len(df_B)}")
        print("  worst stage MSE per method (the metric quoted in the README):")
        for name, worst in df_B.groupby("method_tag")["final_mse"].max().items():
            print(f"    {name:16s} {worst:.3e}")
    perf = report_speedup("B", all_records, elapsed, n_workers)
    return results, df_B, finals, perf
