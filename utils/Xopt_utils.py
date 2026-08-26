import os, sys, copy, time, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

# FELsim backend + beam / lattice constants (same import path as scenario_A-C.ipynb)
sys.path.append(os.path.join(os.path.abspath(__file__), "../experiment/"))
# sys.path.append(os.path.join(os.path.abspath(__file__), "../../"))
# from experiments_utils import (PARTICLES, EXCEL_PATH, ExcelElements, beamOptimizer,
#                                ALPHA_XM, ALPHA_YM, BETA_XM, BETA_YM)
from configs import *

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator

CURRENT_BOUNDS = (0.01, 1.5)     # same bounds as scenario_A-C.ipynb
EPSILON        = 1e-3            # success threshold on the raw MSE
SEED           = 42
XOPT_TAG       = "Xopt-EI"       # 'method_tag' used in the result tables

bl_full = ExcelElements(EXCEL_PATH).create_beamline()



def _v(name):
    """Variable spec understood by beamOptimizer: [name, attribute, x -> attribute value]."""
    return [name, "current", lambda x, _n=name: x]


def measured_names(obj):
    """Column names of the measured quantities, e.g. '8_alpha_x' (same convention as experiments_utils)."""
    return [f"{elem}_{t['measure'][1]}_{t['measure'][0]}" for elem, targets in obj.items() for t in targets]


def make_felsim_evaluator(beamline, seg_var, obj, particles=PARTICLES, bounds=CURRENT_BOUNDS):
    """
    Wrap a FELsim beamline slice into an Xopt-compatible ``evaluate(inputs) -> dict``.

    Returns ``(evaluate, var_names, out_names)``.  ``evaluate`` returns
      - 'MSE' : the raw weighted MSE — the BO objective, identical to what scipy minimises
                in scenario_A-C.ipynb (no log transform)
      - one entry per measured quantity, named like '8_alpha_x'
    Any exception / non-finite MSE is reported as NaN, so Xopt simply skips that point.
    """

    var_names = list(dict.fromkeys(v[0] for v in seg_var.values()))      # spec order, unique
    out_names = measured_names(obj)

    opti = beamOptimizer(beamline, particles.clone())
    start_point = {n: {"bounds": tuple(bounds), "start": 1.0} for n in var_names}
    opti._prepare(seg_var, start_point, copy.deepcopy(obj))              # set-up only, no optimisation
    order = list(opti.variablesToOptimize)                               # beamOptimizer's internal (set) order
    flat_goals = [g for elem in opti.objectives for g in opti.objectives[elem]]

    def evaluate(inputs: dict) -> dict:
        out = {"MSE": np.nan, **{n: np.nan for n in out_names}}
        try:
            x = np.array([float(inputs[n]) for n in order])
            mse = float(opti._optiSpeed(x))
            if np.isfinite(mse):
                out["MSE"] = mse
                for name, g in zip(out_names, flat_goals):
                    val = g["measured"]
                    out[name] = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            pass                                                        
        return out

    return evaluate, var_names, out_names


def run_xopt_bo(evaluate, var_names, n_init, n_steps, seed=SEED, bounds=CURRENT_BOUNDS,
                label="", verbose=True, print_every=5):
    """
    Random initialisation + Expected-Improvement BO loop (same recipe as Scenario A above),
    on the raw MSE objective.  Returns the Xopt object and a stats dict.
    """
    vocs = VOCS(variables={n: list(bounds) for n in var_names},
                objectives={"MSE": "MINIMIZE"})
    X = Xopt(evaluator=Evaluator(function=evaluate),
             generator=ExpectedImprovementGenerator(vocs=vocs), vocs=vocs)

    torch.manual_seed(seed)                       # reproducible GP fits / acquisition optimisation
    t0 = time.perf_counter()
    X.random_evaluate(n_init, seed=seed)          # phase 1: random exploration
    n_fallback = 0
    for i in range(n_steps):                      # phase 2: fit GP -> maximise EI -> evaluate
        try:
            X.step()
        except Exception:                         # e.g. GP fit on degenerate data -> random point instead
            n_fallback += 1
            X.random_evaluate(1, seed=seed + 1000 + i)
        if verbose and print_every and (i + 1) % print_every == 0:
            print(f"      {label:18s} BO step {i + 1:3d}/{n_steps}   best MSE = {X.data['MSE'].min():.3e}")
    wall = time.perf_counter() - t0

    mse = X.data["MSE"].to_numpy(dtype=float)
    n_ok = int(np.isfinite(mse).sum())
    if n_ok:
        best_pos = int(np.nanargmin(mse))
        best = X.data.iloc[best_pos]
        hist = np.minimum.accumulate(np.nan_to_num(mse, nan=np.inf))
        n_to_eps = int(np.argmax(hist < EPSILON)) + 1 if np.any(hist < EPSILON) else None
        best_x = {n: float(best[n]) for n in var_names}
        final_mse = float(best["MSE"])
    else:                                         # every evaluation failed
        best_pos, n_to_eps, best_x, final_mse = -1, None, {n: np.nan for n in var_names}, np.nan
    stats = {
        "final_mse": final_mse,
        "best_x": best_x,
        "best_eval": best_pos + 1,
        "nfev": int(len(X.data)),
        "n_failed": int(len(X.data) - n_ok),
        "n_fallback": n_fallback,
        "n_to_eps": n_to_eps,
        "wall_time": round(wall, 3),
        "success": bool(np.isfinite(final_mse) and final_mse < EPSILON),
    }
    return X, stats


def run_scenario_A_xopt(X, N_INIT_A, N_BO_STEPS_A, A_out_names, FELSIM_S1_CURRENTS, SEED):
    t0 = time.time()
    X.random_evaluate(N_INIT_A, seed=SEED)          # phase 1: random exploration
    print(f"random init ({N_INIT_A} pts):    best MSE = {X.data['MSE'].min():.3e}")

    for i in range(N_BO_STEPS_A):                   # phase 2: fit GP -> maximize EI -> evaluate
        X.step()
        if (i + 1) % 5 == 0:
            print(f"BO step {i + 1:3d}/{N_BO_STEPS_A}:      best MSE = {X.data['MSE'].min():.3e}")

    print(f"\ndone: {len(X.data)} FELsim evaluations in {time.time() - t0:.1f} s")
    best_idx = X.data["MSE"].idxmin()
    best     = X.data.loc[best_idx]
    REF_I = FELSIM_S1_CURRENTS[1], FELSIM_S1_CURRENTS[3]
    success = bool(best["MSE"] < EPSILON)
    print(f"best found : I = {best['I']:.4f},  I2 = {best['I2']:.4f}   (evaluation #{X.data.index.get_loc(best_idx) + 1})")
    print(f"reference  : I = {REF_I[0]:.4f},  I2 = {REF_I[1]:.4f}   (FELsim S1 currents)")
    print(f"|delta I|  : {abs(best['I'] - REF_I[0]):.4f},  {abs(best['I2'] - REF_I[1]):.4f}")
    print(f"alpha_x(8) = {best[A_out_names[0]]:+.4e},   alpha_y(9) = {best[A_out_names[1]]:+.4e}")
    print(f"best MSE   : {best['MSE']:.3e}   (EPSILON = {EPSILON:g} -> {'SUCCESS' if success else 'not converged'})")
   

def plot_scenario_A_xopt_convergence(X, N_INIT_A):

    mse_hist    = X.data["MSE"].to_numpy()
    best_so_far = np.minimum.accumulate(mse_hist)
    evals       = np.arange(1, len(mse_hist) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(evals, mse_hist, "o", ms=5, mfc="#9ecae1", mec="white", mew=0.8,
                label="evaluation")
    ax.semilogy(evals, best_so_far, drawstyle="steps-post", color="#1f4e79", lw=2,
                label="best so far")
    ax.axhline(EPSILON, color="#c44e52", lw=1.2, ls="--", label=f"EPSILON = {EPSILON:g}")
    ax.axvline(N_INIT_A + 0.5, color="gray", lw=1, ls=":")
    ax.text(N_INIT_A + 0.8, ax.get_ylim()[1] * 0.25, "BO starts", fontsize=9, color="gray")
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("MSE")
    ax.set_title("Scenario A — Xopt Bayesian optimization on FELsim (convergence)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    plt.show()

def bo_convergence_plot(ax, data, n_init, title=None, eps=EPSILON):
    """Evaluations + best-so-far MSE on a log axis (same style as the Scenario A plot above)."""
    mse = np.nan_to_num(data["MSE"].to_numpy(dtype=float), nan=np.inf)
    ev = np.arange(1, len(mse) + 1)
    ok = np.isfinite(mse)
    ax.semilogy(ev[ok], mse[ok], "o", ms=4, mfc="#9ecae1", mec="white", mew=0.7, label="evaluation")
    ax.semilogy(ev, np.minimum.accumulate(mse), drawstyle="steps-post", color="#1f4e79", lw=1.8, label="best so far")
    ax.axhline(eps, color="#c44e52", lw=1.1, ls="--", label=f"EPSILON = {eps:g}")
    ax.axvline(n_init + 0.5, color="gray", lw=0.9, ls=":", label="BO starts")
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.tick_params(labelsize=8)
    
    
def run_scenario_B_xopt(STAGES_B,CURRENT_BOUNDS, n_runs=1, n_init=2, n_steps=25, seed=SEED, verbose=True):
    """
    Xopt/BO counterpart of scenario_B_utils.run_scenario_B.
    Returns (results, df_B, final_beamline, xopt_data) with the same per-stage record layout;
    xopt_data[(run, stage)] holds the full Xopt history (X.data) of that stage.
    """
    stage_results, xopt_data = [], {}
    bl_run = None
    t_total = time.perf_counter()
    if verbose:
        print(f"\n▶ 🚀 Starting Scenario B (11-Stage Sequential) — Method: {XOPT_TAG} ({n_runs} runs total)")

    for run_i in range(n_runs):
        if verbose and n_runs > 1:
            print(f"  ── Run {run_i + 1}/{n_runs} ──")
        bl_run = ExcelElements(EXCEL_PATH).create_beamline()          # fresh beamline for every run
        run_seed = seed + run_i * 997

        for s_i, (lbl, seg_var, obj, bl_len, fixed_sp) in enumerate(STAGES_B):
            evaluate, var_names, _ = make_felsim_evaluator(bl_run[:bl_len], seg_var, obj, bounds=CURRENT_BOUNDS)
            X, st = run_xopt_bo(evaluate, var_names, n_init, n_steps,
                                seed=run_seed + s_i, label=lbl, verbose=False)
            xopt_data[(run_i + 1, s_i + 1)] = X.data.copy()

            # 🎯 Crucial hand-off: write the best currents back to the shared beamline for the next stage
            if all(np.isfinite(v) for v in st["best_x"].values()):
                for elem_idx, var_info in seg_var.items():
                    bl_run[elem_idx].current = float(st["best_x"][var_info[0]])
                if s_i == 4:                                             # Stage 5 symmetry (as in scenario_B_utils)
                    bl_run[43].current = bl_run[33].current
                    bl_run[41].current = bl_run[35].current
                    bl_run[39].current = bl_run[37].current
            elif verbose:                                                # every evaluation failed: keep previous currents
                print(f"    {lbl:25s} ⚠ all {st['nfev']} evaluations failed — currents left unchanged")

            rec = {"run": run_i + 1, "stage": s_i + 1, "stage_name": lbl, "method_tag": XOPT_TAG,
                   "final_mse": st["final_mse"], "nfev": st["nfev"], "nit": st["nfev"],
                   "wall_time": st["wall_time"], "success": st["success"],
                   "n_to_eps": st["n_to_eps"], "n_failed": st["n_failed"], "n_fallback": st["n_fallback"],
                   **st["best_x"]}
            stage_results.append(rec)
            if verbose:
                extra = f"  (MSE < EPSILON after {st['n_to_eps']} evals)" if st["n_to_eps"] else ""
                print(f"    {lbl:25s} MSE={st['final_mse']:.3e}  nfev={st['nfev']:<4} t={st['wall_time']:.2f}s{extra}")

    if verbose:
        print(f"✓ {XOPT_TAG} pipeline execution completed, total time: {time.perf_counter() - t_total:.2f}s")
    df_B = pd.DataFrame(stage_results)
    return {XOPT_TAG: stage_results}, df_B, bl_run, xopt_data

def display_scenario_B_xopt_summary(df_B_xopt, STAGES_B, final_beamline_B_xopt, FELSIM_S1_CURRENTS):
    last = df_B_xopt[df_B_xopt.run == df_B_xopt.run.max()].set_index("stage")
    rows = []
    for s_i, (lbl, seg_var, obj, bl_len, _) in enumerate(STAGES_B):
        for elem_idx, var_info in seg_var.items():
            rows.append({"stage": s_i + 1, "stage_name": lbl, "element": elem_idx, "variable": var_info[0],
                        "I_xopt (A)": final_beamline_B_xopt[elem_idx].current,
                        "I_ref (A)": FELSIM_S1_CURRENTS.get(elem_idx, np.nan),
                        "stage MSE": last.loc[s_i + 1, "final_mse"],
                        "MSE < EPS": bool(last.loc[s_i + 1, "final_mse"] < EPSILON)})
    currents_B_xopt = pd.DataFrame(rows)

    print(f"Scenario B ({XOPT_TAG}, run {int(df_B_xopt.run.max())}): stages with MSE < {EPSILON:g}: "
        f"{int(last.success.sum())}/{len(last)}   |   FELsim evaluations: {int(last.nfev.sum())}"
        f"   |   wall time: {last.wall_time.sum():.0f} s")
    print("(note: many reference currents lie outside CURRENT_BOUNDS, as in scenario_A-C.ipynb)")
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}" if abs(v) < 1e-3 else f"{v:.4f}"):
        display(currents_B_xopt)



# per-stage convergence of the (last) Scenario B run
def plot_scenario_B_xopt_convergence(STAGES_B, df_B_xopt, xopt_data_B, N_INIT_B):
    run_id = int(df_B_xopt.run.max())
    n_st = len(STAGES_B)
    ncols = 4
    nrows = int(np.ceil(n_st / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for s_i in range(n_st):
        bo_convergence_plot(axes[s_i], xopt_data_B[(run_id, s_i + 1)], N_INIT_B, title=STAGES_B[s_i][0])
        if s_i % ncols == 0:
            axes[s_i].set_ylabel("MSE", fontsize=8)
        if s_i >= n_st - ncols:
            axes[s_i].set_xlabel("evaluation #", fontsize=8)
    for ax in axes[n_st:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.06), frameon=False, fontsize=9)
    fig.suptitle(f"Scenario B — Xopt BO convergence per stage ({N_INIT_B} random + {N_BO_STEPS_B} BO evaluations)",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()





def run_scenario_C_xopt(VARS, OBJ, beamline_len, ref_currents, CURRENT_BOUNDS, n_runs=1, n_init=11, n_steps=60,
                        seed=SEED, verbose=True):
    """
    Xopt/BO counterpart of experiments_utils.run_scenario_C (single stage, all variables at once).
    Returns (results, df_C, final_summary, xopt_data); records follow run_benchmark's layout:
    'mse_curve' is the raw MSE of every evaluation (NaN where FELsim failed), exactly like
    run_benchmark's list(opti.plotMSE), so plot_stat_convergence from experiments_utils can be
    used on them; 'best_mse_curve' additionally holds the best-so-far envelope used for BO plots.
    """
    var_names = list(dict.fromkeys(v[0] for v in VARS.values()))
    out_names = measured_names(OBJ)
    results, xopt_data = [], {}
    if verbose:
        print(f"\n▶ Scenario C — {XOPT_TAG}  ({n_runs} runs)")

    for run_i in range(n_runs):
        bl = ExcelElements(EXCEL_PATH).create_beamline()[:beamline_len]
        evaluate, _, _ = make_felsim_evaluator(bl, VARS, OBJ, bounds=CURRENT_BOUNDS)
        X, st = run_xopt_bo(evaluate, var_names, n_init, n_steps, seed=seed + run_i * 997,
                            label="Scenario C", verbose=verbose)
        xopt_data[run_i + 1] = X.data.copy()
        best = X.data.iloc[st["best_eval"] - 1] if st["best_eval"] > 0 else None
        raw_mse = X.data["MSE"].to_numpy(dtype=float)                   # NaN where FELsim failed
        rec = {"run": run_i + 1, "method": XOPT_TAG, "scenario": "C", "method_tag": XOPT_TAG,
               "final_mse": st["final_mse"], "nfev": st["nfev"], "nit": st["nfev"],
               "wall_time": st["wall_time"], "success": st["success"],
               "n_to_eps": st["n_to_eps"], "n_failed": st["n_failed"], "n_fallback": st["n_fallback"],
               "mse_curve": [float(v) for v in raw_mse],
               "best_mse_curve": [float(v) for v in np.minimum.accumulate(np.nan_to_num(raw_mse, nan=np.inf))],
               "result_x": st["best_x"], **st["best_x"],
               **{n: (float(best[n]) if best is not None else np.nan) for n in out_names}}
        results.append(rec)
        if verbose:
            print(f"  {XOPT_TAG:22s}  run {run_i + 1}/{n_runs}  MSE={st['final_mse']:.3e}  "
                  f"nfev={st['nfev']:<4} t={st['wall_time']:.2f}s  "
                  f"(failed evals: {st['n_failed']}, random fallbacks: {st['n_fallback']})")

    df_C = pd.DataFrame(results)

    # summary in the spirit of run_scenario_C's final_summary: metrics + champion currents vs reference
    best_run = df_C.loc[df_C["final_mse"].fillna(np.inf).idxmin()]     # all-NaN safe (row 0 -> NaN table)
    final_summary = pd.DataFrame([{
        "method_tag": XOPT_TAG,
        "conv_rate": float(df_C["success"].mean()),
        "nfev_mean": float(df_C["nfev"].mean()),
        "wall_time_mean": float(df_C["wall_time"].mean()),
        "final_mse_best": float(best_run["final_mse"]),
        **{k: v for n in var_names for k, v in ((n, float(best_run[n])),)},
        **{f"{n}_ref": ref_currents[elem] for elem, v in VARS.items() for n in [v[0]]},
        **{n: float(best_run[n]) for n in out_names},
    }])
    return {XOPT_TAG: results}, df_C, final_summary, xopt_data


def display_scenario_C_xopt_summary(final_summary_C_xopt, C_VARS, FELSIM_S1_CURRENTS):
    print("\n" + "=" * 100)
    print(f" 🎯 SCENARIO C ({XOPT_TAG}): Robustness (Tol < {EPSILON}) & Discovered Physics State vs Reference")
    print("=" * 100)
    s = final_summary_C_xopt.iloc[0]
    print(f" conv_rate={s.conv_rate:.0%}   nfev_mean={s.nfev_mean:.1f}   wall_time_mean={s.wall_time_mean:.1f} s"
        f"   best MSE={s.final_mse_best:.3e}")

    # champion currents vs reference (one row per quad, easier to read than the wide table)
    cur_names = list(dict.fromkeys(v[0] for v in C_VARS.values()))
    currents_C_xopt = pd.DataFrame({
        "element": list(C_VARS.keys()), "variable": cur_names,
        "I_xopt (A)": [float(s[n]) for n in cur_names],
        "I_ref (A)": [FELSIM_S1_CURRENTS[e] for e in C_VARS],
    })
    display(currents_C_xopt.round(4))

    # measured quantities of the champion point vs their goals
    goals = [(f"{e}_{t['measure'][1]}_{t['measure'][0]}", t["goal"]) for e, ts in C_OBJ.items() for t in ts]
    measured_C_xopt = pd.DataFrame({"quantity": [g[0] for g in goals],
                                    "measured": [float(s[g[0]]) for g in goals],
                                    "goal": [float(g[1]) for g in goals]})
    display(measured_C_xopt.round(4))



def plot_scenario_C_xopt_convergence(df_C_xopt, xopt_data_C, N_INIT_C):

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bo_convergence_plot(ax, xopt_data_C[int(df_C_xopt.run.max())], N_INIT_C)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("MSE")
    ax.set_title(f"Scenario C — Xopt BO convergence ({N_INIT_C} random + {N_BO_STEPS_C} BO evaluations)")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()