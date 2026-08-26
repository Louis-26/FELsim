import torch
import pickle
from functools import partial
import concurrent.futures
from collections import defaultdict
import pandas as pd
import numpy as np

torch.set_default_dtype(torch.float64)

# cease warnings
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'where' used without 'out'.*",
    category=UserWarning,
    module="scipy.optimize._trustregion_constr"
)

warnings.filterwarnings(
    "ignore",
    message=".*delta_grad == 0.0.*",
    category=UserWarning,
    module="scipy.optimize._differentiable_functions"
)

# ── 0. Imports & path setup ─────────────────────────────────────────
import sys, os, copy, time, warnings

# sys.path.insert(0, "/home/niels/FELsim_clone/backend")
CURRENT_DIR = os.getcwd()
BACKEND_DIR = CURRENT_DIR if os.path.basename(CURRENT_DIR) == "backend" else os.path.abspath(
    os.path.join(CURRENT_DIR, "../../backend"))
sys.path.insert(0, BACKEND_DIR)
# print(BACKEND_DIR)
# os.makedirs(os.path.join(CURRENT_DIR, "../../results"), exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from ebeam import beam as ebeam_class
from beamline import lattice
from excelElements import ExcelElements
from beamOptimizer import beamOptimizer
from evolutionPlotter import EvolutionPlotter
from felsimAdapter import FELsimAdapter

# ── Shared constants ─────────────────────────────────────────────────
# EXCEL_PATH     = "/home/niels/FELsim_clone/beam_excel/Beamline_elements_3.xlsx"
# EXCEL_PATH     = "/home/niels/FELsim_clone/beam_excel/Beamline_elements_3.xlsx"
EXCEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../beam_excel/Beamline_elements_3.xlsx"))

BEAM_ENERGY = 40.0  # MeV as the energy for a single electron
N_PARTICLES = 1000  # number of electrons in the beam for simulation
SEED = 42  # random seed for reproducibility
CURRENT_BOUNDS = (0.01, 1.5)  # current bound in the experiment heuristically

# ── Relativistic factors ─────────────────────────────────────────────
relat = lattice(1, fringeType=None)
relat.setE(E=BEAM_ENERGY)  # set the energy of the relativistic factor
GAMMA_REL = relat.gamma  # Lorentz factor as the mass ratio of the electron and the static electron
BETA_REL = relat.beta  # standardized electron velocity
NORM = GAMMA_REL * BETA_REL  # electron momentom ratio between the current electron and the static one

# ── Undulator matching targets (MkV FEL, K=1.2, λ_u=2.3 cm) ─────────
K = 1.2  # Undulator Parameter
LAMBDA_U = 2.3e-2  # wavelength, unit: m
## twiss parameters
BETA_YM = GAMMA_REL / (K * 2 * np.pi / LAMBDA_U)  # matched β_y at UND entrance, beam size in y direction
BETA_XM = 1.4  # m  — matched β_x at UND entrance, beam size at the x direction
ALPHA_XM = 0.47  # — Courant-Snyder α_x at UND entrance, beam size converge at the direction x
ALPHA_YM = 0.0  # — Courant-Snyder α_y at UND entrance, beam waist is achieved at the y direction

# ── Beam parameters ─────────────────────────────────────────────────
epsilon_n = 8.0  # π·mm·mrad  normalised emittance, inner attribute of electron gun
x_std = 0.8  # mm, initial standard deviation at x direction
y_std = 0.8  # mm, initial standard deviation at y direction
f_RF = 2856e6  # Hz, acceleration frequency
bunch_spread_ps = 2.0  # ps, beam length
energy_spread_pct = 0.5  # %, energy standard deviation inside the beam
h = 5e9  # 1/s  chirp, longitudinal chirp for beam compression

epsilon = epsilon_n / NORM  # geometric emittance
x_prime_std = epsilon / x_std  # alpha_x=0 at the initial point, so that sigma_x * sigma_x'=epsilon
y_prime_std = epsilon / y_std  # alpha_y=0 at the initial point, so that sigma_y * sigma_y'=epsilon
tof_std = bunch_spread_ps * 1e-9 * f_RF  # Time of Flight Standard Deviation, change from cycle to mili-cycle, measuring how much the electron bunch occupies the cycle
energy_std = energy_spread_pct * 10  # transform from percent to per mill

ebeam_obj = ebeam_class()
PARTICLES = ebeam_obj.gen_6d_gaussian(
    0,
    [x_std, x_prime_std, y_std, y_prime_std, tof_std, energy_std],
    N_PARTICLES,
)
# Apply longitudinal chirp
PARTICLES[:, 5] += h * (PARTICLES[:, 4] / f_RF)

# ── Load beamline & map quad indices ───────────────────────────────
beamline_full = ExcelElements(EXCEL_PATH).create_beamline()

import time
import copy
import numpy as np
import pandas as pd

def identity_func(x):
    return x

def method_label(method, label=None):
    if label:
        return label
    if isinstance(method, str):
        return method
    return getattr(method, "__name__", str(method))


# ── Benchmark runner ────────────────────────────────────────────────
def run_benchmark(scenario_name, beamline_slice_len, particles, seg_var, obj, bounds,
                  method, n_runs=5, SEED=42, seed_offset=0, jac=None, options=None, verbose=True, method_tag=None,
                  use_log=False, use_epsilon=1e-13, noise=False, sigma=None):
    """Run beamOptimizer.calc() n_runs times from uniform-random starting currents."""
    var_names = list({seg_var[i][0] for i in seg_var})
    results = []
    method_name = method_label(method, method_tag)

    for run_i in range(n_runs):
        
        bl = ExcelElements(EXCEL_PATH).create_beamline()[:beamline_slice_len]
        p = particles.clone()
        rng = np.random.default_rng(SEED + seed_offset + run_i * 997)
        sp = {v: {"bounds": bounds[v],
                  "start": float(rng.uniform(bounds[v][0], bounds[v][1]))}
              for v in var_names} # start point for all variables
        start_x = {v: sp[v]["start"] for v in var_names}
        obj_copy = copy.deepcopy(obj)
        opti = beamOptimizer(bl, p, use_log=use_log, log_epsilon=use_epsilon, noise=noise, sigma=sigma)
        t0 = time.perf_counter()

        try:
            res = opti.calc(method, seg_var, sp, obj_copy, jac=jac, options=options,
                            printResults=False, plotProgress=False
                            )
            wall = time.perf_counter() - t0

            result_x_dict = dict(zip(var_names, res.x))
            # i_1 = result_x_dict.get("I", np.nan)
            # i_3 = result_x_dict.get("I2", np.nan)

            def safe_get_measured(idx):
                if idx in opti.objectives and len(opti.objectives[idx]) > 0:
                    val = opti.objectives[idx][0].get("measured", np.nan)
                    return float(val.item() if hasattr(val, 'item') else val)
                return np.nan

            # alpha_x = safe_get_measured(8)
            # alpha_y = safe_get_measured(9)
            
            output_dict={
                "run": run_i + 1, "method": method_name, "scenario": scenario_name,
                "final_mse": (float(opti.plotMSE[-1]) if (use_log and opti.plotMSE) else float(res.fun)),
                "nfev": int(getattr(res, 'nfev', len(opti.plotMSE))),
                "nit": int(getattr(res, 'nit', -1)),
                "wall_time": round(wall, 3),
                "mse_curve": list(opti.plotMSE), "success": bool(res.success),
                "start_x": start_x, "result_x": result_x_dict,
            }
            
            # include current values
            for k, v in result_x_dict.items():
                output_dict[k] = v
            
            # include measured
            for eval_pos in obj.keys():
                measured_val = safe_get_measured(eval_pos)
                name=str(eval_pos)+"_"+"_".join(reversed(obj[eval_pos][0]["measure"]))
                output_dict[name] = measured_val
    
            results.append(output_dict)
        except Exception as exc:
            wall = time.perf_counter() - t0
            results.append({
                "run": run_i + 1, "method": method_name, "scenario": scenario_name,
                "final_mse": float("nan"), "nfev": -1, "nit": -1,
                "wall_time": round(wall, 3), "mse_curve": [], "success": False,
                "start_x": start_x, "result_x": {}, "error": str(exc)
            })
            if verbose: print(f"  {method_name} run {run_i + 1} FAILED: {exc}")
            continue
        if verbose:
            print(f"  {method_name:22s}  run {run_i + 1}/{n_runs}  "
                  f"MSE={results[-1]['final_mse']:.3e}  nit={results[-1]['nit']:<3} "
                  f"nfev={results[-1]['nfev']:<4} t={wall:.2f}s")
        if (run_i+1)%10==0:
            print(f"{run_i+1}/{n_runs} runs completed for {method_name} in scenario {scenario_name}.")
    return results


def results_to_df(results_list, current_name=["I_1", "I_3"], evalPos_name=["alpha_x", "alpha_y"]):
    keep = ('run', 'method', 'scenario', 'method_tag', 'final_mse', 'nfev', 'nit',
            'wall_time', 'success', *current_name, *evalPos_name
            # 'I_1', 'I_3', 'alpha_x', 'alpha_y'
            )
    return pd.DataFrame([{k: r[k] for k in keep if k in r} for r in results_list])


def plot_stat_convergence(results_by_tag, title="Convergence", figsize=None,
                          colors=None, ax=None, save_path=None, scale="log",
                          convergence_epsilon=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize or (7, 4.5))
    else:
        fig = ax.get_figure()

    try:
        cmap = plt.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.get_cmap("tab10")

    if colors is None: colors = [cmap(i) for i in range(len(results_by_tag))]

    legend_handles = []
    for (tag, res_list), col in zip(results_by_tag.items(), colors):
        n_t = len(res_list)

        if convergence_epsilon is None:
            converged_iter = [r for r in res_list if r.get('mse_curve')]
        else:
            converged_iter = [r for r in res_list
                              if r.get('mse_curve')
                              and np.isfinite(r['mse_curve'][-1])
                              and r['mse_curve'][-1] < convergence_epsilon]

        n_s = len(converged_iter)
        curves = [r['mse_curve'] for r in converged_iter]

        if not curves:
            legend_handles.append(Line2D([0], [0], color=col, linewidth=2,
                                         label=f"{tag}  (0/{n_t} converged)"))
            continue

        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves], dtype=np.float64)

        padded = np.clip(padded, a_min=1e-20, a_max=1e100)

        evals = np.arange(1, max_len + 1)

        log_padded = np.log10(padded)
        log_mean = np.mean(log_padded, axis=0)
        log_std = np.std(log_padded, axis=0)
        mean = 10 ** log_mean
        band_lower = 10 ** (log_mean - log_std)
        band_upper = 10 ** (log_mean + log_std)

        ax.plot(evals, mean, color=col, linewidth=2.0, zorder=3)
        ax.fill_between(evals, np.maximum(band_lower, 1e-20), band_upper,
                        color=col, alpha=0.20, zorder=2)

        legend_handles.append(Line2D([0], [0], color=col, linewidth=2,
                                     label=f"{tag}  ({n_s}/{n_t} converged)"))

    ax.set_yscale(scale)
    ax.set_xlabel('Function evaluations (Forward Passes)', fontsize=11)
    ax.set_ylabel('Normalised MSE', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9, loc='upper right')
    ax.grid(True, which='both', alpha=0.25)
    ax.tick_params(labelsize=9)
    if standalone:
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
    return ax


def run_scenario_A(CURRENT_BOUNDS, EPSILON, N_RUNS_A, FELSIM_S1_CURRENTS, A_OBJ, A_VARS, A_BEAMLINE_LEN, METHODS_A, SEED=42,
                   scale="log", options=None, METHOD_OPTIONS={},
                   use_log=False, use_epsilon=1e-13, noise = False, sigma=None, plot_curve=True, verbose=True
                   ):
    A_BOUNDS = {i[0]: CURRENT_BOUNDS for i in (A_VARS.values())}

    # get current variables 
    current_name = list(dict.fromkeys([info[0] for info in A_VARS.values()]))
    ref_I_name = [name + "_ref" for name in current_name]
    
    # get evalPos_parameter for measured values
    evalPos_parameter = [str(i)+"_"+"_".join(reversed(j[0]["measure"])) for i,j in A_OBJ.items()]
    
    results_A = {}
    if verbose:
        print(f"use_log: {use_log}")
        print(f"use noise: {noise}")
    if noise and verbose:
        print(f"noise standard deviation: {sigma[0].item()}")
    for method_spec in METHODS_A:
        if len(method_spec) == 3:
            method, jac, label = method_spec
        else:
            method, jac = method_spec
            label = None
        options = METHOD_OPTIONS.get(method_label(method, label), None)
        method_name = method_label(method, label)
        tag = method_name + ("+jac" if jac else "")
        if verbose:
            print(f"\n▶ Scenario A — {tag}  ({N_RUNS_A} runs)")

        # Execute benchmark (Note: passing 'jac' here prepares for PyTorch analytical gradients later)
        res = run_benchmark(
            "A",
            A_BEAMLINE_LEN,
            PARTICLES,
            A_VARS,
            A_OBJ,
            A_BOUNDS,
            method=method,
            n_runs=N_RUNS_A,
            SEED=SEED,
            seed_offset=0,
            jac=jac,
            options=options,
            method_tag=method_name,
            use_log=use_log,
            use_epsilon=use_epsilon,
            noise=noise,
            sigma=sigma,
            verbose=verbose
        )

        for r in res:
            r["method_tag"] = method_name

        results_A[method_name] = res
    # print("results_A: ", results_A)
    # Flatten and merge nested results into a single DataFrame
    df_A = results_to_df([r for v in results_A.values() for r in v], current_name, evalPos_parameter)
    # print("df_A: ",df_A)
    # ==============================================================================
    # 🎯 CORE FIX: Patch the bug where derivative-free algorithms return negative iterations
    # ==============================================================================
    # Since derivative-free algorithms lack internal line searches, their number
    # of function evaluations (nfev) is mathematically equivalent to their true exploration steps (nit).
    # If iteration count < 0 is detected, directly extract 'nfev' as the true iteration count!
    if 'nit' in df_A.columns and 'nfev' in df_A.columns:
        df_A.loc[df_A['nit'] < 0, 'nit'] = df_A.loc[df_A['nit'] < 0, 'nfev']


    # 1. Core Evaluation: Check if each run successfully converged below the target MSE threshold
    df_A['is_converged'] = df_A['final_mse'] < EPSILON

    # 2. Calculate average performance metrics
    def _geom_mean_converged(s):
        conv = s[s < EPSILON]
        return 10 ** np.log10(conv).mean() if len(conv) else np.nan

    # (🎯 Note: The redundant 'nfev_mean' has been completely removed as requested)
    summary_stats = df_A.groupby('method_tag').agg(
        conv_rate=('is_converged', 'mean'),
        nit_mean=('nit', 'mean'),
        nfev_mean=('nfev', 'mean'),
        wall_time_mean=('wall_time', 'mean'),
    ).join(
        df_A.groupby('method_tag')['final_mse']
        .apply(_geom_mean_converged)
        .rename('final_mse_mean')
    )

    # 3. Extract the Champion Solution: Find the run with the lowest error for each method,
    #    extracting its exact hardware currents and resulting alpha observables.
    idx_best = df_A.groupby('method_tag')['final_mse'].idxmin()
    best_runs = df_A.loc[idx_best, ['method_tag', *current_name, *evalPos_parameter]].set_index('method_tag')

    # 4. Merge all statistical data
    final_summary = summary_stats.join(best_runs).reset_index()

    for k, var_info in A_VARS.items():
        var_name = var_info[0]
        final_summary[f"{var_name}_ref"] = FELSIM_S1_CURRENTS[k]
        

    
    hybrid_curr_name = [item for pair in zip(current_name, ref_I_name) for item in pair]
    # Reorder columns: Place the AI-predicted currents right next to the expert's Reference values
    final_summary = final_summary[[
        'method_tag', 'conv_rate', 'nit_mean', 'nfev_mean', 'wall_time_mean',
        'final_mse_mean', *hybrid_curr_name, *evalPos_parameter
    ]]

    # 5. Plain text formatted output to bypass Jupyter HTML rendering bugs (missing tables)
    format_dict = {
        'conv_rate': '{:.0%}',
        'nit_mean': '{:.1f}',
        'nfev_mean': '{:.1f}',
        'wall_time_mean': '{:.3f} s',
        'final_mse_mean': '{:.2e}',
        # 'I_1_ref': '{:.4f} A',
        # 'I_1': '{:.4f} A',
        # 'I_3_ref': '{:.4f} A',
        # 'I_3': '{:.4f} A',
        # 'I_10': '{:.4f} A',
        # 'I_10_ref': '{:.4f} A',
        **{f'{current_name}': '{:.4f} A' for current_name in hybrid_curr_name},
        **{f'{evalPos}': '{:.4f}' for evalPos in evalPos_parameter},
    }

    formatted_df = final_summary.copy()

    # Apply formatting conversions in bulk
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "NaN")

    # Print the final beautifully aligned data table
    if verbose:
        print("\n" + "=" * 115)
        print(f" 🎯 SCENARIO A: Robustness (Tol < {EPSILON}) & Discovered Physics State vs Reference")
        print("=" * 115)
        print(formatted_df.to_string(index=False, justify='center'))
        print("=" * 115)

    if noise and sigma is not None:
        exponent = int(np.log10(sigma[0].item()))
        final_title = f"Scenario A: (Target MSE < {EPSILON}) - Noise $\\sigma = 10^{{{exponent}}}$"
    else:
        final_title = f"Scenario A: (Target MSE < {EPSILON}) - Noise Free"
    # Plot the final algorithm stability curve
    if plot_curve:
        plot_stat_convergence(
            results_A,
            title=final_title,
            convergence_epsilon=EPSILON,
            scale=scale
        )
    return results_A, df_A, final_summary


def display_results(results_A, A_VARS, FELSIM_S1_CURRENTS, verbose=True):
    # ── Scenario A — Best optimised quadrupole currents per method ─────
    var_names = [info[0] for info in A_VARS.values()]
    if verbose:
        print("─── Scenario A: Best Optimised Quadrupole Currents per Method ───")
        header_str = f"{'Method':<22} {'Best MSE':>12} "
        header_str += " ".join([f"{v} (A)".rjust(10) for v in var_names])
        print(header_str)
        print("─" * len(header_str))
    for tag, res_list in results_A.items():
        valid = [r for r in res_list
                 if r.get('result_x') and not np.isnan(r['final_mse'])]
        if not valid:
            print(f"  {tag:<20}  no valid run")
            continue
        best = min(valid, key=lambda r: r['final_mse'])
        # I = best['result_x'].get('I', float('nan'))
        # I2 = best['result_x'].get('I2', float('nan'))
        print(f"  {tag:<20} {best['final_mse']:>12.4e}", end="")
        for name in var_names:
            val = best['result_x'].get(name, float('nan'))
            print(f" {val:>10.4f}", end="")      
        
        # print(f"  {tag:<20} {best['final_mse']:>12.4e} {I:>10.4f} {I2:>10.4f}")
        print()
    ref_str_list = [
        f"{info[0]}={FELSIM_S1_CURRENTS.get(pos, float('nan')):.4f} A" 
        for pos, info in A_VARS.items()
    ]
    
    ref_str = ", ".join(ref_str_list)
    
    print(f"  Reference FELsim S1 currents: {ref_str}")
    # Mean best across all runs
    all_valid = [r for v in results_A.values() for r in v
                 if r.get('result_x') and not np.isnan(r['final_mse'])]
    if all_valid:
        best_overall = min(all_valid, key=lambda r: r['final_mse'])
        print(f"\n  Overall best across all methods/runs:")
        print(f"    Method: {best_overall['method_tag']},  MSE={best_overall['final_mse']:.4e}")
        # print(f"    I={best_overall['result_x']['I']:.5f} A,  "
        #       f"I2={best_overall['result_x']['I2']:.5f} A")
        for name in var_names:
            val = best_overall['result_x'][name]
            print(f"    {name}={val:.4f} A", end="\t")
     
        return {name: best_overall['result_x'][name] for name in var_names}
    else:
        return 

def save_results(results_A, df_A, final_summary, path):
    """
    Save all results after scenario A experiments.
    """

    with open(path, "wb") as f:
        pickle.dump({
            "results_A": results_A,
            "df_A": df_A,
            "final_summary": final_summary
        }, f)
    print(f"✓ experiment results saved to {path}")


def load_results(path):
    """
    Load the results from scenario A experiments.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
        df_A = data["df_A"]
        results_A = data["results_A"]
        final_summary = data["final_summary"]
    print(f"✓ experiment results loaded from {path}")
    return results_A, df_A, final_summary


def plot_MSE(df_A):
    """
    Give the scatter plot of the converged (alpha_x, alpha_y) pairs for each method in scenario A,
    with the target (0, 0) highlighted as a red star.
    """

    methods = df_A['method_tag'].unique()
    colors = plt.colormaps.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(10, 10))

    for i, m in enumerate(methods):
        sub = df_A[df_A['method_tag'] == m]
        ax.scatter(sub['alpha_x'], sub['alpha_y'],
                   s=35, alpha=0.6, color=colors(i),
                   edgecolors='white', linewidths=0.4,
                   label=f"{m} (n={len(sub)})")

    ax.scatter(0, 0, marker='*', s=400, color='red', zorder=10,
               edgecolors='black', linewidths=1.2, label='Target (0, 0)')

    ax.set_xscale('symlog', linthresh=1e-9)
    ax.set_yscale('symlog', linthresh=1e-9)

    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.5)

    ax.set_xlabel(r'$\alpha_x$', fontsize=10)
    ax.set_ylabel(r'$\alpha_y$', fontsize=10)
    ax.set_title('Scenario A: Converged $(\\alpha_x, \\alpha_y)$ per Method\n(symlog scale, target = origin)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    plt.show()


def save_landscape(CURRENT_BOUNDS, A_BEAMLINE_LEN, A_VARS, A_OBJ, N_GRID, save_path):
    bl = ExcelElements(EXCEL_PATH).create_beamline()[:A_BEAMLINE_LEN]
    opti = beamOptimizer(bl, PARTICLES.clone())

    SP = {"I": {"bounds": CURRENT_BOUNDS, "start": 0.5},
          "I2": {"bounds": CURRENT_BOUNDS, "start": 0.5}}

    def mse_at(i1, i2, A_VARS, SP, A_OBJ):
        try:
            v = opti.evaluate([i1, i2], A_VARS, SP, copy.deepcopy(A_OBJ))
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    lo, hi = CURRENT_BOUNDS
    I1_vals = np.linspace(lo, hi, N_GRID)
    I2_vals = np.linspace(lo, hi, N_GRID)
    MSE_grid = np.empty((N_GRID, N_GRID))

    t0 = time.perf_counter()
    for j, i2 in enumerate(I2_vals):
        for i, i1 in enumerate(I1_vals):
            MSE_grid[j, i] = mse_at(i1, i2, A_VARS, SP, A_OBJ)
        if (j + 1) % 5 == 0:
            print(f"  row {j + 1}/{N_GRID}   {time.perf_counter() - t0:.0f}s")

    with open(save_path, "wb") as f:
        pickle.dump({"I1_vals": I1_vals, "I2_vals": I2_vals,
                     "MSE_grid": MSE_grid, "N_GRID": N_GRID, "bounds": (lo, hi)}, f)


def save_trajectories(A_BEAMLINE_LEN, N_STARTS, A_VARS, A_OBJ, METHODS_A, METHOD_OPTIONS, save_path):
    rng = np.random.default_rng(2026)
    STARTS = rng.uniform(0.01, 1.5, size=(N_STARTS, 2))
    print(f"{N_STARTS} start points:\n", np.round(STARTS, 3))

    trajectories = {}
    for method, jac in METHODS_A:
        method_trajs = []
        opts = METHOD_OPTIONS.get(method, None)

        for s_idx, start in enumerate(STARTS):
            bl = ExcelElements(EXCEL_PATH).create_beamline()[:A_BEAMLINE_LEN]
            opti = beamOptimizer(bl, PARTICLES.clone())

            traj = [start.copy()]

            def cb(xk, *args, **kwargs):
                traj.append(np.array(xk, dtype=float).copy())

            SP = {"I": {"bounds": CURRENT_BOUNDS, "start": float(start[0])},
                  "I2": {"bounds": CURRENT_BOUNDS, "start": float(start[1])}}

            res = opti.calc(method, A_VARS, SP, copy.deepcopy(A_OBJ),
                            jac=jac, options=opts, callback=cb)

            traj.append(np.array(res.x, dtype=float))
            method_trajs.append({"path": np.array(traj),
                                 "start": start.copy(),
                                 "final_mse": float(res.fun),
                                 "nit": int(getattr(res, 'nit', -1))})
            print(f"{method:14s} start{s_idx + 1} ({start[0]:.2f},{start[1]:.2f})"
                  f"  steps={len(traj):3d}  MSE={res.fun:.2e}  x={np.round(res.x, 4)}")

        trajectories[method] = method_trajs

    with open(save_path, "wb") as f:
        pickle.dump({"trajectories": trajectories, "starts": STARTS}, f)
    print(f"\n✓ trajectories saved ({len(METHODS_A)} methods * {N_STARTS} starts)")


def plot_contour(path):
    # MSE landscape heatmap

    with open(path, "rb") as f:
        L = pickle.load(f)
    I1_vals, I2_vals, MSE_grid = L["I1_vals"], L["I2_vals"], L["MSE_grid"]

    print(f"Grid {L['N_GRID']} {L['N_GRID']}, MSE range [{np.nanmin(MSE_grid):.2e}, {np.nanmax(MSE_grid):.2e}]")

    Z = np.log10(np.maximum(MSE_grid, 1e-16))

    fig, ax = plt.subplots(figsize=(10, 10))

    cf = ax.contourf(I1_vals, I2_vals, Z, levels=40, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r'$\log_{10}(\mathrm{MSE})$', fontsize=12)

    ax.contour(I1_vals, I2_vals, Z, levels=12, colors='white', linewidths=0.4, alpha=0.4)

    ax.plot(0.8218, 1.0430, 'P', color='magenta', markersize=15,
            markeredgecolor='black', label='Reference (0.82, 1.04)', zorder=11)

    ax.set_xlabel(r'$I_1$ (A)', fontsize=13)
    ax.set_ylabel(r'$I_2$ (A)', fontsize=13)
    ax.set_title('Scenario A: MSE Landscape', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.92)
    ax.set_xlim(I1_vals[0], I1_vals[-1])
    ax.set_ylim(I2_vals[0], I2_vals[-1])
    fig.tight_layout()
    plt.show()


def plot_trajectories(landscape_path, trajectories_path):
    with open(landscape_path, "rb") as f:
        L = pickle.load(f)
    with open(trajectories_path, "rb") as f:
        T = pickle.load(f)

    I1_vals, I2_vals, MSE_grid = L["I1_vals"], L["I2_vals"], L["MSE_grid"]
    trajectories, STARTS = T["trajectories"], T["starts"]
    Z = np.log10(np.maximum(MSE_grid, 1e-16))

    methods = list(trajectories.keys())
    start_colors = plt.colormaps.get_cmap('tab10')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.flatten()
    cf = None

    for idx, method in enumerate(methods):
        ax = axes[idx]
        cf = ax.contourf(I1_vals, I2_vals, Z, levels=40, cmap='viridis')
        ax.contour(I1_vals, I2_vals, Z, levels=10, colors='white', linewidths=0.3, alpha=0.3)

        for s_idx, d in enumerate(trajectories[method]):
            path = d["path"];
            c = start_colors(s_idx)
            ax.plot(path[:, 0], path[:, 1], '-', color=c, linewidth=1.3, alpha=0.9)
            ax.plot(path[0, 0], path[0, 1], 'o', color=c, ms=7, mec='white', mew=1.0, zorder=9)
            ax.plot(path[-1, 0], path[-1, 1], '*', color=c, ms=14, mec='black', mew=0.6, zorder=10)

        ax.plot(0.8218, 1.0430, 'P', color='magenta', ms=12, mec='black', zorder=11)
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.set_xlabel(r'$I_1$ (A)', fontsize=11);
        ax.set_ylabel(r'$I_2$ (A)', fontsize=11)
        ax.set_xlim(I1_vals[0], I1_vals[-1]);
        ax.set_ylim(I2_vals[0], I2_vals[-1])

    # add the legend in the last subplot
    axes[5].axis('off')
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=start_colors(i),
                  markeredgecolor='gray', ms=9,
                  label=f"Start {i + 1}: ({STARTS[i, 0]:.2f}, {STARTS[i, 1]:.2f})")
           for i in range(len(STARTS))]
    leg += [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', ms=9, label='○ start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='black', ms=14,
                   label='★ converged'),
            Line2D([0], [0], marker='P', color='w', markerfacecolor='magenta', markeredgecolor='black', ms=12,
                   label='Reference (0.82,1.04)')]
    axes[5].legend(handles=leg, loc='center', fontsize=11, frameon=True, title='Legend')
    if cf is not None:
        fig.colorbar(cf, ax=list(axes), label=r'$\log_{10}(\mathrm{MSE})$', shrink=0.55, aspect=35)
    fig.suptitle('Scenario A: Convergence Trajectories per Method (5 shared random starts)',
                 fontsize=15, fontweight='bold')
    plt.show()


from functools import partial
import concurrent.futures
from collections import defaultdict
import pandas as pd
import numpy as np


def _benchmark_worker(task_params, A_BEAMLINE_LEN, PARTICLES, A_VARS, A_OBJ, A_BOUNDS, 
                      SEED, METHOD_OPTIONS, use_log, use_epsilon, noise, sigma):
    
    method, jac, method_name, run_idx = task_params
    options = METHOD_OPTIONS.get(method_name, None)
    
    try:
        res = run_benchmark(
            "A",
            A_BEAMLINE_LEN,
            PARTICLES,
            A_VARS,
            A_OBJ,
            A_BOUNDS,
            method=method,
            n_runs=1,                  
            SEED=SEED,
            seed_offset=run_idx,       
            jac=jac,
            options=options,
            method_tag=method_name,
            use_log=use_log,
            use_epsilon=use_epsilon,
            noise=noise,
            sigma=sigma,
            verbose=False              
        )
        
        r = res[0]
        r["method_tag"] = method_name
        return method_name, r, True
        
    except Exception as e:
        return method_name, {"method_tag": method_name, "error": str(e)}, False
    
    
def run_scenario_A_parallel(CURRENT_BOUNDS, EPSILON, N_RUNS_A, FELSIM_S1_CURRENTS, A_OBJ, A_VARS, A_BEAMLINE_LEN, METHODS_A, SEED=42,
                   scale="log", METHOD_OPTIONS={},
                   use_log=False, use_epsilon=1e-13, noise = False, sigma=None, plot_curve=True, verbose=True,
                   max_cores=6):  
    
    A_BOUNDS = {i[0]: CURRENT_BOUNDS for i in (A_VARS.values())}
    current_name = list(dict.fromkeys([info[0] for info in A_VARS.values()]))
    ref_I_name = [name + "_ref" for name in current_name]
    evalPos_parameter = [str(i)+"_"+"_".join(reversed(j[0]["measure"])) for i,j in A_OBJ.items()]
    
    if verbose:
        print(f"use_log: {use_log}")
        print(f"use noise: {noise}")
        if noise:
            print(f"noise standard deviation: {sigma[0].item()}")
        print(f"🚀 Utilizing {max_cores} cores, with {len(METHODS_A) * N_RUNS_A} tasks...")

    tasks = []
    for method_spec in METHODS_A:
        if len(method_spec) == 3:
            method, jac, label = method_spec
        else:
            method, jac = method_spec
            label = None
            
        method_name = method_label(method, label)
        
        for run_idx in range(N_RUNS_A):
            tasks.append((method, jac, method_name, run_idx))
            
    worker_func = partial(
        _benchmark_worker, 
        A_BEAMLINE_LEN=A_BEAMLINE_LEN, PARTICLES=PARTICLES, A_VARS=A_VARS, 
        A_OBJ=A_OBJ, A_BOUNDS=A_BOUNDS, SEED=SEED, 
        METHOD_OPTIONS=METHOD_OPTIONS, use_log=use_log, 
        use_epsilon=use_epsilon, noise=noise, sigma=sigma
    )

    results_A = defaultdict(list)
    
    import time
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        for method_name, result_dict, success in executor.map(worker_func, tasks):
            if success:
                results_A[method_name].append(result_dict)
            else:
                if verbose:
                    print(f"⚠️ {method_name} Failed: {result_dict.get('error')}")
                    
    if verbose:
        print(f"✅ {len(tasks)} finish with time cost: {time.time() - t0:.2f} seconds\n")

    df_A = results_to_df([r for v in results_A.values() for r in v], current_name, evalPos_parameter)
    
    if 'nit' in df_A.columns and 'nfev' in df_A.columns:
        df_A.loc[df_A['nit'] < 0, 'nit'] = df_A.loc[df_A['nit'] < 0, 'nfev']

    df_A['is_converged'] = df_A['final_mse'] < EPSILON

    def _geom_mean_converged(s):
        conv = s[s < EPSILON]
        return 10 ** np.log10(conv).mean() if len(conv) else np.nan

    summary_stats = df_A.groupby('method_tag').agg(
        conv_rate=('is_converged', 'mean'),
        nit_mean=('nit', 'mean'),
        nfev_mean=('nfev', 'mean'),
        wall_time_mean=('wall_time', 'mean'),
    ).join(
        df_A.groupby('method_tag')['final_mse']
        .apply(_geom_mean_converged)
        .rename('final_mse_mean')
    )

    idx_best = df_A.groupby('method_tag')['final_mse'].idxmin().dropna() 
    best_runs = df_A.loc[idx_best, ['method_tag', *current_name, *evalPos_parameter]].set_index('method_tag')
    final_summary = summary_stats.join(best_runs).reset_index()

    for k, var_info in A_VARS.items():
        var_name = var_info[0]
        final_summary[f"{var_name}_ref"] = FELSIM_S1_CURRENTS[k]
        
    hybrid_curr_name = [item for pair in zip(current_name, ref_I_name) for item in pair]
    final_summary = final_summary[[
        'method_tag', 'conv_rate', 'nit_mean', 'nfev_mean', 'wall_time_mean',
        'final_mse_mean', *hybrid_curr_name, *evalPos_parameter
    ]]

    # 5. Plain text formatted output
    format_dict = {
        'conv_rate': '{:.0%}',
        'nit_mean': '{:.1f}',
        'nfev_mean': '{:.1f}',
        'wall_time_mean': '{:.3f} s',
        'final_mse_mean': '{:.2e}',
        **{f'{curr}': '{:.4f} A' for curr in hybrid_curr_name},
        **{f'{evalPos}': '{:.4f}' for evalPos in evalPos_parameter},
    }

    formatted_df = final_summary.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "NaN")

    if verbose:
        print("\n" + "=" * 115)
        print(f" 🎯 SCENARIO A: Robustness (Tol < {EPSILON}) & Discovered Physics State vs Reference")
        print("=" * 115)
        print(formatted_df.to_string(index=False, justify='center'))
        print("=" * 115)

    if noise and sigma is not None:
        exponent = int(np.log10(sigma[0].item()))
        final_title = f"Scenario A: (Target MSE < {EPSILON}) - Noise $\\sigma = 10^{{{exponent}}}$"
    else:
        final_title = f"Scenario A: (Target MSE < {EPSILON}) - Noise Free"
        
    if plot_curve:
        plot_stat_convergence(results_A, title=final_title, convergence_epsilon=EPSILON, scale=scale)
        
    return results_A, df_A, final_summary


def run_scenario_C(scenario_name, CURRENT_BOUNDS, EPSILON, N_RUNS, REF_CURRENTS, OBJ, VARS, BEAMLINE_LEN, METHODS, SEED=42,
                              scale="log", options=None, METHOD_OPTIONS=None,
                              use_log=False, use_epsilon=1e-13, noise=False, sigma=None, plot_curve=True, verbose=True):
    """
    Universal single-stage optimization runner (used for Scenarios A, C, etc.).
    Simultaneously optimizes all provided variables over the specified beamline length.
    """
    if METHOD_OPTIONS is None: METHOD_OPTIONS = {}
    
    # 1. Dynamically extract robust column names preserving order
    BOUNDS = {info[0]: CURRENT_BOUNDS for info in VARS.values()}
    current_name = list(dict.fromkeys([info[0] for info in VARS.values()]))
    ref_I_name = [name + "_ref" for name in current_name]
    evalPos_parameter = [str(i) + "_" + "_".join(reversed(j[0]["measure"])) for i, j in OBJ.items()]
    
    results = {}
    
    if verbose:
        print(f"use_log: {use_log} | use_noise: {noise}")
        if noise and sigma is not None:
            print(f"noise standard deviation: {sigma[0].item()}")
            
    for method_spec in METHODS:
        if len(method_spec) == 3:
            method, jac, label = method_spec
        else:
            method, jac = method_spec
            label = None
            
        opts = METHOD_OPTIONS.get(method_label(method, label), None)
        method_name = method_label(method, label)
        tag = method_name + ("+jac" if jac else "")
        
        if verbose:
            print(f"\n▶ Scenario {scenario_name} — {tag}  ({N_RUNS} runs)")

        # Execute benchmark
        res = run_benchmark(
            scenario_name, BEAMLINE_LEN, PARTICLES, VARS, OBJ, BOUNDS,
            method=method, n_runs=N_RUNS, SEED=SEED, seed_offset=0,
            jac=jac, options=opts, method_tag=method_name,
            use_log=use_log, use_epsilon=use_epsilon, noise=noise, sigma=sigma, verbose=verbose
        )

        for r in res:
            r["method_tag"] = method_name
        results[method_name] = res

    # Flatten and merge nested results into a single DataFrame
    df = results_to_df([r for v in results.values() for r in v], current_name, evalPos_parameter)

    # 🎯 CORE FIX: Patch the bug where derivative-free algorithms return negative iterations
    if 'nit' in df.columns and 'nfev' in df.columns:
        df.loc[df['nit'] < 0, 'nit'] = df.loc[df['nit'] < 0, 'nfev']

    # Evaluate convergence
    df['is_converged'] = df['final_mse'] < EPSILON

    def _geom_mean_converged(s):
        conv = s[s < EPSILON]
        return 10 ** np.log10(conv).mean() if len(conv) else np.nan

    summary_stats = df.groupby('method_tag').agg(
        conv_rate=('is_converged', 'mean'),
        nit_mean=('nit', 'mean'),
        nfev_mean=('nfev', 'mean'),
        wall_time_mean=('wall_time', 'mean'),
    ).join(
        df.groupby('method_tag')['final_mse'].apply(_geom_mean_converged).rename('final_mse_mean')
    )

    # Extract the Champion Solution
    idx_best = df.groupby('method_tag')['final_mse'].idxmin().dropna()
    best_runs = df.loc[idx_best, ['method_tag', *current_name, *evalPos_parameter]].set_index('method_tag')
    final_summary = summary_stats.join(best_runs).reset_index()

    # Align Reference Currents safely
    for k, var_info in VARS.items():
        var_name = var_info[0]
        final_summary[f"{var_name}_ref"] = REF_CURRENTS[k]
        
    hybrid_curr_name = [item for pair in zip(current_name, ref_I_name) for item in pair]
    
    # Reorder columns
    final_summary = final_summary[[
        'method_tag', 'conv_rate', 'nit_mean', 'nfev_mean', 'wall_time_mean',
        'final_mse_mean', *hybrid_curr_name, *evalPos_parameter
    ]]

    # Format table for output
    format_dict = {
        'conv_rate': '{:.0%}',
        'nit_mean': '{:.1f}',
        'nfev_mean': '{:.1f}',
        'wall_time_mean': '{:.3f} s',
        'final_mse_mean': '{:.2e}',
        **{f'{curr}': '{:.4f} A' for curr in hybrid_curr_name},
        **{f'{evalPos}': '{:.4f}' for evalPos in evalPos_parameter},
    }

    formatted_df = final_summary.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "NaN")

    if verbose:
        print("\n" + "=" * 125)
        print(f" 🎯 SCENARIO {scenario_name}: Robustness (Tol < {EPSILON}) & Discovered Physics State vs Reference")
        print("=" * 125)
        print(formatted_df.to_string(index=False, justify='center'))
        print("=" * 125)

    if plot_curve:
        final_title = f"Scenario {scenario_name}: (Target MSE < {EPSILON})"
        if noise and sigma is not None:
            exponent = int(np.log10(sigma[0].item()))
            final_title += f" - Noise $\\sigma = 10^{{{exponent}}}$"
        else:
            final_title += " - Noise Free"
            
        plot_stat_convergence(results, title=final_title, convergence_epsilon=EPSILON, scale=scale)
        
    return results, df, final_summary