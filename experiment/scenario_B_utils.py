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
CURRENT_BOUNDS_BC = (0.01, 5.0)  # current bound in the experiment heuristically

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


def run_scenario_B(STAGES_B, PARTICLES, EXCEL_PATH, METHODS_B, N_RUNS_B=1, SEED=42, METHOD_OPTIONS=None, verbose=True):
    """
    Sequential optimization pipeline specifically designed for Scenario B 
    (supports multiple independent random restarts).
    The optimized currents from each stage are preserved and used as the 
    initial physical state for the subsequent stage.
    """
    if METHOD_OPTIONS is None: METHOD_OPTIONS = {}
    
    results_all_methods = {}
    df_list = []

    for method_spec in METHODS_B:
        if len(method_spec) == 3:
            method, jac, label = method_spec
        else:
            method, jac = method_spec
            label = None

        method_name = method_label(method, label)
        options = METHOD_OPTIONS.get(method_name, None)

        if verbose:
            print(f"\n▶ 🚀 Starting Scenario B (11-Stage Sequential) — Method: {method_name} ({N_RUNS_B} runs total)")

        stage_results = []
        t_total = time.perf_counter()

        # 🎯 Outer Loop: Execute the full pipeline N_RUNS_B times
        for run_i in range(N_RUNS_B):
            if verbose and N_RUNS_B > 1:
                print(f"  ── Run {run_i+1}/{N_RUNS_B} ──")
                
            # Re-initialize the global beamline and particles for each run to ensure independence
            bl_run = ExcelElements(EXCEL_PATH).create_beamline()
            p_run = PARTICLES.clone()
            
            # [Optional] Initialize rng here if random perturbations are needed for starting points
            rng = np.random.default_rng(SEED + run_i * 997)

            # Execute the 11 stages sequentially
            for s_i, (lbl, seg_var, obj, bl_len, fixed_sp) in enumerate(STAGES_B):
                sp = copy.deepcopy(fixed_sp)
                obj_copy = copy.deepcopy(obj)
                bl_stage = bl_run[:bl_len]

                opti = beamOptimizer(bl_stage, p_run.clone())
                t0 = time.perf_counter()

                try:
                    res = opti.calc(method, seg_var, sp, obj_copy, jac=jac, options=options,
                                    printResults=False, plotProgress=False)
                    wall = time.perf_counter() - t0

                    # 🎯 Crucial Handoff: Write optimized currents back to the global beamline for the next stage
                    for elem_idx, var_info in seg_var.items():
                        var_name = var_info[0]
                        var_idx = opti.variablesToOptimize.index(var_name)
                        setattr(bl_run[elem_idx], 'current', float(res.x[var_idx]))

                    # 🎯 Hardcoded symmetry constraint for Stage 5 (Double Triplet)
                    if s_i == 4:
                        bl_run[43].current = bl_run[33].current
                        bl_run[41].current = bl_run[35].current
                        bl_run[39].current = bl_run[37].current

                    # Extract evaluation and iteration counts
                    nfev = int(getattr(res, 'nfev', len(opti.plotMSE)))
                    nit = int(getattr(res, 'nit', nfev))
                    if nit < 0: nit = nfev

                    stage_dict = {
                        "run": run_i + 1,
                        "stage": s_i + 1,
                        "stage_name": lbl,
                        "method_tag": method_name,
                        "final_mse": float(res.fun),
                        "nfev": nfev,
                        "nit": nit,
                        "wall_time": round(wall, 3),
                        "success": bool(res.success)
                    }
                    stage_results.append(stage_dict)
                    df_list.append(stage_dict)

                    if verbose:
                        print(f"    {lbl:25s} MSE={res.fun:.3e}  nfev={nfev:<4} t={wall:.2f}s")

                except Exception as exc:
                    if verbose:
                        print(f"    {lbl:25s} ❌ FAILED: {exc}")
                    break # Drop the baton: Break out of the current run's pipeline and proceed to the next run_i

        if verbose:
            print(f"✓ {method_name} pipeline execution completed, total time: {time.perf_counter() - t_total:.2f}s")

        results_all_methods[method_name] = stage_results

    # Convert results into a DataFrame
    df_B = pd.DataFrame(df_list)
    return results_all_methods, df_B, bl_run