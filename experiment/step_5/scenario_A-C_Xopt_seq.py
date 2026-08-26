import os, sys, copy, time, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

# FELsim backend + beam / lattice constants (same import path as scenario_A-C.ipynb)
sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(os.getcwd(), "../.."))
from experiments_utils import (PARTICLES, EXCEL_PATH, ExcelElements, beamOptimizer,
                               ALPHA_XM, ALPHA_YM, BETA_XM, BETA_YM)
from configs import *

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator

from utils.Xopt_utils import *
from IPython.display import display
import argparse

evaluate_A, A_var_names, A_out_names = make_felsim_evaluator(bl_full[:A_BEAMLINE_LEN], A_VARS, A_OBJ)

vocs_A = VOCS(
    variables={name: list(CURRENT_BOUNDS_A) for name in A_var_names},   # {'I': [0.01, 1.5], 'I2': [0.01, 1.5]}
    objectives={"MSE": "MINIMIZE"},                                   # BO target: the raw MSE
)
evaluator_A = Evaluator(function=evaluate_A)
generator_A = ExpectedImprovementGenerator(vocs=vocs_A)

X = Xopt(evaluator=evaluator_A, generator=generator_A, vocs=vocs_A)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run Scenarios A, B, and C")
    parser.add_argument("--N_RUNS_A", type=int, default=1, help="Number of runs for Scenario A")
    parser.add_argument("--N_RUNS_B", type=int, default=1, help="Number of runs for Scenario B")
    parser.add_argument("--N_RUNS_C", type=int, default=1, help="Number of runs for Scenario C")
    parser.add_argument("--use_multi", type=int, default=0, help="Whether to use multi-processing")
    parser.add_argument("--SEED", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_cores", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Worker processes to use when --use_multi 1 "
                                "(capped at the number of tasks)")
    args=parser.parse_args()
    if args.use_multi==0: # sequential execution
        # scenario A
        run_scenario_A_xopt(X, N_INIT_A, N_BO_STEPS_A, A_out_names,FELSIM_S1_CURRENTS, SEED)
        plot_scenario_A_xopt_convergence(X, N_INIT_A)
        
        # scenario B
        results_B_xopt, df_B_xopt, final_beamline_B_xopt, xopt_data_B = run_scenario_B_xopt(
        STAGES_B, CURRENT_BOUNDS_BC, n_runs=N_RUNS_B, n_init=N_INIT_B, n_steps=N_BO_STEPS_B, seed=SEED, verbose=True)

        display(df_B_xopt[["run", "stage", "stage_name", "method_tag", "final_mse",
                        "nfev", "wall_time", "success", "n_to_eps", "n_failed", "n_fallback"]])
        plot_scenario_B_xopt_convergence(STAGES_B, df_B_xopt, xopt_data_B, N_INIT_B)
        
        # scenario C
        results_C_xopt, df_C_xopt, final_summary_C_xopt, xopt_data_C = run_scenario_C_xopt(
        C_VARS, C_OBJ, C_BEAMLINE_LEN, FELSIM_S1_CURRENTS, CURRENT_BOUNDS_BC,
        n_runs=N_RUNS_C, n_init=N_INIT_C, n_steps=N_BO_STEPS_C, seed=SEED, verbose=True)
        display_scenario_C_xopt_summary(final_summary_C_xopt, C_VARS, FELSIM_S1_CURRENTS)
        plot_scenario_C_xopt_convergence(df_C_xopt, xopt_data_C, N_INIT_C)
    else: # parallel execution
        pass