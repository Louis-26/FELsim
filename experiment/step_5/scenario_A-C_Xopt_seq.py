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
from utils.Xopt_parallel_utils import (
    run_scenario_single_xopt_parallel,
    run_scenario_B_xopt_parallel,
    currents_to_beamline_view,
    plot_xopt_restarts_convergence,
)
from IPython.display import display
import argparse

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
    parser.add_argument("--q", type=int, default=1,
                        help="Candidates evaluated per BO step (Xopt's Evaluator.max_workers). "
                             "1 = plain sequential BO inside each restart; q>1 turns every "
                             "step into batched q-EI, so the pool holds (max_cores//q)*q "
                             "processes. Only pays off when one FELsim evaluation costs much "
                             "more than a GP fit (Scenario C, not Scenario A).")
    args=parser.parse_args()
    if args.use_multi==0: # sequential execution
        # scenario A
        evaluate_A, A_var_names, A_out_names = make_felsim_evaluator(
            bl_full[:A_BEAMLINE_LEN], A_VARS, A_OBJ)
        vocs_A = VOCS(
            variables={name: list(CURRENT_BOUNDS_A) for name in A_var_names},   # {'I': [0.01, 1.5], 'I2': [0.01, 1.5]}
            objectives={"MSE": "MINIMIZE"},                                   # BO target: the raw MSE
        )
        X = Xopt(evaluator=Evaluator(function=evaluate_A),
                 generator=ExpectedImprovementGenerator(vocs=vocs_A), vocs=vocs_A)

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
        # Same three problems, same FELsim objective, same raw-MSE target and the same record
        # layout — only the scheduling changes. A BO loop cannot be split (step k+1 needs the
        # result of step k), so the parallel axis is the *restart*: one independent Xopt object
        # per process, each with its own random init and its own GP. For scenario B a task is a
        # whole 11-stage pipeline, since stage k+1 starts from the currents stage k wrote back.
        # `--q > 1` adds Xopt's own second axis on top (batched q-EI inside every restart).
        print(f"🚀 multiprocessing on up to {args.max_cores} of {os.cpu_count()} cores"
              f"   |   q = {args.q} candidate(s) per BO step")

        # scenario A
        results_A_xopt, df_A_xopt, final_summary_A_xopt, xopt_data_A, perf_A = \
            run_scenario_single_xopt_parallel(
                "A",
                VARS=A_VARS,
                OBJ=A_OBJ,
                BEAMLINE_LEN=A_BEAMLINE_LEN,
                CURRENT_BOUNDS=CURRENT_BOUNDS_A,
                EPSILON=EPSILON,
                N_RUNS=args.N_RUNS_A,
                PARTICLES=PARTICLES,
                REF_CURRENTS=FELSIM_S1_CURRENTS,
                N_INIT=N_INIT_A,
                N_STEPS=N_BO_STEPS_A,
                SEED=args.SEED,
                max_cores=args.max_cores,
                q=args.q,
            )
        # (display_scenario_C_xopt_summary is hard-wired to C_OBJ, so show A's summary directly)
        display(final_summary_A_xopt)
        plot_xopt_restarts_convergence(
            xopt_data_A, N_INIT_A,
            f"Scenario A — Xopt BO, {args.N_RUNS_A} parallel restarts "
            f"({N_INIT_A} random + {N_BO_STEPS_A} BO evaluations each)")

        # scenario B — parallel ACROSS pipelines, the 11 stages stay sequential inside one
        results_B_xopt, df_B_xopt, final_currents_B, xopt_data_B, perf_B = \
            run_scenario_B_xopt_parallel(
                STAGES_B=STAGES_B,
                CURRENT_BOUNDS=CURRENT_BOUNDS_BC,
                EPSILON=EPSILON,
                N_RUNS_B=args.N_RUNS_B,
                PARTICLES=PARTICLES,
                N_INIT=N_INIT_B,
                N_STEPS=N_BO_STEPS_B,
                SEED=args.SEED,
                max_cores=args.max_cores,
                q=args.q,
            )
        display(df_B_xopt[["run", "stage", "stage_name", "method_tag", "final_mse",
                           "nfev", "wall_time", "success", "n_to_eps", "n_failed", "n_fallback"]])
        # both helpers below index every stage of the last run, so they need a complete pipeline
        last_run = int(df_B_xopt.run.max())
        last_complete = (df_B_xopt.run == last_run).sum() == len(STAGES_B)
        if final_currents_B and last_complete:
            # the workers ship currents, not beamline objects; restore the `bl[i].current`
            # interface the display helper expects (last pipeline = the one df_B reports on)
            display_scenario_B_xopt_summary(
                df_B_xopt, STAGES_B,
                currents_to_beamline_view(final_currents_B[-1]), FELSIM_S1_CURRENTS)
            plot_scenario_B_xopt_convergence(STAGES_B, df_B_xopt, xopt_data_B, N_INIT_B)
        else:
            print(f"⚠️ pipeline {last_run} did not complete all {len(STAGES_B)} stages — "
                  f"skipping the per-stage summary/plot")

        # scenario C
        results_C_xopt, df_C_xopt, final_summary_C_xopt, xopt_data_C, perf_C = \
            run_scenario_single_xopt_parallel(
                "C",
                VARS=C_VARS,
                OBJ=C_OBJ,
                BEAMLINE_LEN=C_BEAMLINE_LEN,
                CURRENT_BOUNDS=CURRENT_BOUNDS_BC,
                EPSILON=EPSILON,
                N_RUNS=args.N_RUNS_C,
                PARTICLES=PARTICLES,
                REF_CURRENTS=FELSIM_S1_CURRENTS,
                N_INIT=N_INIT_C,
                N_STEPS=N_BO_STEPS_C,
                SEED=args.SEED,
                max_cores=args.max_cores,
                q=args.q,
            )
        display_scenario_C_xopt_summary(final_summary_C_xopt, C_VARS, FELSIM_S1_CURRENTS)
        plot_scenario_C_xopt_convergence(df_C_xopt, xopt_data_C, N_INIT_C)