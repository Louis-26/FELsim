import os, sys

# ── Multiprocessing hygiene — MUST come before numpy / torch are imported ──
# One BLAS thread per process: P worker processes each running a multi-threaded BLAS would
# oversubscribe the cores and end up slower than the sequential run.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
# Spawned children inherit this. `run_benchmark` picks its random start vector by iterating a
# *set of strings*, whose order follows Python's per-interpreter hash randomisation — pinning
# the seed makes a given run_idx mean the same start point in every worker and every run.
os.environ["PYTHONHASHSEED"] = "0"

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..'))

from experiments_utils import *
from configs import *
from scenario_B_utils import run_scenario_B
from parallel_utils import run_scenario_single_parallel, run_scenario_B_parallel
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
    args=parser.parse_args()
    if args.use_multi==0: # sequential execution
        # scenario A
        results_A, df_A, final_summary = run_scenario_A(
            CURRENT_BOUNDS,
            EPSILON,
            args.N_RUNS_A,
            FELSIM_S1_CURRENTS,
            A_OBJ,
            A_VARS,
            A_BEAMLINE_LEN,
            METHODS_A,
            scale="log",
            METHOD_OPTIONS=METHOD_OPTIONS,
        )
        display_results(results_A, A_VARS, FELSIM_S1_CURRENTS)

        # scenario B
        results_B, df_B, final_beamline_B = run_scenario_B(
            STAGES_B=STAGES_B,
            PARTICLES=PARTICLES,
            EXCEL_PATH=EXCEL_PATH,
            METHODS_B=METHODS_B,
            N_RUNS_B=args.N_RUNS_B,         
            METHOD_OPTIONS=METHOD_OPTIONS,
            verbose=True
        )

        display(df_B)
        
        # scenario C
        results_C, df_C, final_summary_C = run_scenario_C(
            scenario_name="C",
            CURRENT_BOUNDS=CURRENT_BOUNDS_BC,
            EPSILON=EPSILON,
            N_RUNS=args.N_RUNS_C,
            REF_CURRENTS=FELSIM_S1_CURRENTS,
            OBJ=C_OBJ,
            VARS=C_VARS,
            BEAMLINE_LEN=C_BEAMLINE_LEN,
            METHODS=METHODS_C,
            METHOD_OPTIONS=METHOD_OPTIONS,
            verbose=True
        )
    else: # multi-processing execution
        # Same three scenarios, same optimisers and objectives — only the scheduling differs.
        # Each (method, random start) pair is an independent task; for scenario B a task is a
        # whole 11-stage pipeline, because the stages hand their currents on to each other.
        print(f"🚀 multiprocessing on up to {args.max_cores} of {os.cpu_count()} cores")

        # scenario A
        results_A, df_A, final_summary, perf_A = run_scenario_single_parallel(
            "A",
            VARS=A_VARS,
            OBJ=A_OBJ,
            BEAMLINE_LEN=A_BEAMLINE_LEN,
            CURRENT_BOUNDS=CURRENT_BOUNDS_A,
            EPSILON=EPSILON,
            N_RUNS=args.N_RUNS_A,
            METHODS=METHODS_A,
            PARTICLES=PARTICLES,
            REF_CURRENTS=FELSIM_S1_CURRENTS,
            METHOD_OPTIONS=METHOD_OPTIONS,
            SEED=args.SEED,
            max_cores=args.max_cores,
        )
        display_results(results_A, A_VARS, FELSIM_S1_CURRENTS)

        # scenario B
        results_B, df_B, final_currents_B, perf_B = run_scenario_B_parallel(
            STAGES_B=STAGES_B,
            PARTICLES=PARTICLES,
            METHODS_B=METHODS_B,
            N_RUNS_B=args.N_RUNS_B,
            METHOD_OPTIONS=METHOD_OPTIONS,
            SEED=args.SEED,
            max_cores=args.max_cores,
            EPSILON=EPSILON,
        )
        display(df_B)

        # scenario C
        results_C, df_C, final_summary_C, perf_C = run_scenario_single_parallel(
            "C",
            VARS=C_VARS,
            OBJ=C_OBJ,
            BEAMLINE_LEN=C_BEAMLINE_LEN,
            CURRENT_BOUNDS=CURRENT_BOUNDS_BC,
            EPSILON=EPSILON,
            N_RUNS=args.N_RUNS_C,
            METHODS=METHODS_C,
            PARTICLES=PARTICLES,
            REF_CURRENTS=FELSIM_S1_CURRENTS,
            METHOD_OPTIONS=METHOD_OPTIONS,
            SEED=args.SEED,
            max_cores=args.max_cores,
        )