import os, sys
import argparse
current_dir = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../")))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../../backend")))
import numpy as np
import importlib
import experiments_utils, SOBOL_scan

importlib.reload(experiments_utils)
importlib.reload(SOBOL_scan)
from experiments_utils import *
from SOBOL_scan import *
from copy import deepcopy


def identity_func(x):
    return x


A_VARS = {
    1: ["I_1", "current", identity_func],
    3: ["I_3", "current", identity_func],
    10: ["I_10", "current", identity_func],
}

A_OBJ_test_1 = {11: [{"measure": ["x", "dispersion"], "goal": 0.0, "weight": 1.0}]}

A_OBJ_test_2 = {
    11: [
        {"measure": ["x", "dispersion"], "goal": 0.0, "weight": 1.0},
        {"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0},
        {"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0},
        {"measure": ["x", "beta"], "goal": 0.0, "weight": 1.0},
        {"measure": ["y", "beta"], "goal": 0.0, "weight": 1.0},
    ]
}

A_OBJ_test_3 = {
    11: [{"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0}],
}


A_BEAMLINE_LEN = 13


os.makedirs("../../results/benchmark_scan_multi", exist_ok=True)


parameters_case_template = {
    "CURRENT_BOUNDS": CURRENT_BOUNDS,
    "EPSILON": EPSILON,
    "N_RUNS_A": N_RUNS_A,
    "FELSIM_S1_CURRENTS": FELSIM_S1_CURRENTS,
    "A_OBJ": A_OBJ,
    "A_VARS": A_VARS,
    "A_BEAMLINE_LEN": A_BEAMLINE_LEN,
    "METHODS_A": METHODS_A,
    "scale": "log",
    "METHOD_OPTIONS": METHOD_OPTIONS,
    "use_log": use_log,
    "use_epsilon": 1e-4,
    "noise": noise,
    "sigma": sigma,
    "plot_curve": False,
    "verbose": False,
    "threshold_1": 0.01,
    "threshold_2": 0.5,
}


def compute_parameters_case(
    parameters_case_template, A_OBJ, save_dir, N_RUNS_A, sample_num, case_num=1
):
    t0 = time.time()
    parameters_case = deepcopy(parameters_case_template)
    parameters_case.update(
        {
            "A_OBJ": A_OBJ,
            "save_dir": save_dir,
            "N_RUNS_A": N_RUNS_A,
            "sample_num": sample_num
        }
    )
    run_baseline_sim(**parameters_case)
    t1 = time.time()
    print(
        f"Total time cost for case {case_num} with multiprocessing of core number {max_cores_r} for optimization random start, \
and core number {max_cores_s} for SOBOL scan: {t1-t0:.2f} seconds"
    )


save_dir_1 = "../../results/benchmark_scan_multi/SOBOL_results_1.pkl"
save_dir_2 = "../../results/benchmark_scan_multi/SOBOL_results_2.pkl"
save_dir_3 = "../../results/benchmark_scan_multi/SOBOL_results_3.pkl"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run baseline evaluation without multiprocessing")
    parser.add_argument("--n_runs", type=int, default=6, 
                        help="Number of random starts for optimization (N_RUNS_A)")
    parser.add_argument("--sample_num", type=int, default=6, 
                        help="Number of SOBOL scan samples")
    
    args = parser.parse_args()
    
    # case 1 parameters
    compute_parameters_case(
        parameters_case_template,
        A_OBJ_test_1,
        save_dir_1,
        args.n_runs,
        args.sample_num,
        case_num=1,
    )
    # case 2 parameters
    compute_parameters_case(
        parameters_case_template,
        A_OBJ_test_2,
        save_dir_2,
        args.n_runs,
        args.sample_num,
        case_num=2
    )
    # case 3 parameters
    compute_parameters_case(
        parameters_case_template,
        A_OBJ_test_3,
        save_dir_3,
        args.n_runs,
        args.sample_num,
        case_num=3
    )
