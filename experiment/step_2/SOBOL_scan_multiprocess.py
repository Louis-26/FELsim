"""
Use multiple cores to accelerate the SOBOL scan process. 
Specifically, we randomly sample current values and compute the weighted MSE, Jacobian, 
Jacobian_rank, Hessian, Hessian_GN, distance between Hessian and Hessian_GN, and eigenvalues/eigenvectors of the Hessian for each sample point.
"""

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..")))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import argparse
import time
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial  

from SOBOL_scan import *
# =================================================================
# Main execution logic
# =================================================================

def identity_func(x):
    return x

A_VARS = {1: ["I_1", "current", identity_func], 3: ["I_3", "current", identity_func]}

A_OBJ = {
    8: [{"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0}],
    9: [{"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOBOL scan with multiprocessing.")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of sample points to process.")
    parser.add_argument("--save", type=bool, default=False, help="Whether to save the results.")
    args = parser.parse_args()
    sample_num = args.sample_num
    
    var_names = [val[0] for val in A_VARS.values()] 
    inputs = [
        {name: np.random.uniform(0.01, 1.5) for name in var_names} 
        for _ in range(sample_num)
    ]
    
    slurm_cores = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cores is not None:
        n_cores = int(slurm_cores)
    else:
        # For local execution, reserve one core to prevent system freeze
        n_cores = max(1, multiprocessing.cpu_count() - 1)

    print(f"🚀 Starting parallel computation, processing {sample_num} sample points using {n_cores} core(s)...")
    start_time = time.time()
    
    target_func = partial(evaluate_single_sample, A_VARS=A_VARS, A_OBJ=A_OBJ)
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(target_func, inputs))
    
    output_list = results
    print(output_list)
    
    t1 = time.time()
    print("=" * 50)
    print(f"✅ Parallel scan completed! Total wall time: {time.time()-start_time:.4f} seconds")
    print(f"Number of valid data points: {len(output_list)}")
    print("=" * 50)

    if args.save:
        with open("../../results/scan_parameters.pkl", "wb") as f:
            pickle.dump(
                {
                    "output_dict": output_list,
                },
            f,
        )
        print("📁 Results saved to ../../results/scan_parameters.pkl")