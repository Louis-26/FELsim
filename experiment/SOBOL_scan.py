import numpy as np
import torch
import copy
import time
import pickle
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../backend")))
# print(sys.path)

# Initialize the beamline
from configs import *
from experiments_utils import run_scenario_A, display_results, run_scenario_A_parallel
from ebeam import beam as ebeam_class
from beamline import lattice
from excelElements import ExcelElements
from beamOptimizer import beamOptimizer
from copy import deepcopy
import time


def time_computation(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"Time elasted: {end_time - start_time:.4f} seconds")
        dt = end_time - start_time
        if isinstance(result, tuple):
            return (*result, dt)
        else:
            return result, dt
    return wrapper


tunable_current=list(FELSIM_S1_CURRENTS.keys())

@time_computation
def compute_weighted_MSE(I_dict, A_VARS, A_OBJ):
    """
    I_list: list of currents at different positions
    weight: weight for the objectives
    eval_pos: positions to evaluate the objectives
    targets: list of target values for the objectives
    A_OBJ: the object to be optimized
    """
    bl = ExcelElements(EXCEL_PATH).create_beamline()
    p = PARTICLES.clone()
    obj_copy = copy.deepcopy(A_OBJ)
    opti = beamOptimizer(bl, p, noise=False)
    weight=[i[0]["weight"] for i in list(A_OBJ.values())]
    eval_pos=list(A_OBJ.keys())
    SP = {var_name: {"bounds": CURRENT_BOUNDS, "start": 0.5} for var_name in I_dict.keys()}
    opti._prepare(A_VARS, SP, obj_copy)

    I_list = [I_dict[var_name] for var_name in opti.variablesToOptimize]
    phi=opti._optiSpeed(I_list)
    num_goals = 2
    r_list = []
    # print(opti.objectives)
    for i in range(len(opti.objectives)):
        stat = float(opti.objectives[eval_pos[i]][0]["measured"])
        goal=float(opti.objectives[eval_pos[i]][0]["goal"])
        r = np.sqrt(weight[i] / num_goals) * (stat -goal) / weight[i]
        r_list.append(r)
    
    r = np.array(r_list)
    return r, phi


def perturb_currents(I_dict, delta=1e-5):
    """
    Perturb the currents in I_dict by delta, suppose variable number is k
    Returns k number of dictionary set, where each set is a tuple with two dictionaries, one with the current perturbed by +delta and the other by -delta.
    """
    I_dict_plus_dict = {}
    I_dict_minus_dict = {}
    for i in range(len(I_dict)):
        I_dict_plus = deepcopy(I_dict)
        I_dict_minus = deepcopy(I_dict)
        I_dict_plus[list(I_dict.keys())[i]] += delta
        I_dict_minus[list(I_dict.keys())[i]] -= delta
        I_dict_plus_dict[i] = I_dict_plus
        I_dict_minus_dict[i] = I_dict_minus
    return I_dict_plus_dict, I_dict_minus_dict

@time_computation
def compute_jacobian(I_dict, A_VARS, A_OBJ, delta=1e-5):
    """
    Compute the Jacobian matrix J(k) for scenario A, with dimension 2x2 
    I_1: current at position 1 for qrodrupole focusing Lattice 1
    I_3: current at position 3 for qrodrupole defocusing Lattice 1
    r: the R vector computed from compute_weighted_MSE
    delta: small perturbation for numerical differentiation
    """
    var_num = len(A_VARS)
    obj_num = len(A_OBJ)
    J = np.zeros((obj_num, var_num))
    I_dict_plus_dict, I_dict_minus_dict = perturb_currents(I_dict, delta)
    # print("plus", I_dict_plus_dict)
    # print("minus", I_dict_minus_dict)
    for i in range(var_num):
        I_dict_plus = I_dict_plus_dict[i]
        I_dict_minus = I_dict_minus_dict[i]
        r_plus_delta = compute_weighted_MSE(I_dict_plus, A_VARS, A_OBJ)[0]
        r_minus_delta = compute_weighted_MSE(I_dict_minus, A_VARS, A_OBJ)[0]
        J[:, i] = (r_plus_delta - r_minus_delta) / (2 * delta)

    return J

@time_computation
def compute_hessian_phi(I_dict, A_VARS, A_OBJ, delta=1e-5):
    var_num = len(I_dict)
    def compute_grad_phi(I_dict, A_VARS, A_OBJ, delta=1e-5):
        r=compute_weighted_MSE(I_dict, A_VARS, A_OBJ)[0]
        return 2*compute_jacobian(I_dict, A_VARS, A_OBJ, delta)[0].T @ r
    # print(f"Gradient of phi: {compute_grad_phi(I_dict_test, A_VARS_test, A_OBJ_test)}")
    H = np.zeros((var_num, var_num))
    I_dict_plus_dict, I_dict_minus_dict = perturb_currents(I_dict, delta)
    for i in range(var_num):
        I_dict_plus = I_dict_plus_dict[i]
        I_dict_minus = I_dict_minus_dict[i]
        grad_plus_delta = compute_grad_phi(I_dict_plus, A_VARS, A_OBJ, delta)
        grad_minus_delta = compute_grad_phi(I_dict_minus, A_VARS, A_OBJ, delta)
        H[:, i] = (grad_plus_delta - grad_minus_delta) / (2 * delta)
    return H

@time_computation
def compute_eig(H):
    """
    Compute the eigenvalues and eigenvectors of the Hessian matrix H
    H: Hessian matrix
    """
    eigvals, eigvecs = np.linalg.eig(H)
    return eigvals, eigvecs

# without multiprocessing, 2 variables, 2 objectives
def total_scan_execution(A_VARS_test, A_OBJ_test, sample_num):
    print(f"Scan current variables from position: {list(A_VARS_test.keys())}")

    print(f'Scan objectives from position: {["_".join(reversed(i[0]["measure"]))+"="+str(i[0]["goal"])+" @ position "+str(list(A_OBJ_test.keys())[j])+ " with weight "+str(i[0]["weight"]) for j, i in enumerate(list(A_OBJ_test.values()))]}')
    t0=time.time()    
    output_list=list()
    time_log=dict()
    for i in range(sample_num):
        I_dict_test = {}
        for j in range(len(A_VARS_test)):
            if j==0:
                I_dict_test["I"] = np.random.uniform(0.01, 1.5) 
            else:
                I_dict_test["I"+str(j+1)] = np.random.uniform(0.01, 1.5)
        # I = np.random.uniform(0.01, 1.5)
        # I2 = np.random.uniform(0.01, 1.5)
        # I_dict_test = {"I": I, "I2": I2}
        # compute mse phi
        r, phi, time_mse = compute_weighted_MSE(I_dict_test, A_VARS_test, A_OBJ_test)
        # compute jacobian 
        J, time_jac = compute_jacobian(I_dict_test, A_VARS_test, A_OBJ_test)
        # compute hessian
        H, time_hess = compute_hessian_phi(I_dict_test, A_VARS_test, A_OBJ_test)
        # compute eigenvalues and eigenvectors
        eigvals, eigvecs, time_eig = compute_eig(H)
        output_list.append({
            "current": tuple(I_dict_test.values()),
            "parameters": (phi, J, H, eigvals, eigvecs)
        })
        time_log["time_mse"] = time_log.get("time_mse", 0) + time_mse
        time_log["time_jac"] = time_log.get("time_jac", 0) + time_jac
        time_log["time_hess"] = time_log.get("time_hess", 0) + time_hess
        time_log["time_eig"] = time_log.get("time_eig", 0) + time_eig

    for i in time_log.keys():
        time_log[i]/=sample_num 

    t1=time.time()

    print(f"Total time taken without multiprocessing: {t1-t0:.4f} seconds")
    print(f"Time log: {time_log}")
    return output_list, time_log, t1-t0


def get_random_samples(local_min, bounds, num_samples, thres_1, thres_2, mode):

    keys = list(local_min.keys())
    ref_point = np.array([local_min[k] for k in keys])
    low, high = bounds
    dim = len(keys)
    
    samples = []
    
    while len(samples) < num_samples:
        v = np.random.normal(0, 1, dim)
        direction = v / np.linalg.norm(v) # randomly sample a direction in n dim
        if mode == 'close':
            
            u = np.random.uniform(0, 1)
            r = thres_1 * (u ** (1.0 / dim))
            
            candidate = ref_point + r * direction
            
            if np.all(candidate >= low) and np.all(candidate <= high):
                samples.append(dict(zip(keys, candidate)))
                
        elif mode == 'far':

            with np.errstate(divide='ignore', invalid='ignore'):
                t_high = (high - ref_point) / direction
                t_low = (low - ref_point) / direction
                t_max = np.where(direction > 0, t_high, t_low)
                d_max = np.min(t_max)
            
            if d_max <= thres_2:
                continue
                
            u = np.random.uniform(0, 1)
            r = (thres_2**dim + u * (d_max**dim - thres_2**dim))**(1.0 / dim)
            
            candidate = ref_point + r * direction
            samples.append(dict(zip(keys, candidate)))
                
    return samples


def SOBOL_sim(current, sample_number, A_VARS, A_OBJ, threshold_1=0.1, threshold_2=0.5):
    """
    simulate a number of samples close to/far away from the local minimum
    current: the current value dictionary from optimization algorithm
    sample_number: the number of samples to simulate, both close to and far away from the local minimum
    threshold_1: the threshold for determining if a sample is close to the local minimum
    threshold_2: the threshold for determining if a sample is far away from the local minimum
    """
    samples_close = get_random_samples(local_min=current, bounds=CURRENT_BOUNDS, num_samples=sample_number, thres_1=threshold_1, thres_2=threshold_2, mode='close')
    samples_far = get_random_samples(local_min=current, bounds=CURRENT_BOUNDS, num_samples=sample_number, thres_1=threshold_1, thres_2=threshold_2, mode='far')
    result_close=[]
    
    result_far=[]
    # close
    for sample in samples_close:
        # compute mse phi
        r, phi, time_mse = compute_weighted_MSE(sample, A_VARS, A_OBJ)
        # compute jacobian 
        J, time_jac = compute_jacobian(sample, A_VARS, A_OBJ)
        # compute hessian
        H, time_hess = compute_hessian_phi(sample, A_VARS, A_OBJ)
        # compute eigenvalues and eigenvectors
        eigvals, eigvecs, time_eig = compute_eig(H)
        result_close.append({
            "current": tuple(sample.values()),
            "parameters": (phi, J, H, eigvals, eigvecs)
        })


    # far
    for sample in samples_far:
        # compute mse phi
        r, phi, time_mse = compute_weighted_MSE(sample, A_VARS, A_OBJ)
        # compute jacobian 
        J, time_jac = compute_jacobian(sample, A_VARS, A_OBJ)
        # compute hessian
        H, time_hess = compute_hessian_phi(sample, A_VARS, A_OBJ)
        # compute eigenvalues and eigenvectors
        eigvals, eigvecs, time_eig = compute_eig(H)
        result_far.append({
            "current": tuple(sample.values()),
            "parameters": (phi, J, H, eigvals, eigvecs)
        })


    return result_close, result_far

def run_baseline_sim(CURRENT_BOUNDS,EPSILON,N_RUNS_A,FELSIM_S1_CURRENTS,A_OBJ,
        A_VARS,A_BEAMLINE_LEN,METHODS_A,scale,METHOD_OPTIONS,use_log,use_epsilon,noise,
        sigma,plot_curve,verbose, sample_num, threshold_1, threshold_2, save_dir):
    results_A, df_A, final_summary = run_scenario_A(
        CURRENT_BOUNDS,
        EPSILON,
        N_RUNS_A,
        FELSIM_S1_CURRENTS,
        A_OBJ,
        A_VARS,
        A_BEAMLINE_LEN,
        METHODS_A,
        scale=scale,
        METHOD_OPTIONS=METHOD_OPTIONS,
        use_log=use_log,
        use_epsilon=use_epsilon,
        noise=noise,
        sigma=sigma,
        plot_curve=plot_curve,
        verbose=verbose
    )
    results_1 = display_results(results_A, A_VARS, FELSIM_S1_CURRENTS)
    res_close, res_far = SOBOL_sim(current=results_1, sample_number=sample_num, A_VARS=A_VARS, A_OBJ=A_OBJ, threshold_1=threshold_1, threshold_2=threshold_2)
    with open(save_dir, "wb") as f:
        pickle.dump((results_1, res_close, res_far), f)
    print("close results: ", res_close)
    print("far results: ", res_far)
    
    
    
# enable multiprocessing 
from functools import partial
import concurrent.futures

def evaluate_single_sample(sample, A_VARS, A_OBJ):
    """
    compute the weighted MSE, Jacobian, Hessian, and eigenvalues/eigenvectors for a single sample of currents
    """
    try:
        r, phi, time_mse = compute_weighted_MSE(sample, A_VARS, A_OBJ)
        J, time_jac = compute_jacobian(sample, A_VARS, A_OBJ)
        H, time_hess = compute_hessian_phi(sample, A_VARS, A_OBJ)
        eigvals, eigvecs, time_eig = compute_eig(H)
        
        return {
            "current": tuple(sample.values()),
            "parameters": (phi, J, H, eigvals, eigvecs)
        }
    except Exception as e:
        return None
    
# --- parallelized SOBOL_sim ---
def SOBOL_sim_parallel(current, sample_number, A_VARS, A_OBJ, threshold_1=0.01, threshold_2=0.5, max_cores=6):
    
    samples_close = get_random_samples(local_min=current, bounds=CURRENT_BOUNDS, num_samples=sample_number, thres_1=threshold_1, thres_2=threshold_2, mode='close')
    samples_far = get_random_samples(local_min=current, bounds=CURRENT_BOUNDS, num_samples=sample_number, thres_1=threshold_1, thres_2=threshold_2, mode='far')
    
    worker_func = partial(evaluate_single_sample, A_VARS=A_VARS, A_OBJ=A_OBJ)
    
    print(f"⚡ Utilize {max_cores} cores, and scanning {sample_number * 2} points...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        raw_close = list(executor.map(worker_func, samples_close))
        raw_far = list(executor.map(worker_func, samples_far))
        
    result_close = [res for res in raw_close if res is not None]
    result_far = [res for res in raw_far if res is not None]

    return result_close, result_far

def run_baseline_sim_parallel(CURRENT_BOUNDS,EPSILON,N_RUNS_A,FELSIM_S1_CURRENTS,A_OBJ,
        A_VARS,A_BEAMLINE_LEN,METHODS_A,scale,METHOD_OPTIONS,use_log,use_epsilon,noise,
        sigma,plot_curve,verbose, sample_num, threshold_1, threshold_2, save_dir,
        max_cores_r=6, max_cores_s=6):

    results_A, df_A, final_summary = run_scenario_A_parallel(
        CURRENT_BOUNDS, EPSILON, N_RUNS_A, FELSIM_S1_CURRENTS, A_OBJ, A_VARS,
        A_BEAMLINE_LEN, METHODS_A, scale=scale, METHOD_OPTIONS=METHOD_OPTIONS,
        use_log=use_log, use_epsilon=use_epsilon, noise=noise, sigma=sigma,
        plot_curve=plot_curve, verbose=verbose, max_cores=max_cores_r
    )
    
    results_1 = display_results(results_A, A_VARS, FELSIM_S1_CURRENTS)
    
    res_close, res_far = SOBOL_sim_parallel(
        current=results_1, 
        sample_number=sample_num, 
        A_VARS=A_VARS, 
        A_OBJ=A_OBJ, 
        threshold_1=threshold_1, 
        threshold_2=threshold_2,
        max_cores=max_cores_s  
    )
    print(f"Close results: {res_close}")
    print(f"Far results: {res_far}")
    with open(save_dir, "wb") as f:
        pickle.dump((results_1, res_close, res_far), f)
