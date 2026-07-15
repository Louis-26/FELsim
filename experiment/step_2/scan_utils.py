import numpy as np
import numpy as np
import torch
import copy
import time

import os, sys
current_dir=os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../")))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../../backend")))
# print(sys.path)

# Initialize the beamline
from configs import *
from experiment.experiments_utils import load_results
from ebeam import beam as ebeam_class
from beamline import lattice
from excelElements import ExcelElements
from beamOptimizer import beamOptimizer

def compute_weighted_MSE(I_1, I_3, weight=[1, 1], target_alpha_x=0, target_alpha_y=0):
    """
    I_1: current at position 1 for qrodrupole focusing Lattice 1
    I_3: current at position 3 for qrodrupole defocusing Lattice 1
    weight: weight for the objectives
    target_alpha_x: target value of alpha_x at position 8
    target_alpha_y: target value of alpha_y at position 9
    """
    bl = ExcelElements(EXCEL_PATH).create_beamline()[:beamline_slice_len]
    p = PARTICLES.clone()
    obj_copy = copy.deepcopy(A_OBJ)
    opti = beamOptimizer(bl, p, noise=False)
    k_vals = [I_1, I_3]
    SP = {
        "I": {"bounds": CURRENT_BOUNDS, "start": 0.5},
        "I2": {"bounds": CURRENT_BOUNDS, "start": 0.5},
    }
    opti._prepare(A_VARS, SP, obj_copy)
    input_dict = {"I": I_1, "I2": I_3}
    k_vals = [input_dict[var_name] for var_name in opti.variablesToOptimize]
    phi=opti._optiSpeed(k_vals)
    num_goals = 2
    r_list = []
    
    stat_x = float(opti.objectives[8][0]["measured"])
    r_x = np.sqrt(weight[0] / num_goals) * (stat_x - target_alpha_x) / weight[0]
    r_list.append(r_x)
    
    stat_y = float(opti.objectives[9][0]["measured"])
    r_y = np.sqrt(weight[1] / num_goals) * (stat_y - target_alpha_y) / weight[1]
    r_list.append(r_y)
    
    r = np.array(r_list)
    return r, phi

def compute_jacobian(I_1, I_3, r, delta=1e-5):
    """
    Compute the Jacobian matrix J(k) for scenario A, with dimension 2x2 
    I_1: current at position 1 for qrodrupole focusing Lattice 1
    I_3: current at position 3 for qrodrupole defocusing Lattice 1
    r: the R vector computed from compute_weighted_MSE
    delta: small perturbation for numerical differentiation
    """
    J = np.zeros((2, 2))
    
    # Perturb I_1
    r_plus_delta, _ = compute_weighted_MSE(I_1 + delta, I_3)
    r_minus_delta, _ = compute_weighted_MSE(I_1 - delta, I_3)
    
    J[:, 0] = (r_plus_delta - r_minus_delta) / (2 * delta)
    
    # Perturb I_3
    r_plus_delta, _ = compute_weighted_MSE(I_1, I_3 + delta)
    r_minus_delta, _ = compute_weighted_MSE(I_1, I_3 - delta)
    
    J[:, 1] = (r_plus_delta - r_minus_delta) / (2 * delta)
    
    return J

def compute_grad_phi(I_1, I_3, r, delta=1e-5):
    return 2*compute_jacobian(I_1, I_3, r, delta).T @ r

def compute_hessian_phi(I_1, I_3, r, delta=1e-5):
    grad_phi=compute_grad_phi(I_1, I_3, r, delta)
    H = np.zeros((2, 2))
    for i in range(2):
        # Perturb I_1 and I_3
        if i == 0:
            grad_plus_delta = compute_grad_phi(I_1 + delta, I_3, r, delta)
            grad_minus_delta = compute_grad_phi(I_1 - delta, I_3, r, delta)
        else:
            grad_plus_delta = compute_grad_phi(I_1, I_3 + delta, r, delta)
            grad_minus_delta = compute_grad_phi(I_1, I_3 - delta, r, delta)
        
        H[:, i] = (grad_plus_delta - grad_minus_delta) / (2 * delta)
    return H


def compute_eig(H):
    """
    Compute the eigenvalues and eigenvectors of the Hessian matrix H
    H: Hessian matrix
    """
    eigvals, eigvecs = np.linalg.eig(H)
    return eigvals, eigvecs

def worker_task(current_pair):
    I_1, I_3 = current_pair
    try:
        r, phi = compute_weighted_MSE(I_1, I_3, weight=[1, 1], target_alpha_x=0, target_alpha_y=0)
        J = compute_jacobian(I_1, I_3, r)
        H = compute_hessian_phi(I_1, I_3, r)
        eigvals, eigvecs = compute_eig(H)
        
        return {
            "current": (I_1, I_3),
            "parameters": (phi, J, H, eigvals, eigvecs),
            "success": True
        }
    except Exception as e:
        return {"current": (I_1, I_3), "success": False, "error": str(e)}