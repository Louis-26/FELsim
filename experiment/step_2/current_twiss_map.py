import os, sys

CURRENT_DIR = os.getcwd()
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../backend"))
sys.path.insert(0, BACKEND_DIR)

import torch
import numpy as np
from excelElements import ExcelElements
from ebeam import beam as ebeam_class

# Beamline index for each dimension of R^26 (fixed order)
QUAD_INDICES = [1, 3, 10, 16, 18, 20, 27, 33, 35, 37, 39, 41, 43,
                50, 56, 58, 61, 63, 70, 76, 78, 80, 87, 93, 95, 97]

_ebeam = ebeam_class()


# 1. Build the beamline and load the 26 currents

def currents_to_twiss(bl, currents, particles, evaluation_pos= 138, noise=False, sigma=None):
    """
    R^26 -> R^9

    Parameters
    ----------
    bl: beamline object
    currents : list/array, len 26
        Currents of the 26 quads/dipoles, ordered to match QUAD_INDICES
    particles : tensor (N, 6)
        Initial particle distribution


    Returns
    -------
    list, len 9
        [[alpha_x, beta_x, eps_x]
        [alpha_y, beta_y, eps_y]
        [alpha_z, beta_z, eps_z]]
    """
    if not (0 <= evaluation_pos <= len(bl)):
        raise ValueError(
            f"evaluation_pos must be in [0, {len(bl)}], got {evaluation_pos}"
        )

    # consider noise on currents

    currents_t = torch.as_tensor(currents, dtype=torch.float64)

    if noise:
        if sigma is None:
            noise_sigma = 1e-3
        else:
            noise_sigma = torch.as_tensor(sigma, dtype=torch.float64)

        noisy_currents = currents_t + torch.randn_like(currents_t) * noise_sigma

        current_map = {idx: float(noisy_currents[k]) for k, idx in enumerate(QUAD_INDICES)}
    else:
        current_map = {idx: float(currents_t[k]) for k, idx in enumerate(QUAD_INDICES)}

    # 2. Track particles through the full beamline
    p = particles.clone()
    for i, elem in enumerate(bl):
        if i >= evaluation_pos:
            break
        if i in current_map:
            p = elem.useMatrice(p, current=current_map[i])
        else:
            p = elem.useMatrice(p)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float64)

    # 3. Compute all x/y/z Twiss parameters in one pass
    dist_avg, dist_cov, twiss = _ebeam.cal_twiss(p, ddof=1)

    # 4. Extract the 9 values
    A = r"$\alpha$"
    B = r"$\beta$ (m)"
    E = r"$\epsilon$ ($\pi$.mm.mrad)"
    output_twiss = [
        float(twiss.loc["x", A]), float(twiss.loc["y", A]), float(twiss.loc["z", A]),
        float(twiss.loc["x", B]), float(twiss.loc["y", B]), float(twiss.loc["z", B]),
        float(twiss.loc["x", E]), float(twiss.loc["y", E]), float(twiss.loc["z", E]),
    ]
    # column 1 as (alpha_x, alpha_y, alpha_z), column 2 as (beta_x, beta_y, beta_z), column 3 as (eps_x, eps_y, eps_z)
    # each column represents one twiss parameter in all axis, each row represents all twiss parameters in one axis
    return torch.tensor(output_twiss, dtype=torch.float64).reshape(3, 3).T
