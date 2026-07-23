import os, sys
import numpy as np


# set the system path
current_dir=os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../backend")))

from beamline import lattice
from ebeam import beam as ebeam_class
from beamline import lattice
from excelElements import ExcelElements
from beamOptimizer import beamOptimizer
from evolutionPlotter import EvolutionPlotter
# beamline configuration

EXCEL_PATH = os.path.abspath(os.path.join(current_dir, "../../beam_excel/Beamline_elements_3.xlsx"))
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

beamline_slice_len=10

# ── Load beamline & map quad indices ───────────────────────────────
beamline_full = ExcelElements(EXCEL_PATH).create_beamline()

# beamline parameters
CURRENT_BOUNDS = (0.01, 1.5)
EPSILON = 1

FELSIM_S1_CURRENTS = {
     1: 0.8218,  3: 1.0430,
    10: 3.8834,
    16: 2.2396, 18: 4.9532, 20: 3.4258,
    27: 4.6657,
    33: 2.6942, 35: 2.6523, 37: 0.2768, 39: 0.2768, 41: 2.6523, 43: 2.6942,
    50: 4.6739,
    56: 3.1219, 58: 3.3129,
    61: 5.1775, 63: 4.0434,
    70: 4.6818,
    76: 3.9336, 78: 4.0787, 80: 0.0139,
    87: 1.3624, 93: 0.9452, 95: 2.8851, 97: 2.1921,
}

# parameter setting
N_RUNS_A = 100
A_BEAMLINE_LEN = 10

METHODS_A = [
    ("Nelder-Mead", None),
    ("L-BFGS-B", "2-point"),
    ("SLSQP", "2-point"),
    ("COBYLA", None),
    ("trust-constr", "2-point"),
]

A_VARS = {
    1: ["I", "current", lambda x: x],
    3: ["I2", "current", lambda x: x],
}
A_OBJ = {
    8: [{"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0}],
    9: [{"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}],
}

noise = False
sigma = None

# optimizer parameters
METHOD_OPTIONS = {
    "L-BFGS-B": {
        "maxcor": 10,
        "ftol": 2.220446049250313e-09,
        "gtol": 1e-5,
        "eps": 1e-8,
        "maxfun": 15000,
        "maxiter": 15000,
        "maxls": 20,
    },
    "SLSQP": {
        "maxiter": 100,
        "ftol": 1e-6,
        "iprint": 1,
        "disp": False,
        "eps": 1.4901161193847656e-08,
        "finite_diff_rel_step": None,
        "workers": None,
    },
    "trust-constr": {
        "xtol": 1e-8,
        "gtol": 1e-8,
        "barrier_tol": 1e-8,
        "sparse_jacobian": None,
        "maxiter": 1000,
        "verbose": 0,
        "finite_diff_rel_step": None,
        "initial_constr_penalty": 1.0,
        "initial_tr_radius": 1.0,
        "initial_barrier_parameter": 0.1,
        "initial_barrier_tolerance": 0.1,
        "factorization_method": None,
        "disp": False,
        "workers": None,
    },
    "Nelder-Mead": {
        "maxiter": None,
        "maxfev": None,
        "disp": False,
        "return_all": False,
        "initial_simplex": None,
        "xatol": 1e-4,
        "fatol": 1e-4,
        "adaptive": False,
    },
    "COBYLA": {
        "rhobeg": 1.0,
        "tol": 1e-4,
        "maxiter": 1000,
        "disp": 0,
        "catol": None,
        "f_target": float("-inf"),
    },
}
use_log = False
log_epsilon = 1e-12