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
CURRENT_BOUNDS_A = (0.01, 1.5)
CURRENT_BOUNDS_BC = (0.01, 5.0)

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
## scenario A
N_RUNS_A = 1
A_BEAMLINE_LEN = 10

METHODS_A = [
    ("Nelder-Mead", None),
    ("L-BFGS-B", "2-point"),
    ("SLSQP", "2-point"),
    ("COBYLA", None),
    ("trust-constr", "2-point"),
]

def identity_func(x):
    """Identity transform for a variable spec.

    A module-level def, NOT a lambda: the variable specs are shipped to worker processes for
    the multiprocessing runs, and lambdas/closures cannot be pickled.
    """
    return x


A_VARS = {
    1: ["I", "current", identity_func],
    3: ["I2", "current", identity_func],
}
A_OBJ = {
    8: [{"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0}],
    9: [{"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}],
}

# ---- Bayesian-optimization budget ----
N_INIT_A     = 5                      # random initial evaluations
N_BO_STEPS_A = 30                     # BO iterations (1 FELsim evaluation per step)

## scenario B
METHODS_B = METHODS_A

# scenario B setting (identical to scenario_A-C.ipynb)
N_RUNS_B     = 1          # independent repetitions of the whole 11-stage pipeline
N_INIT_B     = 3          # random initial evaluations per stage
N_BO_STEPS_B = 25         # BO steps per stage (1 FELsim evaluation each)


def _v(name):
    """Variable spec understood by beamOptimizer: [name, attribute, x -> attribute value].

    Uses the module-level `identity_func` rather than a lambda so that STAGES_B / C_VARS stay
    picklable for the multiprocessing runs.
    """
    return [name, "current", identity_func]

def get_stages_b(CURRENT_BOUNDS):
    STAGES_B = [
        (
            "Stage 1 Doublet",
            {1: _v("I"), 3: _v("I2")},
            {
                8: [
                    {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.0},
                ],
                9: [
                    {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
                ],
            },
            10,
            {
                "I": {"bounds": CURRENT_BOUNDS, "start": 1},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 1},
            },
        ),
        (
            "Stage 2 Chrom.1",
            {10: _v("I")},
            {15: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
            16,
            {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
        ),
        (
            "Stage 3 Triplet1",
            {16: _v("I"), 18: _v("I2"), 20: _v("I3")},
            {
                25: [
                    {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.5},
                ],
                26: [
                    {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
                ],
            },
            27,
            {
                "I": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 5},
                "I3": {"bounds": CURRENT_BOUNDS, "start": 3},
            },
        ),
        (
            "Stage 4 Chrom.2",
            {27: _v("I")},
            {32: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
            33,
            {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
        ),
        (
            "Stage 5 DblTriplet",
            {37: _v("I"), 35: _v("I2"), 33: _v("I3")},
            {
                37: [
                    {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "envelope"], "goal": 2.0, "weight": 1},
                    {"measure": ["y", "envelope"], "goal": 2.0, "weight": 1},
                ]
            },
            38,
            {
                "I": {"bounds": CURRENT_BOUNDS, "start": 0.28},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 2.65},
                "I3": {"bounds": CURRENT_BOUNDS, "start": 2.69},
            },
        ),
        (
            "Stage 6 Chrom.3",
            {50: _v("I")},
            {55: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
            56,
            {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
        ),
        (
            "Stage 7 IP",
            {56: _v("I"), 58: _v("I2")},
            {
                59: [
                    {"measure": ["x", "envelope"], "goal": 0.0, "weight": 1},
                    {"measure": ["y", "envelope"], "goal": 0.0, "weight": 1},
                ]
            },
            60,
            {
                "I": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
            },
        ),
        (
            "Stage 8 Doublet2",
            {61: _v("I"), 63: _v("I2")},
            {
                68: [
                    {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.5},
                ],
                69: [
                    {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
                ],
            },
            70,
            {
                "I": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
            },
        ),
        (
            "Stage 9 Chrom.4",
            {70: _v("I")},
            {75: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]},
            76,
            {"I": {"bounds": CURRENT_BOUNDS, "start": 1}},
        ),
        (
            "Stage 10 Triplet3",
            {76: _v("I"), 78: _v("I2"), 80: _v("I3")},
            {
                85: [
                    {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "beta"], "goal": 0.1, "weight": 0.5},
                ],
                86: [
                    {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5},
                ],
            },
            87,
            {
                "I": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I3": {"bounds": CURRENT_BOUNDS, "start": 2},
            },
        ),
        (
            "Stage 11 UND Match",
            {87: _v("Ic"), 93: _v("I"), 95: _v("I2"), 97: _v("I3")},
            {
                92: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 0.5}],
                117: [
                    {"measure": ["x", "alpha"], "goal": ALPHA_XM, "weight": 1},
                    {"measure": ["y", "alpha"], "goal": ALPHA_YM, "weight": 1},
                    {"measure": ["x", "beta"], "goal": BETA_XM, "weight": 1},
                    {"measure": ["y", "beta"], "goal": BETA_YM, "weight": 1},
                ],
            },
            118,
            {
                "Ic": {"bounds": CURRENT_BOUNDS, "start": 4},
                "I": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I2": {"bounds": CURRENT_BOUNDS, "start": 2},
                "I3": {"bounds": CURRENT_BOUNDS, "start": 2},
            },
        ),
    ]
    return STAGES_B
STAGES_B = get_stages_b(CURRENT_BOUNDS_BC)

N_RUNS_B     = 1          # independent repetitions of the whole 11-stage pipeline
N_INIT_B     = 3          # random initial evaluations per stage
N_BO_STEPS_B = 25         # BO steps per stage (1 FELsim evaluation each)

## scenario C
METHODS_C = METHODS_A
C_BEAMLINE_LEN = 118
N_RUNS_C     = 1
N_INIT_C     = 22        # random initial evaluations (= number of variables)
N_BO_STEPS_C = 200        # BO steps (1 FELsim evaluation each)

C_VARS = {
    56: _v("I_56"), 58: _v("I_58"),
    61: _v("I_61"), 63: _v("I_63"),
    76: _v("I_76"), 78: _v("I_78"), 80: _v("I_80"),
    87: _v("I_87"), 93: _v("I_93"), 95: _v("I_95"), 97: _v("I_97")
}

# Objectives: IP spot + UND beta matching
C_OBJ = {
    59: [
        {"measure": ["x", "envelope"], "goal": 0.0, "weight": 1.0},
        {"measure": ["y", "envelope"], "goal": 0.0, "weight": 1.0},
    ],
    117: [
        {"measure": ["x", "alpha"], "goal": ALPHA_XM, "weight": 1.0},
        {"measure": ["y", "alpha"], "goal": ALPHA_YM, "weight": 1.0},
        {"measure": ["x", "beta"], "goal": BETA_XM, "weight": 1.0},
        {"measure": ["y", "beta"], "goal": BETA_YM, "weight": 1.0},
    ]
}

# XOpt
XOPT_TAG = "Xopt-EI"       # 'method_tag' used in the result tables
bl_full = ExcelElements(EXCEL_PATH).create_beamline()



# model setting
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