# basic parameter settings
# ===========================================================================
from current_twiss_map import *
import numpy as np
from beamline import lattice
from tqdm.auto import tqdm

EXCEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../beam_excel/Beamline_elements_3.xlsx"))
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

bl = ExcelElements(EXCEL_PATH).create_beamline()
ebeam_obj = ebeam_class()
PARTICLES = ebeam_obj.gen_6d_gaussian(
    0,
    [x_std, x_prime_std, y_std, y_prime_std, tof_std, energy_std],
    N_PARTICLES,
)

for i in tqdm(range(1000), desc="Simulating Twiss Parameters", unit="trial"):
    currents = np.random.uniform(low=0.01, high=1.5, size=26)
    twiss_result = currents_to_twiss(bl, currents, PARTICLES, evaluation_pos=138, noise=False, sigma=None)
    tqdm.write(f"Trial {i + 1}/1000")
    tqdm.write(f"currents {i + 1}: {currents}")
    tqdm.write(f"twiss parameters {i + 1}:\n{twiss_result}")
    tqdm.write("=" * 50)