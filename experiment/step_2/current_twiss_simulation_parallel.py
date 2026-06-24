# basic parameter settings
# ===========================================================================
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import torch
import numpy as np

# Import your existing modules
from current_twiss_map import currents_to_twiss, QUAD_INDICES
from beamline import lattice
from excelElements import ExcelElements
from ebeam import beam as ebeam_class

CURRENT_DIR = os.getcwd()
EXCEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../beam_excel/Beamline_elements_3.xlsx"))

# ── Experiment Hyperparameters ─────────────────────────────────────────────
N_CORES = 6        # 🚀 Number of CPU cores to utilize
N_TRIALS = 1000     # Total number of simulation trials
BEAM_ENERGY = 40.0  
N_PARTICLES = 1000  
SEED = 42           
CURRENT_BOUNDS = (0.01, 1.5)  

# ── Relativistic factors & Targets ──────────────────────
relat = lattice(1, fringeType=None)
relat.setE(E=BEAM_ENERGY)  
GAMMA_REL = relat.gamma  
BETA_REL = relat.beta  
NORM = GAMMA_REL * BETA_REL  

K = 1.2  
LAMBDA_U = 2.3e-2  
BETA_YM = GAMMA_REL / (K * 2 * np.pi / LAMBDA_U)  
BETA_XM = 1.4  
ALPHA_XM = 0.47  
ALPHA_YM = 0.0  

# ── Beam parameters ─────────────────────────────────────
epsilon_n = 8.0  
x_std = 0.8  
y_std = 0.8  
f_RF = 2856e6  
bunch_spread_ps = 2.0  
energy_spread_pct = 0.5  
h = 5e9  

epsilon = epsilon_n / NORM  
x_prime_std = epsilon / x_std  
y_prime_std = epsilon / y_std  
tof_std = bunch_spread_ps * 1e-9 * f_RF  
energy_std = energy_spread_pct * 10  


# ===========================================================================
# 🚀 Core Worker Function - Will run independently on each CPU core
# ===========================================================================
def simulate_trial(trial_idx, bl_obj, particles_tensor):
    """
    Task executed by a single process: 
    Generate random currents -> Track particles -> Calculate Twiss
    """
    # ⚠️ CRITICAL: Limit PyTorch threads per process to 1. 
    # This prevents 24 cores from thrashing and freezing the operating system.
    torch.set_num_threads(1)
    
    # ⚠️ Ensure each process has an independent random seed based on its trial ID
    # to prevent identical currents across different cores.
    np.random.seed(SEED + trial_idx) 
    
    # 1. Generate random currents
    currents = np.random.uniform(low=CURRENT_BOUNDS[0], high=CURRENT_BOUNDS[1], size=26)
    
    # 2. Run Twiss mapping engine
    twiss_result = currents_to_twiss(bl_obj, currents, particles_tensor, evaluation_pos=138, noise=False, sigma=None)
    sigma = torch.ones(len(currents)) * 1e-3
    twiss_result_noisy = currents_to_twiss(bl_obj, currents, particles_tensor, evaluation_pos=138, noise=True, sigma=sigma)
    
    # Return Trial ID for resorting purposes later
    return trial_idx, currents, twiss_result, twiss_result_noisy


# ===========================================================================
# 🚀 Start Parallel Scheduler (Main Execution)
# ===========================================================================
# ⚠️ In Jupyter or Windows environments, multiprocessing code must be guarded 
# by the if __name__ == '__main__': block.
if __name__ == '__main__':
    
    print("⏳ Initializing beamline physics model and initial electron beam distribution...")
    bl = ExcelElements(EXCEL_PATH).create_beamline()
    ebeam_obj = ebeam_class()
    PARTICLES = ebeam_obj.gen_6d_gaussian(
        0,
        [x_std, x_prime_std, y_std, y_prime_std, tof_std, energy_std],
        N_PARTICLES,
    )
    print("✅ Initialization complete!")
    
    results = []
    
    print(f"🚀 Starting parallel simulation cluster (Cores: {N_CORES}, Total Trials: {N_TRIALS})...")
    
    # Start process pool
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        # Distribute all tasks
        futures = [executor.submit(simulate_trial, i, bl, PARTICLES) for i in range(N_TRIALS)]
        
        # Attach tqdm progress bar to display real-time collection progress
        for future in tqdm(as_completed(futures), total=N_TRIALS, desc="Parallel Simulation", unit="trial"):
            try:
                trial_idx, currents, twiss_res, twiss_res_noisy = future.result()
                results.append({
                    "trial_id": trial_idx,
                    "currents": currents,
                    "twiss": twiss_res,
                    "twiss_noisy": twiss_res_noisy
                })
            except Exception as e:
                tqdm.write(f"❌ Task error in trial: {e}")
                
    # Since as_completed yields as soon as a task finishes, the results order is scrambled.
    # We sort it by trial_id to restore the strict 1 to 1000 chronological order.
    results.sort(key=lambda x: x["trial_id"])
    
    print("🎉 Congratulations! All 1000 simulation tasks have been rapidly completed via multiprocessing!")
    
    # If you want to check the 1st result, uncomment below:
    # print(results[0]["currents"])
    # print(results[0]["twiss"])