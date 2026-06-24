# Date
2026/06/15 - 2026/06/21 

# Tasks
- ✅swap objective function from MSE to logMSE in scenario A
- ✅get N=1000 samples for function $R^{26} \rightarrow R^9$, mapping from current set of current to twiss parameters
- ✅implement noise model with gaussian setting 

# Finished Work
- ⭐the Koa cluster configuration and connection has been fully set on a compute node, in partition of `sandbox`, 19 cores and 120 GB memory, the future connection will stick with this setting
- after transforming from MSE to logMSE from the [file](../experiment/step_2/scenario_A_logMSE.ipynb), the results comparison is:
  - 🟢although the overall MSE is smaller, but it is mainly due to the truncation error
  - 🔴the training time from 5 minutes ➡️ 24 minutes for 3 random starts for all five methods
  - 🔴the error loss curve is highly unstable, especially for trust-region method
   
In conclusion, it is not worthwhile to consider switching to log of MSE, because of the substantial larger time cost, increasing iteration and unstable loss curve.
- parallelization has been enabled with [demo](../experiment/step_2/current_twiss_simulation_parallel.py), tested with 6 cores, shortening time from 102 min ➡️ 17 minutes(roughly just 1/6 of the original time) for 1000 samples
- noise model with gaussian noise has been implemented [here](../experiment/step_2/current_twiss_map.py), with prespecified standard deviation

Additionally,
- finish three html files to visualize electron beamline, updated [here](../background_knowledge_summary/)
  - [`accelerator_6d_phase_space_3d_viewer.html`](../background_knowledge_summary/accelerator_6d_phase_space_3d_viewer.html): visualize meaning of 6D phase parameters
  - [`accelerator_beamline_elements_3d_viewer.html`](../background_knowledge_summary/accelerator_beamline_elements_3d_viewer.html): visualize function of differerent lattices
  - [`diagnostic_chicane_138_element_simulator.html`](../background_knowledge_summary/diagnostic_chicane_138_element_simulator.html): visualize the entire DC with 138 elements

# Potential Next Steps
TBD

# Questions/Requests
Just one observation, the Koa cluster can't support the request of too many cores, 24-core request will take three days to get assigned, which is not realistic enough. 