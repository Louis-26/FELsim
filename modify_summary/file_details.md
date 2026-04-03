# Backend File Details (Integrated Summary)

This document integrates the 18-file backend review into one place.  
Scope includes production Python modules in `backend/` and experiment/test scripts under `backend/test/`.  

## API Layer

## 1) `backend/felAPI.py`
- **Purpose:** FastAPI service entrypoint exposing simulation workflows to the frontend.
- **Implemented functions:**
  - `getPngObjFromBeamList(beamlist, plotParams)`
  - `root()`
  - `excelToBeamline(excelJson)`
  - `loadAxes(plotParams)`
  - `getBeamSegmentInfo()`
  - `plot_parameters(graphParams)`
  - `getParticlesFromTwiss(twissParams)` *(stub)*
- **Realized capability:** request validation + beamline simulation execution + image/twiss serialization + parameter sweep endpoints.

## 2) `backend/ApiSchemas.py`
- **Purpose:** Pydantic request/response models for API contracts.
- **Implemented classes:**
  - `BeamSegmentsInfo`, `AxisTwiss`, `TwissParameters`, `BeamlineInfo`, `PlottingParameters`,
  - `LineAxObject`, `AxesPNGData`, `GraphParameters`, `GraphPlotPointResponse`, `GraphPlotData`, `ExcelBeamlineElement`
- **Realized capability:** strict payload validation, typed API I/O, consistent frontend/backend data exchange.

## ⭐Beam Physics Core

## 3) `backend/beamline.py`
- **Purpose:** Core beamline transfer-matrix physics and element definitions.
- **Implemented classes/method groups:**
  - Base `lattice`: particle properties, relativistic updates, matrix application
  - Elements: `driftLattice`, `qpfLattice`, `qpdLattice`, `dipole`, `dipole_wedge`
  - Utility `Beamline`: segment position bookkeeping, interpolation/fitting helpers, line reconfiguration
- **Realized capability:** beam transport through configurable lattice elements for electrons/protons/isotopes.

## 4) `backend/ebeam.py`
- **Purpose:** Beam distribution generation, Twiss/statistical analysis, and phase-space plotting helpers.
- **Implemented method groups:**
  - Distribution: `gen_6d_gaussian`, `gen_6d_from_twiss`
  - Twiss/statistics: `cal_twiss`, `alpha/beta/gamma/epsilon/phi/disper/envelope/std`
  - Geometry/plot: `ellipse_sym`, `getXYZ`, `plotXYZ`, `heatmap`
  - WIP analysis: ellipse inclusion counting utilities
- **Realized capability:** derive beam optics parameters and plotting data from particle clouds.

## 5) `backend/physicalConstants.py`
- **Purpose:** Centralized CODATA constants and particle helper utilities.
- **Implemented structures/functions:**
  - `ParticleProperties` typed dict
  - `PhysicalConstants` class methods (particle lookup, relativistic helpers, isotope parsing)
  - module-level `get_electron()`, `get_proton()` convenience wrappers
- **Realized capability:** standardized constants + reusable particle/relativistic utilities.

## ⭐Optimization

## 6) `backend/beamOptimizer.py`
- **Purpose:** Numerical optimization over beamline parameters using SciPy.
- **Implemented methods:**
  - core: `__init__`, `_optiSpeed`, `calc`
  - diagnostics: `testSpeed`, `testFuncEval`, `testFuncIt`
- **Realized capability:** tune segment parameters to minimize weighted objective mismatch at chosen indices.

## 7) `backend/AlgebraicOptimization.py`
- **Purpose:** Symbolic optimization and sigma-matrix objective construction (SymPy).
- **Implemented methods:**
  - sigma/matrix builders: `getDistSigmai`, `getTwissSigmai`, `getM`, `getSigmaF`
  - objective solver flow: `findSymmetricObjective`, `getRootsUni`, `getRootsMulti`
- **Realized capability:** symbolic final-beam objective equations and root-finding for lattice tuning.

## Visualization & Analysis Utilities

## 8) `backend/schematic.py`
- **Purpose:** Beamline propagation orchestration and UI/rendering logic.
- **Implemented method groups:**
  - Propagation: `simulateData`, `plotBeamPositionTransform`
  - Plot generation: `createLinePlot`, `currentcreateUI`
  - Utility helpers: scaling, csv writing, closest-z lookup, eps export, drift scatter transform
- **Realized capability:** stepped transport along z with Twiss/envelope aggregation and API-ready plots.

## 9) `backend/radiation.py`
- **Purpose:** Inverse Compton scattering analysis utilities.
- **Implemented methods:**
  - `plot_ICS_angularDist`
  - `plotScatteringPhoton`
  - `photonEnergySpectrum`
- **Realized capability:** visualize angular and energy distributions of scattered photons for chosen beam/laser settings.

## 10) `backend/beamUtility.py`
- **Purpose:** Auxiliary beam/material interaction and power-deposition analysis.
- **Implemented methods:**
  - `chargePerMacropulse`, `getPowerDF`
  - penetration/stopping models: `model_Grunn`, `model_Bethe`
  - deposition/plot helpers: `compute_deposition_profile`, `plot_deposition_profile`, `plot_penetration_depth`, `plot_stopping_power`
- **Realized capability:** quick studies of deposited power, heating trends, and penetration depth across materials.

## Data Ingestion

## 11) `backend/excelElements.py`
- **Purpose:** Import lattice definitions from Excel/JSON and build beamline objects.
- **Implemented methods:**
  - loading (`load_dictionary_lattice`, `load_excel_lattice`)
  - construction (`create_beamline`)
  - queries (`get_dataframe`, `find_element_by_position`, `__str__`)
- **Realized capability:** convert spreadsheet-style lattice tables into executable beamline segments.

## Entry / Execution

## 12) `backend/main.py`
- **Purpose:** Legacy exploratory script tying together simulation/plotting/optimization modules.
- **Implemented functions/classes:** none (procedural script).
- **Realized capability:** ad hoc local experimentation (not production API flow).

## Tests / Experimental Scripts (`backend/test/`)

## 13) `backend/test/Test_UHBeamline.py`
- **Purpose:** Broad UH beamline scenario script.
- **Implemented functions/classes:** none (procedural).
- **Realized capability:** generate beam, load lattice, run symbolic + numeric optimization experiments.

## 14) `backend/test/UHM_beamline_opt.py`
- **Purpose:** Sequential optimization campaign on UH beamline sections.
- **Implemented functions/classes:** none (procedural).
- **Realized capability:** run staged matching/tuning blocks over many indices and objectives.

## 15) `backend/test/goldTwiss.py`
- **Purpose:** reference/golden Twiss and optimization trial script.
- **Implemented functions/classes:** none (procedural).
- **Realized capability:** benchmark-like optics setup and tuned-current experimentation.

## 16) `backend/test/testDipole.py`
- **Purpose:** focused dipole + wedge behavior test.
- **Implemented functions/classes:** none (procedural).
- **Realized capability:** propagate particle cloud through custom dipole/wedge lines and inspect behavior.

## 17) `backend/test/testOptimization.py`
- **Purpose:** compact optimizer smoke test.
- **Implemented functions/classes:** none (procedural).
- **Realized capability:** minimal end-to-end objective-based tuning example with plotting.

## 18) `backend/test/unused.py`
- **Purpose:** archive of deprecated snippets.
- **Implemented functions:** legacy standalone `getSymbolicMatrice(self)` and `useMatrice(self, values, matrice)` (not active architecture).
- **Realized capability:** historical reference only.

---

## Quick architecture map
1. **API layer:** `felAPI.py` + `ApiSchemas.py`
2. **Beam physics core:** `beamline.py`, `ebeam.py`, `physicalConstants.py`
3. **Optimization:** `beamOptimizer.py`, `AlgebraicOptimization.py`
4. **Visualization & analysis utilities:** `schematic.py`, `radiation.py`, `beamUtility.py`
5. **Data ingestion:** `excelElements.py`
6. **Entry / execution:** `main.py`
7. **Experiments/tests:** `backend/test/*.py`
