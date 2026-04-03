# Backend Code Modification Scope

This document marks which backend files should change for the PyTorch-oriented update, based on the category rule recorded in `modify_summary/file_category.md`.


The details of changed script can be visible in `../modify_test_cases.md` and `../modify_summary/file_details.md`

# files category
❌: certainly no need to change to pytorch
✅: need to change to pytorch and finished
⏳: need to change to pytorch but not finished yet
❓: not sure whether change to pytorch can help, not changed yet

## 1) API Layer
### ❌`backend/felAPI.py`
no need to change

### ❌`backend/ApiSchemas.py`
no need to change


## 2) Beam Physics Core
### ✅`backend/beamline.py`
changed functions
- `backend` > `beamline.py`
- `lattice` > `useMatrice`
- `driftLattice` > `getSymbolicMatrice`
- `Beamline` > `defineEndFrontPos`
- `Beamline` > `interpolateData`
- `Beamline` > `_testModeOrder2end`
- `Beamline` > `_testModeOrder2front`
- `Beamline` > `_endModel`
- `Beamline` > `_frontModel`
- `Beamline` > `reconfigureLine`


### ✅`backend/ebeam.py`
changed functions
- `ellipse_sym`
- `cal_twiss`
- `gen_6d_gaussian`
- `is_within_ellipse`
- `particles_in_ellipse`
- `envelope`
- `twiss_to_cov`
- `rotate_cov`
- `gen_6d_from_twiss`


### ❌`backend/physicalConstants.py`
no need to change


## 3) Optimization
### ❓`backend/beamOptimizer.py`
no need to change


### ❓`backend/AlgebraicOptimization.py`
no need to change



## 4) Visualization & Analysis Utilities
### ✅`backend/schematic.py`
- `driftTransformScatter`
- `checkMinMax`


### ✅`backend/radiation.py`
- `plot_ICS_angularDist` 
- `plotScatteringPhoton` 
- `photonEnergySpectrum`

### ✅`backend/beamUtility.py`
- `chargePerMacropulse`
- `getPowerDF`
- `model_Bethe`
- `compute_deposition_profile`


## 5) Data Ingestion

### ❌`backend/excelElements.py`
no need to change



## 6) Entry / Execution
### ❌`backend/main.py`
no need to change 


## ❌7) Tests / Experimental Scripts
no need to change
- `backend/test/Test_UHBeamline.py`
- `backend/test/UHM_beamline_opt.py`
- `backend/test/beamline_optimization.ipynb`
- `backend/test/goldTwiss.py`
- `backend/test/testDipole.py`
- `backend/test/testOptimization.py`
- `backend/test/unused.py`

## Short Interpretation

- Change first: API, core physics, and optimization code.
- Leave stable: visualization, ingestion, execution scripts, environment files, and tests.
- If later validation shows downstream breakage, update non-target files only as a narrow compatibility fix.
