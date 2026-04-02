# Backend File Categorization

This document categorizes files in `backend/` by their functional role.

## 1) API Layer
- `backend/felAPI.py`  
  FastAPI app, routes, request handling, and response shaping.
- `backend/ApiSchemas.py`  
  Pydantic schemas for request/response validation and typed API contracts.

## 2) Beam Physics Core
- `backend/beamline.py`  
  Beamline lattice elements, transport matrices, and segment-level propagation logic.
- `backend/ebeam.py`  
  Beam distribution generation, Twiss/statistical calculations, and related beam math.
- `backend/physicalConstants.py`  
  Physical constants and particle-property helpers used by physics computations.

## 3) Optimization
- `backend/beamOptimizer.py`  
  Numerical optimization workflow for beamline parameter tuning.
- `backend/AlgebraicOptimization.py`  
  Symbolic/algebraic optimization utilities (SymPy-based equations/root solving).

## 4) Visualization & Analysis Utilities
- `backend/schematic.py`  
  Beamline plotting and diagnostic visualization utilities.
- `backend/radiation.py`  
  Radiation-related calculations and plotting helpers.
- `backend/beamUtility.py`  
  Supplemental beam/energy/power utility functions and analysis helpers.

## 5) Data Ingestion
- `backend/excelElements.py`  
  Parsing and converting Excel beamline definitions into internal segment objects.

## 6) Entry/Execution
- `backend/main.py`  
  Script-style entry/experimentation module for backend workflows.
- `backend/runServer.sh`  
  Convenience script for launching backend services.

## 7) Environment & Containerization
- `backend/requirements.txt`  
  Python dependency list for backend runtime.
- `backend/Dockerfile`  
  Backend container image definition.
- `backend/compose.yaml`  
  Backend-oriented compose configuration.

## 8) Tests / Experimental Scripts
- `backend/test/Test_UHBeamline.py`
- `backend/test/UHM_beamline_opt.py`
- `backend/test/beamline_optimization.ipynb`
- `backend/test/goldTwiss.py`
- `backend/test/testDipole.py`
- `backend/test/testOptimization.py`
- `backend/test/unused.py`

These files are used for validation, experimental workflows, and manual testing scenarios.
