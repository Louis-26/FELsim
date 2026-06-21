# FELsim Project Overview

## Core idea
FELsim is a full-stack beamline simulation toolkit for Free Electron Laser (FEL) and accelerator studies. The backend models particle transport through beamline elements using 6D transfer matrices and Twiss formalism, while the frontend provides an interactive UI to construct beamlines, run simulations, and visualize phase-space/Twiss evolution.

## Implemented capabilities

### Backend (Python)
- **Beamline physics model (`backend/beamline.py`)**
  - Base `lattice` class with particle/energy handling.
  - Segment classes for drifts, quadrupoles, dipoles, wedge dipoles, and transfer-matrix propagation.
- **Beam distribution + beam optics analytics (`backend/ebeam.py`)**
  - Generate 6D particle distributions (Gaussian and Twiss-based).
  - Compute Twiss parameters, envelope/dispersive metrics, and plotting helpers.
- **Simulation and plotting orchestration (`backend/schematic.py`)**
  - Propagate particles along beamline segments.
  - Produce per-location phase-space views and line-plot data over longitudinal distance.
- **FastAPI service (`backend/felAPI.py`)**
  - `/beamsegmentinfo` for available segment schema/defaults.
  - `/excel-to-beamline` to translate Excel-like rows into segment objects.
  - `/axes` to run simulation and return base64 image payloads + Twiss data.
  - `/plot-parameters` for parameter scan plots.
- **Excel translator (`backend/excelElements.py`)**
  - Load beamline definitions from Excel or JSON-like dictionaries.
  - Build typed beamline segment lists from tabular element descriptions.
- **Optimization utilities (`backend/beamOptimizer.py`, `backend/AlgebraicOptimization.py`)**
  - Numerical optimization over beamline parameters against Twiss/statistical objectives.
- **Radiation utilities (`backend/radiation.py`)**
  - Inverse-Compton-scattering angular and energy-spectrum analysis plots.

### Frontend (React/Vite)
- **Interactive beamline editor + simulation UI (`fel-app/src/App.jsx`)**
  - Compose/edit beam segments with defaults and per-segment parameters.
  - Configure particle count, interval, beam type/energy, and Twiss/base distribution.
  - Call backend APIs, display image/line plots, and inspect simulation outputs.
- **Reusable visualization and control components (`fel-app/src/components/*`)**
  - Graphs, dropdowns, editable table cells, simulation/beam/particle settings, and error modals.

## Architectural pattern
- **Input path**: Beamline from manual UI edits or Excel-derived JSON.
- **Compute path**: Backend converts to segment objects, propagates 6D particles, computes Twiss.
- **Output path**: Backend returns encoded plot assets + data; frontend renders and allows parameter iteration.
