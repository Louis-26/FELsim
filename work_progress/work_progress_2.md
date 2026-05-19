# tasks overview
1. ✅compare with/without pytorch, for `backend/test/beamline_optimization.ipynb`
do similar things to `optimizer_benchmarking.ipynb`,
2. consider using Jacobian/Hessian for optimization
3. adjust parameter number, objective and constraints, see how it affects the optimization process
4. ✅evaluate optimizer in the ways
- number of iterations
- residual 
- time
- computational/memory cost


# finished work
1. finish `backend/test/beamline_optimization.ipynb` test execution together with pytorch integration,
   - download the files from the [latest branch](https://github.com/komochristian/FELsim/tree/nginx) of [FELsim](https://github.com/komochristian/FELsim)
     - `evolutionPlotter.py`
     - `felsimAdapter.py`
     - `beamEvolution.py` 
     - `loggingConfig.py`
     - `simulatorBase.py`
     - `beamPropagator.py`
     - `latticeLoader.py`
     - `cosyAdapter.py`
     - `cosySimulator.py`
     - `cosyParticleSimulator.py`
     - `beamlineBuilder.py`
     - `cosyResultsReader.py`
   - `optimizer_benchmarking.ipynb` completely verified and executed in [pytorch version](../backend/test/optimizer_benchmarking.ipynb)

2. summarize optimization results comparison [here](../optimizer_compare/runtime_comparison_beamline_optimization.md)

## requests
1. get COSY Infinity installed on desktop, specifically `cosy.exe`

## questions
1. for coordinates, why the independent variable is `s` instead of `t`?
2. How does the current directly affect the twiss parameters?

## potential next steps
- figure out the direction to modify and improve the current optimizers(consider second order optimizers)
- determine some test cases as the benchmark to evaluate goodness of optimizers
- evaluate whether GPU can help shorten execution time
- figure out how to adjust test cases