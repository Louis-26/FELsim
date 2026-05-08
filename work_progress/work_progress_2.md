# tasks

1. Bayesian optimization using NN/Gaussian processes
2. look into https://botorch.org/
3. pytorch implementation for gradient descent, compared with benchmark(f)
4. use bayesian optimization

# next step
1. compare with/without pytorch, for `backend/test/beamline_optimization.ipynb`
do similar things to `optimizer_benchmarking.ipynb`,
2. consider using Jacobian/Hessian for optimization
3. adjust parameter number, objective and constraints, see how it affects the optimization process
4. evaluate optimizer in the ways
- number of iterations
- residual 
- time
- computational/memory cost
5. do research on GPU