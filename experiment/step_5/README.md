Bayesian Optimization on scenario A/B/C with/without XOpt

# Methods

## scipy optimizer
The following are time cost and average MSE for one run because of huge time cost, for **five benchmark optimizers**.
- scenario A
  - average/total time cost: `15 seconds`/`75.3 seconds`
  - best of average MSE: 6.951e-15
- scenario B(check worst MSE of all stages)
  - time cost: `104 minutes 8 seconds`/`519 minutes 41 seconds`
  - average worst MSE(SLSQP stands out as worst MSE is below 1): 6.962e-01
- scenario C
  - time cost: `51 minutes 29 seconds`/`257 minutes 29 seconds`
  - best of average MSE: 7.222e+13

Execution
```bash
# use sequential execution
python scenario_A-C_seq.py \
    --N_RUNS_A 1000 \
    --N_RUNS_B 1000 \
    --N_RUNS_C 1000 \
    --use_multi 0

# use parallel execution
python scenario_A-C_seq.py \
    --N_RUNS_A 1000 \
    --N_RUNS_B 1000 \
    --N_RUNS_C 1000 \
    --use_multi 1
```

## XOpt
Number of Bayesian optimization steps can be increased to further reduce MSE.

- scenario A(BO steps: 30)
  - time cost: 38.9 seconds
  - average MSE: 2.591e-02

- scenario B(BO steps: 25)
  - time cost: 722.79 seconds
  - average MSE: 2.008228e+03	
  
- scenario C(BO steps: 200)
  - time cost: 1834.91 seconds
  - average MSE: 1.387e+05

Execution
```bash
# use sequential execution
python scenario_A-C_Xopt_seq.py \
    --N_RUNS_A 1000 \
    --N_RUNS_B 1000 \
    --N_RUNS_C 1000 \
    --use_multi 0

# use parallel execution
python scenario_A-C_Xopt_seq.py \
    --N_RUNS_A 1000 \
    --N_RUNS_B 1000 \
    --N_RUNS_C 1000 \
    --use_multi 1
```


## Table Summary

| Scenario | Variables | scipy: time per optimizer | XOpt: time | scipy: MSE | XOpt: MSE | Faster | Lower MSE |
|---|---|---|---|---|---|---|---|
| A | 2 | `15 s` (total `75.3 s`) | `38.9 s` | 6.951e-15 | 2.591e-02 | scipy, 2.6x | **scipy**, ~13 orders |
| B | 23 over 11 stages (1–4 at a time) | `104 min 08 s` ≈ `6248 s` (total `519 min 41 s`) | `722.79 s` | 6.962e-01 | 2.008e+03 | **XOpt, 8.6x** | scipy, ~3.5 orders |
| C | 11 simultaneously | `51 min 29 s` ≈ `3089 s` (total `257 min 29 s`) | `1834.91 s` | 7.222e+13 | 1.387e+05 | **XOpt, 1.7x** | **XOpt, ~8.7 orders** |



# Conclusion
When the number of variables gets larger(scenario B and C), Xopt saves significantly more time. In terms of the optimizer performance, Xopt is not as precise as 
Unfortunately, both Xopt and scipy optimizers completely fail in scenario C.
