# original case
## optimization algorithm
- Nelder-Mead
- L-BFGS-B
- L-BFGS-B, 2 point
- SLSQP
- COBYLA

## decision variable
- $I$: current at the lattice **1** as quadrupole focusing lattice
- $I_2$: current at the lattice **3** as quadrupole defocusing lattice

## current bound
- $I$: (0.01, 10) A
- $I_2$: (0.01, 10) A

## global optimal solution ground truth 
- $I$=0.8218 A
- $I_2$=1.0430 A

## start point
uniformly select **5** random points in current bound for start point


## performance
method_tag  conv_rate nit_mean wall_time_mean final_mse_mean I_1_ref    I_1    I_3_ref    I_3     alpha_x   alpha_y 
      COBYLA     0%      32.4       5.419 s       5.30e+05    0.8218 A 0.8690 A 1.0430 A 1.1419 A -2.45e-02  1.04e-03
    L-BFGS-B     0%       0.0       0.477 s       5.51e+19    0.8218 A 4.5721 A 1.0430 A 1.3853 A -5.42e+02 -3.04e+01
L-BFGS-B+jac     0%       0.0       0.481 s       5.51e+19    0.8218 A 4.5721 A 1.0430 A 1.3853 A -5.42e+02 -3.04e+01
 Nelder-Mead    20%      39.8      17.943 s       3.04e+05    0.8218 A 0.8896 A 1.0430 A 1.1536 A  3.37e-05  1.08e-05
   SLSQP+jac     0%       1.0       0.548 s       5.51e+19    0.8218 A 4.5721 A 1.0430 A 1.3853 A -5.42e+02 -3.04e+01


## conclusion
Unfortunately, each method has failed to converge to the global optimizer with the current current bound.


# adjusted case - change variable bound
now that we know the mse is smaller when the current value is closer to 0, we contract the current bound

## adjusted current bound
- $I$: (0.01, 1.5) A
- $I_2$: (0.01, 1.5) A


## start point tuning
uniformly random generated start points for **5** times in the current bounds for both **I_1** and **I_3**



## performance
 method_tag  conv_rate nit_mean wall_time_mean final_mse_mean I_1_ref    I_1    I_3_ref    I_3     alpha_x   alpha_y 
      COBYLA     40%    313.2      69.001 s       1.25e-04    0.8218 A 0.2903 A 1.0430 A 0.2781 A -7.33e-03 -4.20e-03
    L-BFGS-B      0%      0.2       2.818 s       3.21e+01    0.8218 A 0.6904 A 1.0430 A 0.6424 A  3.11e-01  1.19e+00
L-BFGS-B+jac      0%      0.2       2.688 s       3.21e+01    0.8218 A 0.6904 A 1.0430 A 0.6424 A  3.11e-01  1.19e+00
 Nelder-Mead    100%     41.4      13.873 s       1.58e-09    0.8218 A 0.8896 A 1.0430 A 1.1536 A  2.33e-05  4.29e-06
   SLSQP+jac      0%      1.4       1.474 s       4.04e+01    0.8218 A 0.5484 A 1.0430 A 0.8218 A -1.35e+00  1.27e+00

# incorporate gradient descent and newton's method

## Gradient Descent
method_tag    conv_rate nit_mean wall_time_mean final_mse_mean I_1_ref    I_1    I_3_ref    I_3    alpha_x   alpha_y 
Gradient-Descent    40%      38.8     101.230 s       5.69e+01    0.8218 A 0.7471 A 1.0430 A 0.5802 A 2.31e-06 -1.65e-06

## Newton's method
method_tag conv_rate nit_mean wall_time_mean final_mse_mean I_1_ref    I_1    I_3_ref    I_3     alpha_x  alpha_y 
  Newton      100%     25.4      62.032 s       1.46e-13    0.8218 A 0.8620 A 1.0430 A 1.1375 A -1.82e-07 1.27e-08

Conclusion:
Gradient Descent is not good enough as the convergence rate is just 40%. Newton's method is much better, howeve