optimization results comparison with and without pytorch

## `optimizer_benchmarking.ipynb` 
mathematical objective
$$
\mathcal{L} = \frac{1}{N} \sum_i w_i \, (m_i - g_i)^2
$$

### Scenario A: Doublet (2 quadrupoles)
method: 
- Nelder-Mead
- L-BFGS-B
- L-BFGS-B(2-point)
- SLSQP
- COBYLA

objective: 
- $\alpha_x=0$ at position **8**
- $\alpha_y=0$ at position **9**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice |   Variable    | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:-------------:|------------:|----------:|:--------:|
| 1      |       I       |      XXX    |    XXX | (0.01, 10)  |
| 3      | I<sub>2</sub> |     XXX    |    XXX | (0.01, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 8      | α<sub>x</sub> |  0.0 |    1.0 |  XXX  |  XXX  |
| 9      | α<sub>y</sub> |  0.0 |    1.0 |  XXX  |  XXX  |



#### Evaluation Analysis
best method: L-BFGS-B+jac           2.1179e-13     0.2827     0.2687

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | XXX            | XXX            |
| Execution time  | 2 min 35 s      | 7 min 28 s      |
| Residual        | failed(reason investigating)  | XXX  |


### Scenario B: 11-Stage Sequential Optimization
method: Nelder-Mead

objective: 
listed in notebook

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice |   Variable    | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:-------------:|------------:|----------:|:--------:|
| 87      |       I<sub>c</sub>       |      XXX    |    XXX | (0.01, 10)  |
| 93      | I |     XXX    |    XXX | (0.01, 10)  |
| 95      | I<sub>2</sub> |     XXX    |    XXX | (0.01, 10)  |
| 97      | I<sub>3</sub> |     XXX    |    XXX | (0.01, 10)  |


**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 8      | α<sub>x</sub> |  0.0 |    1.0 |  XXX  |  XXX  |
| 9      | α<sub>y</sub> |  0.0 |    1.0 |  XXX  |  XXX  |



#### Evaluation Analysis

#### Multi-Stage Optimization — PyTorch vs NumPy

| Metric         | PyTorch       | NumPy         |
|----------------|--------------:|--------------:|
| Iterations     | XXX           | XXX           |
| Execution time | 142 min 35 s  | 207 min 38 s  |

**Per-stage residual & time**

| Stage                | PyTorch MSE | NumPy MSE  | PyTorch t (s) | NumPy t (s) |
|:---------------------|------------:|-----------:|--------------:|------------:|
| Stage 1 Doublet      |     0.00016 |    0.00013 |         13.56 |       14.65 |
| Stage 2 Chrom.1      |     0.00000 |    0.00000 |         18.59 |       19.86 |
| Stage 3 Triplet1     |     0.00195 |    0.00166 |         86.04 |      502.53 |
| Stage 4 Chrom.2      |     0.00000 |    0.00000 |         45.41 |       47.92 |
| Stage 5 DblTriplet   |     0.07654 |    0.15171 |        244.35 |      422.21 |
| Stage 6 Chrom.3      |     0.00000 |    0.00000 |         66.20 |       76.64 |
| Stage 7 IP           |     0.00141 |    0.00369 |        144.33 |      142.64 |
| Stage 8 Doublet2     |     0.00118 |    0.00107 |        184.87 |      313.45 |
| Stage 9 Chrom.4      |     0.00000 |    0.00000 |         93.13 |       94.36 |
| Stage 10 Triplet3    |     0.00010 |    0.00146 |        566.62 |      472.49 |
| Stage 11 UND Match   |     0.22094 |    0.00002 |       1384.78 |     2046.07 |
| **Total**            |             |            |   **2847.86** | **4152.85** |



### Scenario C: Combined ~11-parameter, Dual Objective (IP + UND)
method: 
- Nelder-Mead
- L-BFGS-B
- SLSQP

objective: 
listed in notebook

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice |   Variable    | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:-------------:|------------:|----------:|:--------:|
| 56      |       I<sub>c</sub>       |      XXX    |    XXX | (0.01, 10)  |
| 58      | I<sub>1</sub> |     XXX    |    XXX | (0.01, 10)  |
| 61      | I<sub>2</sub> |     XXX    |    XXX | (0.01, 10)  |
| 63      | I<sub>3</sub> |     XXX    |    XXX | (0.01, 10)  |
| 76      | I<sub>4</sub> |     XXX    |    XXX | (0.01, 10)  |
| 78      | I<sub>5</sub> |     XXX    |    XXX | (0.01, 10)  |
| 80      | I<sub>6</sub> |     XXX    |    XXX | (0.01, 10)  |
| 87      | I<sub>7</sub> |     XXX    |    XXX | (0.01, 10)  |
| 93      | I<sub>8</sub> |     XXX    |    XXX | (0.01, 10)  |
| 95      | I<sub>9</sub> |     XXX    |    XXX | (0.01, 10)  |
| 97      | I<sub>10</sub> |     XXX    |    XXX | (0.01, 10)  |


**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 8      | α<sub>x</sub> |  0.0 |    1.0 |  XXX  |  XXX  |
| 9      | α<sub>y</sub> |  0.0 |    1.0 |  XXX  |  XXX  |



#### Evaluation Analysis


| Metric         | PyTorch       | NumPy         |
|----------------|--------------:|--------------:|
| Iterations     | XXX           | XXX           |
| Execution time | 152 min 35 s  | seem not to converge as it takes super long  |

MSE: 
pytorch 

                 final_mse                nfev       wall_time          
                      mean          min   mean  min       mean       min
method_tag                                                              
L-BFGS-B+jac  2.023616e+32  863562.8125   69.6   12   249.7644    42.512
Nelder-Mead   2.757128e+05  216968.8750  416.4  356  1541.5370  1354.323
SLSQP+jac     2.023616e+32  863562.8125   12.0   12    41.6082    37.745


## `beamline_optimization.ipynb` 
mathematical objective
$$
\mathcal{L} = \frac{1}{N} \sum_i w_i \, (m_i - g_i)^2
$$

### First Quadrupole Doublet
method: Nelder-Mead

objective: 
- $\alpha_x=0, \beta_x=0.1$ at position **8**
- $\alpha_y=0, \beta_y=0.1$ at position **9**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice |   Variable    | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:-------------:|------------:|----------:|:--------:|
| 1      |       I       |      0.8416 |    0.8408 | (0, 10)  |
| 3      | I<sub>2</sub> |      1.0544 |    1.0520 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 8      | α<sub>x</sub> |  0.0 |    1.0 |  −6.475 × 10⁻⁴  |  −5.659 × 10⁻⁴  |
| 8      | β<sub>x</sub> |  0.1 |    0.0 |   0.9442 m      |   0.9342 m      |
| 9      | α<sub>y</sub> |  0.0 |    1.0 |  −6.072 × 10⁻⁴  |  −5.401 × 10⁻⁴  |
| 9      | β<sub>y</sub> |  0.1 |    0.5 |   0.1342 m      |   0.1342 m      |



#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 60            | 58            |
| Execution time  | 280.95 s      | 230.56 s      |
| Residual        | 1.461 × 10⁻⁴  | 1.463 × 10⁻⁴  |


### First Chromacity Quad
method: Nelder-Mead

objective:
- $D_x = 0$ at position **15**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 10     |    I     |      4.1843 |    3.9067 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 15     | D<sub>x</sub> |  0.0 |    1.0 |  −3.230 × 10⁻⁷  |  −2.841 × 10⁻⁷  |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 40            | 40            |
| Execution time  | 176.53 s      | 156.92 s      |
| Residual        | 1.043 × 10⁻¹³ | 8.072 × 10⁻¹⁴ |


### Quadrupole Triplet
method: Nelder-Mead

objective:
- $\alpha_x = 0, \beta_x = 0.1$ at position **25**
- $\alpha_y = 0, \beta_y = 0.1$ at position **26**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 16     |    I     |      2.1148 |    2.6496 | (0, 10)  |
| 18     |   I<sub>2</sub> | 4.9061 | 5.0412 | (0, 10)  |
| 20     |   I<sub>3</sub> | 3.4873 | 3.1322 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 25     | α<sub>x</sub> |  0.0 |    1.0 |  −7.002 × 10⁻⁵  |   3.770 × 10⁻⁴  |
| 25     | β<sub>x</sub> |  0.1 |    0.5 |   0.0118 m      |   0.0111 m      |
| 26     | α<sub>y</sub> |  0.0 |    1.0 |   3.179 × 10⁻⁴  |  −2.348 × 10⁻³  |
| 26     | β<sub>y</sub> |  0.1 |    0.5 |   0.0109 m      |   0.0210 m      |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 109           | 600           |
| Execution time  | 480.85 s      | 2045.76 s     |
| Residual        | 1.963 × 10⁻³  | 1.768 × 10⁻³  |


### Second Chromacity Quad
method: Nelder-Mead

objective:
- $D_x = 0$ at position **32**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 27     |    I     |      4.7647 |    4.6855 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 32     | D<sub>x</sub> |  0.0 |    1.0 |   5.054 × 10⁻⁷  |   1.143 × 10⁻⁶  |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 40            | 40            |
| Execution time  | 169.16 s      | 170.32 s      |
| Residual        | 2.554 × 10⁻¹³ | 2.813 × 10⁻¹⁴ |


### Double Quadrupole Triplet
method: Nelder-Mead

objective:
- $\alpha_x = 0, \alpha_y = 0$ at position **37**
- envelope$_x = 2.0$, envelope$_y = 2.0$ at position **37**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 37     |    I     |      0.0000 |    0.2832 | (0, 10)  |
| 35     |   I<sub>2</sub> | 2.3280 | 2.6536 | (0, 10)  |
| 33     |   I<sub>3</sub> | 2.6621 | 2.6901 | (0, 10)  |

**Objectives**

| Indice | Quantity            | Goal | Weight | PyTorch   | NumPy     |
|:------:|:-------------------:|-----:|-------:|----------:|----------:|
| 37     | α<sub>x</sub>       |  0.0 |    1.0 |  −0.0055  |  −0.0079  |
| 37     | α<sub>y</sub>       |  0.0 |    1.0 |  −0.0059  |  −0.0170  |
| 37     | envelope<sub>x</sub>|  2.0 |    1.0 |   1.6259  |   1.4800  |
| 37     | envelope<sub>y</sub>|  2.0 |    1.0 |   1.6076  |   1.2685  |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 283           | 121           |
| Execution time  | 1130.48 s     | 437.33 s      |
| Residual        | 7.351 × 10⁻²  | 2.015 × 10⁻¹  |


### Third Chromacity Quad
method: Nelder-Mead

objective:
- $D_x = 0$ at position **55**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 50     |    I     |      4.7162 |    4.6817 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 55     | D<sub>x</sub> |  0.0 |    1.0 |  −1.277 × 10⁻⁶  |   4.527 × 10⁻⁸  |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 40            | 40            |
| Execution time  | 157.41 s      | 149.99 s      |
| Residual        | 5.242 × 10⁻¹⁴ | 2.050 × 10⁻¹⁵ |


### Quadrupole Doublet and Interaction Point
method: Nelder-Mead

objective:
- envelope$_x = 0$ at position **59**
- envelope$_y = 0$ at position **59**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 56     |    I     |      3.1222 |    3.1235 | (0, 10)  |
| 58     |   I<sub>2</sub> | 3.3169 | 3.3140 | (0, 10)  |

**Objectives**

| Indice | Quantity            | Goal | Weight | PyTorch | NumPy |
|:------:|:-------------------:|-----:|-------:|--------:|------:|
| 59     | envelope<sub>x</sub>|  0.0 |    1.0 |  0.0201 | 0.0189 |
| 59     | envelope<sub>y</sub>|  0.0 |    1.0 |  0.0485 | 0.0699 |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 78            | 72            |
| Execution time  | 299.51 s      | 274.74 s      |
| Residual        | 1.378 × 10⁻³  | 2.623 × 10⁻³  |


### Quadrupole Doublet
method: Nelder-Mead

objective:
- $\alpha_x = 0, \beta_x = 0.1$ at position **68**
- $\alpha_y = 0, \beta_y = 0.1$ at position **69**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 61     |    I     |      5.1765 |    5.1754 | (0, 10)  |
| 63     |   I<sub>2</sub> | 4.0452 | 4.0348 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 68     | α<sub>x</sub> |  0.0 |    1.0 |   6.958 × 10⁻⁴  |  −9.083 × 10⁻⁴  |
| 68     | β<sub>x</sub> |  0.1 |    0.5 |   0.0527 m      |   0.0472 m      |
| 69     | α<sub>y</sub> |  0.0 |    1.0 |   5.829 × 10⁻⁴  |   4.094 × 10⁻⁵  |
| 69     | β<sub>y</sub> |  0.1 |    0.5 |   0.0131 m      |   0.0249 m      |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 98            | 94            |
| Execution time  | 352.30 s      | 342.32 s      |
| Residual        | 1.224 × 10⁻³  | 1.054 × 10⁻³  |


### Fourth Chromacity Quad
method: Nelder-Mead

objective:
- $D_x = 0$ at position **75**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 70     |    I     |      4.6260 |    4.6721 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 75     | D<sub>x</sub> |  0.0 |    1.0 |   5.585 × 10⁻⁷  |   7.311 × 10⁻⁷  |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 40            | 40            |
| Execution time  | 134.15 s      | 141.32 s      |
| Residual        | 3.012 × 10⁻¹³ | 1.266 × 10⁻¹³ |


### Quadrupole Triplet
method: Nelder-Mead

objective:
- $\alpha_x = 0, \beta_x = 0.1$ at position **85**
- $\alpha_y = 0, \beta_y = 0.1$ at position **86**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 76     |    I     |      3.9413 |    3.9258 | (0, 10)  |
| 78     |   I<sub>2</sub> | 4.2122 | 4.0720 | (0, 10)  |
| 80     |   I<sub>3</sub> | 0.2049 | 0.0090 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 85     | α<sub>x</sub> |  0.0 |    1.0 |   2.732 × 10⁻⁴  |   1.296 × 10⁻⁴  |
| 85     | β<sub>x</sub> |  0.1 |    0.5 |   0.0860 m      |   0.0724 m      |
| 86     | α<sub>y</sub> |  0.0 |    1.0 |  −1.158 × 10⁻⁵  |  −1.718 × 10⁻³  |
| 86     | β<sub>y</sub> |  0.1 |    0.5 |   0.0748 m      |   0.1505 m      |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 208           | 150           |
| Execution time  | 717.06 s      | 552.42 s      |
| Residual        | 1.041 × 10⁻⁴  | 4.150 × 10⁻⁴  |


### Fifth Chromacity Quad
method: Nelder-Mead

objective:
- $D_x = 0$ at position **92**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 87     |    I     |      3.8863 |    3.9451 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-----:|-------:|----------------:|----------------:|
| 92     | D<sub>x</sub> |  0.0 |    1.0 |   5.874 × 10⁻⁷  |   5.509 × 10⁻⁷  |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 40            | 40            |
| Execution time  | 143.35 s      | 147.98 s      |
| Residual        | 7.025 × 10⁻¹⁶ | 2.491 × 10⁻¹⁵ |


### Quadrupole Triplet and MkIII Undulator Start
method: Nelder-Mead

objective:
- $\alpha_x = 0, \beta_x = 0.2418$ at position **117**
- $\alpha_y = 0, \beta_y = 0.2418$ at position **117**

#### Optimization Results - pytorch vs numpy

**Decision variables**

| Indice | Variable | PyTorch (A) | NumPy (A) | Bounds   |
|:------:|:--------:|------------:|----------:|:--------:|
| 93     |    I     |      1.3977 |    1.2627 | (0, 10)  |
| 95     |   I<sub>2</sub> | 3.5109 | 3.5298 | (0, 10)  |
| 97     |   I<sub>3</sub> | 2.2532 | 2.3209 | (0, 10)  |

**Objectives**

| Indice | Quantity      | Goal   | Weight | PyTorch         | NumPy           |
|:------:|:-------------:|-------:|-------:|----------------:|----------------:|
| 117    | α<sub>x</sub> | 0.0000 |    1.0 |   8.466 × 10⁻³  |   8.676 × 10⁻³  |
| 117    | α<sub>y</sub> | 0.0000 |    1.0 |   3.514 × 10⁻⁵  |   6.698 × 10⁻⁴  |
| 117    | β<sub>x</sub> | 0.2418 |    1.0 |   0.0806 m      |   0.0823 m      |
| 117    | β<sub>y</sub> | 0.2418 |    1.0 |   0.2248 m      |   0.2227 m      |

#### Evaluation Analysis

| Metric          | PyTorch       | NumPy         |
|-----------------|---------------|---------------|
| Iterations      | 243           | 310           |
| Execution time  | 786.13 s      | 1149.20 s     |
| Residual        | 6.588 × 10⁻³  | 6.476 × 10⁻³  |