optimization methods summary

# Nelder-Mead

## Problem

Nelder-Mead is a derivative-free method for unconstrained minimization:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

It does not require gradient information. Instead, it maintains a simplex of $n+1$ vertices in an $n$-dimensional search space.

## Overview

At each iteration, Nelder-Mead orders the simplex vertices by objective value and tries to replace the worst vertex with a better point using geometric transformations:

1. **Reflection**: move the worst point through the centroid.
2. **Expansion**: move further if reflection is very successful.
3. **Contraction**: move closer if reflection fails.
4. **Shrink**: collapse the simplex toward the best point if contraction fails.

## Notation

Let the simplex at iteration $k$ be

$$
V_k = \{x_1, x_2, \dots, x_{n+1}\}
$$

After sorting by objective value:

$$
f(x_1) \le f(x_2) \le \cdots \le f(x_n) \le f(x_{n+1})
$$

where:

- $x_1$: best vertex
- $x_n$: second-worst vertex
- $x_{n+1}$: worst vertex

The centroid of all vertices except the worst is

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

Common coefficients are:

| coefficient | meaning | typical value |
|---|---|---:|
| $\alpha$ | reflection coefficient | $1$ |
| $\gamma$ | expansion coefficient | $2$ |
| $\rho$ | contraction coefficient | $0.5$ |
| $\sigma$ | shrink coefficient | $0.5$ |

## Algorithm

- Step 0: Initialization

Choose an initial point $x^{(0)}$ and construct an initial simplex with $n+1$ vertices:

$$
V_0 = \{x_1, x_2, \dots, x_{n+1}\}
$$

Usually, the additional vertices are created by perturbing $x^{(0)}$ along each coordinate direction.

Evaluate $f(x_i)$ for all simplex vertices.

- Step 1: Sort vertices

Sort the vertices so that

$$
f(x_1) \le f(x_2) \le \cdots \le f(x_{n+1})
$$

- Step 2: Compute centroid

Compute the centroid excluding the worst vertex $x_{n+1}$:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

- Step 3: Reflection

Reflect the worst vertex across the centroid:

$$
x_r = \bar{x} + \alpha(\bar{x} - x_{n+1})
$$

Evaluate $f(x_r)$.

Decision:

$$
x_{n+1} \leftarrow
\begin{cases}
x_r, & f(x_1) \le f(x_r) < f(x_n) \\
\text{go to expansion}, & f(x_r) < f(x_1) \\
\text{go to contraction}, & f(x_r) \ge f(x_n)
\end{cases}
$$

- Step 4: Expansion

If the reflected point is better than the current best point, try expanding further:

$$
x_e = \bar{x} + \gamma(x_r - \bar{x})
$$

Evaluate $f(x_e)$ and update:

$$
x_{n+1} \leftarrow
\begin{cases}
x_e, & f(x_e) < f(x_r) \\
x_r, & f(x_e) \ge f(x_r)
\end{cases}
$$

- Step 5: Contraction

If reflection is not good enough, perform contraction.

Outside contraction, when $f(x_n) \le f(x_r) < f(x_{n+1})$:

$$
x_c = \bar{x} + \rho(x_r - \bar{x})
$$

Inside contraction, when $f(x_r) \ge f(x_{n+1})$:

$$
x_c = \bar{x} - \rho(\bar{x} - x_{n+1})
$$

If the contracted point improves the objective, replace the worst vertex with $x_c$. Otherwise, go to the shrink step.

- Step 6: Shrink

If contraction fails, shrink all vertices toward the best vertex:

$$
x_i \leftarrow x_1 + \sigma(x_i - x_1), \quad i = 2, \dots, n+1
$$

Then re-evaluate the objective values for the updated vertices.

- Step 7: Check convergence

Stop if both the simplex size and objective spread are small:

$$
\max_{2 \le i \le n+1}\|x_i - x_1\|_\infty < \epsilon_x
$$

and

$$
\max_{2 \le i \le n+1}|f(x_i) - f(x_1)| < \epsilon_f
$$

Otherwise, repeat from Step 1.



# L-BFGS-B

Limited Memory Broyden-Fletcher-Goldfarb-Shanno algorithm with Box constraints

## BFGS algorithm

Broyden-Fletcher-Goldfarb-Shanno algorithm
while $\|\nabla f(x_k)\| > \epsilon$:

- step 1: compute descent direction **$d_k$** from $B_k d_k = - \nabla f(x_k)$, so that $d_k = - B_k^{-1} \nabla f(x_k)$
- step 2: Perform line search method(exact/backtracking) and find step size $\alpha_k$ that satisfying the Wolfe
  conditions
- step 3: update $x_{k+1} \leftarrow x_k + \alpha_k d_k$, set $s_k=\alpha_k d_k$
  and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$
- step 4: update $B_{k+1} \leftarrow B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$, so
  that $B_{k+1}^{-1} = B_k^{-1} + \frac{(s_k^T y_k + y_k^T B_k^{-1} y_k)(s_k s_k^T)}{(s_k^T y_k)^2} - \frac{B_k^{-1} y_k s_k^T + s_k y_k^T B_k^{-1}}{s_k^T y_k} = (I-\frac{s_k y_k^T}{y_k^T s_k}) B_k^{-1} (I-\frac{y_k s_k^T}{y_k^T s_k}) + \frac{s_k s_k^T}{y_k^T s_k}$,
  store it for the next iteration
- step 5: $k \leftarrow k + 1$

## L-BFGS algorithm

while $\|\nabla f(x_k)\| > \epsilon$:
- step 1: compute descent direction $d_k$ without explicitly forming the matrix $B_k^{-1}$ via the **Two-Loop Recursion**:
  - initialize $q = \nabla f(x_k)$
  - let $\hat{m} = \min(k, m)$ be the number of currently stored history pairs.
  - **backward loop:** for $i = k-1$ down to $k-\hat{m}$:
      $\rho_i = \frac{1}{y_i^T s_i}$
      $\alpha_i = \rho_i s_i^T q$
      $q \leftarrow q - \alpha_i y_i$
  - **base matrix scaling:** if $k > 0$, calculate $\gamma_k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}$; else, $\gamma_k = 1$
  - set $r = \gamma_k q$ 
  - **forward loop:** for $i = k-\hat{m}$ up to $k-1$:
      $\beta = \rho_i y_i^T r$
      $r \leftarrow r + s_i (\alpha_i - \beta)$
  - optimal direction: $d_k = -r$
  
- step 2: Perform line search method (exact/backtracking) and find step size $\alpha_k$ that satisfies the **Wolfe conditions**
- step 3: update $x_{k+1} \leftarrow x_k + \alpha_k d_k$, set $s_k=\alpha_k d_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$
- step 4: if $s_k^T y_k > 0$, push the new vector pair $(s_k, y_k)$ into the memory queue. If the queue size exceeds $m$, pop and discard the oldest vector pair $(s_{k-m}, y_{k-m})$.
- step 5: $k \leftarrow k + 1$





# SLSQP

Sequential Least Squares Programming

## Sequential quadratic programming

## problem

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{subject to} \quad & h(x) = 0 \in \mathbb{R}^m \\
& g(x) \le 0 \in \mathbb{R}^k \\
& l_i \le x_i \le u_i, i\in M \subseteq [n]
\end{aligned}
$$

## algorithm

suppose Lagrangian function is $\mathcal{L}(x, \lambda, \mu) = f(x) + \lambda^T h(x) + \mu^T g(x)$,
where $\lambda \in \mathbb{R}^m$ and $\mu \ge 0, \mu \in \mathbb{R}^k$ are the Lagrange multipliers for equality and
inequality constraints respectively.

while $\|\nabla \mathcal{L}(x_k, \lambda_k, \mu_k)\| > \epsilon$:

- step 0: initialize $x_0$ as initial solution and $B_0$ as Hessian estimation for $\mathcal{L}$
- step 1: solve the following quadratic programming subproblem and get the optimal solution $d_k$ as the descent
  direction:

$$
\begin{aligned}
\min_{d \in \mathbb{R}^n} \quad &
\nabla f(x_k)^T d + \frac{1}{2} d^T B_k d \\
\text{s.t.} \quad & h(x_k) + \nabla h(x_k)^T d = 0 \in \mathbb{R}^m \\
& g(x_k) + \nabla g(x_k)^T d \le 0 \in \mathbb{R}^k \\
& l_i \le x_{ki} + d_i \le u_i, i\in M \subseteq [n] \Leftrightarrow l - x_{k,M} \le P_M d \le u - x_{k,M}
\end{aligned}
$$

- step 2: By Armijo rule, determine step size $\alpha_k$ in backtracking line search, so that
    - Define $\Phi(x) = f(x) + \sum\limits_{i=1}^m\nu_i|h_i(x)|+\sum\limits_{j=1}^k\rho_jReLU(g_j(x))$
    - $\Phi(x_k + \alpha_k d_k) \le \Phi(x_k) + c \alpha_k \nabla \Phi(x_k)^T d_k$, where $c$ is a constant in (0, 1)
- step 3: update $x_{k+1} \leftarrow x_k + \alpha_k d_k$, set $s_k=\alpha_k d_k$
  and $y_k = \nabla \mathcal{L}(x_{k+1}, \lambda_{k+1}, \mu_{k+1}) - \nabla \mathcal{L}(x_k, \lambda_{k+1}, \mu_{k+1})$
- step 4:
    - $y_k \leftarrow \theta y_k+(1-\theta)B_ks_k$, and pick
      $\theta_k =
      \begin{cases}
      1, & \text{if } s_k^T y_k \ge 0.2 \, s_k^T B_k s_k \\
      \frac{0.8 \, s_k^T B_k s_k}{s_k^T B_k s_k - s_k^T y_k}, & \text{if } s_k^T y_k < 0.2 \, s_k^T B_k s_k
      \end{cases}
      $
    - update $B_{k+1} \leftarrow B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$
- step 5: $k \leftarrow k + 1$

solve step 1 by least squares method:

- By cholesky decomposition, we can decompose $B_k$ as $B_k = L L^T$, where $L$ is a lower triangular matrix.
- notice $z=L^{\top} d$, and $d=L^{-\top} z$
- transform the original problem into

$$
\begin{aligned}
\min_{z \in \mathbb{R}^n} \quad &
\frac{1}{2}\|z - \tilde{b}\|_2^2 \\
\text{s.t.} \quad & h(x_k) + M(x_k)z = 0 \in \mathbb{R}^m \\
& g(x_k) + N(x_k)z \le 0 \in \mathbb{R}^k \\
& l - x_{k,M} \le P_M (L^{\top})^{-1}z \le u - x_{k,M}
\end{aligned}
$$

where
$$
z = L^{\top} d, \tilde{b} = - L^{-1} \nabla f(x_k), M(x_k) = (L^{-1}\nabla h(x_k))^{\top}, N(x_k) = (L^{-1}\nabla g(x_k))^{\top}
$$

# COBYLA

full name: Constrained Optimization BY Linear Approximation

gradient free algorithm

## problem
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \,\, g(x) \le 0 \in \mathbb{R}^k
\end{aligned}
$$

## algorithm
- **step 0: Initialization** 
  - Choose an initial point $x_0$
  - Define the initial trust-region radius $\rho_{beg}$ (macro step size) and the final stopping tolerance $\rho_{end}$.
  - Construct an initial simplex $V = \{x_0, x_1, \dots, x_n\}$ by perturbing $x_0$ along each coordinate axis by the distance $\rho_{beg}$.
  - Evaluate the true objective $f(x)$ and constraints $g_j(x)$ at all $n+1$ vertices using the physical simulation black-box. Set $\rho = \rho_{beg}$ and $k=0$.

while $\rho > \rho_{end}$:

denote $x_k=x_{best}$ as the best solution in the simplex

- step 1: define the linear interpolation model
  
  - $\hat{f}(x)=a_0+\mathbf{a}^{\top}x$
  - $\hat{g}_i(x)=b_{0,i}+\mathbf{b_i}^{\top}x \quad \forall i \in [k]$
  
- step 2: solve both $\hat{f}$ and $\hat{g}_i\, \forall i \in [k]$ from the following square linear systems

  - $\hat{f}(x_j)=f(x_j) \quad\forall j=0, \dots, n$ to solve $\hat{f}$
  - $\hat{g_i}(x_j)=g_i(x_j) \quad\forall j=0, \dots, n$ to solve $\hat{g_i}$

  

- step 3: solve the Trust-Region LP Subproblem
  $$
  \begin{aligned} \min_{d \in \mathbb{R}^n} \quad & \hat{f}(x_k+d)=a_0+a^{\top}(x_k+d) \\ \text{s.t.} \quad & g_j(x_{k}) + b_j^{\top} d \le 0, \quad \forall j \\ & \|d\|_{\infty} \le \rho \quad \text{(Trust-Region bound)} \end{aligned}
  $$
  

  - Obtain the candidate trial point: $x_{k+1} \leftarrow x_{k} + d_k$.

- step 4: Evaluation & Merit Function

  - Feed $x_{k+1}$ into the physical simulation to evaluate the true $f(x_{k+1})$ and $g_j(x_{k+1})$.
  - Define a penalty-based Merit Function to evaluate the actual improvement, balancing objective minimization and constraint violations with a self-adjusted parameter $\mu$:
    $$
    \Phi(x) = f(x) + \mu \max_{j \in [k]} \big( \text{ReLU}(g_j(x)) \big) \\
    \hat\Phi(x) = \hat{f}(x) + \mu \max_{j \in [k]} \big( \text{ReLU}(\hat{g}_j(x)) \big)
    $$

- **step 4: Simplex Update & Geometry Rescue**

  - compute $r_{actual}=\Phi(x_{k})-\Phi(x_{k+1})$, $r_{pred}=\hat\Phi(x_{k})-\hat\Phi(x_{k+1})$​
  - and then get $\gamma_k=\frac{r_{actual}}{r_{pred}}$

- **step 5: Trust-Region Radius ($\rho$) and simplex update**

  - $\rho_{k+1}=\begin{cases}
    \min(ρ_{begin},1.2ρ_k), & \gamma_k \gt 0.7 \\
    \rho_{k}, & 0.1 \le \gamma_k \le 0.7 \\
    \frac{1}{2}\rho_{k}, & \gamma_k \lt 0.1 \\
    \end{cases}$
  - replace $x_{worst}$ from V as $x_{k+1}$ if $\gamma_k \ge 0.1$

- **step 6:** $k \leftarrow k + 1$

# trust-constr
