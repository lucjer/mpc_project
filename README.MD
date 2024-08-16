# Nonlinear Model Predictive Control (NMPC) Solver

This repository contains an implementation of a Nonlinear Model Predictive Control (NMPC) solver, utilizing Sequential Quadratic Programming (SQP) and Real-Time Iteration (RTI) algorithms. These algorithms are derived from the work of Gros et al. (2020), which bridges the gap between linear and nonlinear MPC. The purpose of this repository is to demonstrate how to effectively solve a complex NMPC problem using OSQP in Python.

## Background

The NMPC solver is based on the following paper:

**Reference:**
Gros, S., Zanon, M., Quirynen, R., Bemporad, A., & Diehl, M. (2020). From linear to nonlinear MPC: bridging the gap via the real-time iteration. *International Journal of Control, 93*(1), 62-80. [Read the paper](https://cse.lab.imtlucca.it/~bemporad/publications/papers/ijc_rtiltv.pdf)

## Inspiration and Design

The implementation of the NMPCSolver class draws inspiration from the [MPC class](https://github.com/matssteinweg/Multi-Purpose-MPC/blob/master/src/MPC.py) found in a public repository by Mats Steinweg. However, this solver differs in several important ways:

- **Modularity:** The implementation is highly modular, allowing for easy customization and adaptation to various use cases.
- **General Applicability:** Unlike the referenced implementation, this solver does not assume any specific dynamical model (e.g., vehicle model). It is designed to be general-purpose, capable of handling a wide range of dynamical systems with minimal adaptation.

## Implementation Details - NMPC and SQP Formulation [Gros et al. (2020)]

**Nonlinear Programming (NLP) Problem:**

Given the state estimate \(\hat{x}_i\) and reference trajectories \(\boldsymbol{x}_i^{\text{ref}}\) and \(\boldsymbol{u}_i^{\text{ref}}\), the optimization problem is:

\[
\operatorname{NLP}\left(\hat{x}_i, \boldsymbol{x}_i^{\text{ref}}, \boldsymbol{u}_i^{\text{ref}}\right) = \arg \min_{x, u} \sum_{k=0}^{N-1} \frac{1}{2}
\begin{bmatrix}
x_{i, k} - x_{i, k}^{\text{ref}} \\
u_{i, k} - u_{i, k}^{\text{ref}}
\end{bmatrix}^\top
W_{i, k}
\begin{bmatrix}
x_{i, k} - x_{i, k}^{\text{ref}} \\
u_{i, k} - u_{i, k}^{\text{ref}}
\end{bmatrix}
\]

Subject to:

- Initial state: \(x_{i, 0} = \hat{x}_i\)
- System dynamics: \(x_{i, k+1} = f(x_{i, k}, u_{i, k})\) for \(k = 0, \dots, N-1\)


**Sequential quadratic programming (SQP) for NMPC**
In the SQP approach, the optimization problem is iteratively approximated by quadratic programming (QP) problems. Each QP provides a Newton direction, guiding steps toward the solution from the current guess. These steps, partial or full, are repeated until convergence.


At a guess $\left(\boldsymbol{x}_i^{\text {guess }}, \boldsymbol{u}_i^{\text {guess }}\right)$, Problem (9) is approximated by the QP:
$$
\begin{array}{r}
\mathrm{QP}_{\mathrm{NMPC}}\left(\hat{x}_i, \boldsymbol{x}_i^{\text {guess }}, \boldsymbol{u}_i^{\text {guess }}, \boldsymbol{x}_i^{\text {ref }}, \boldsymbol{u}_i^{\text {ref }}\right)= \\
\arg \min _{\Delta x_i, \Delta u_i} \sum_{k=0}^{N-1} \frac{1}{2}\left[\begin{array}{c}
\Delta x_{i, k} \\
\Delta u_{i, k}
\end{array}\right] H_{i, k}\left[\begin{array}{c}
\Delta x_{i, k} \\
\Delta u_{i, k}
\end{array}\right]+J_{i, k}^T\left[\begin{array}{c}
\Delta x_{i, k} \\
\Delta u_{i, k}
\end{array}\right] \\
\text { s. t. } \Delta x_{i, 0}=\hat{x}_i-x_{i, 0}^{\text {guess }}, \\
\Delta x_{i, k+1}=A_{i, k} \Delta x_{i, k}+B_{i, k} \Delta u_{i, k}+r_{i, k}, \\
C_{i, k} \Delta x_{i, k}+D_{i, k} \Delta u_{i, k}+h_{i, k} \leq 0,
\end{array}
$$
with 
$$
\begin{aligned}
& A_{i, k}=\left.\frac{\partial f(x, u)}{\partial x}\right|_{x_{i, k}^{\text {guess }}, u_{i, k}^{\text {guess }}}, \\
& B_{i, k}=\left.\frac{\partial f(x, u)}{\partial u}\right|_{x_{i, k} \text { guess }, u_{i, k}^{\text {guess }}}, \\
& C_{i, k}=\left.\frac{\partial h(x, u)}{\partial x}\right|_{x_{i, k}^{\text {guess }}, u_{i, k}^{\text {guess }}}, \\
& D_{i, k}=\left.\frac{\partial h(x, u)}{\partial u}\right|_{x_{i, k} \text { guess }, u_{i, k}^{\text {guess }}}, \\
& r_{i, k}=f\left(x_{i, k}^{\text {guess }}, u_{i, k}^{\text {guess }}\right)-x_{i, k+1}^{\text {guess }}, h_{i, k}=h\left(x_{i, k}^{\text {guess }}, u_{i, k}^{\text {guess }}\right), \\
& J_{i, k}=W_{i, k}\left[\begin{array}{l}
x_{i k}^{\text {guess }}-x_{i, k}^{\text {ref }} \\
u_{i, k}^{\text {guess }}-u_{i, k}^{\text {ref }}
\end{array}\right], \\
&
\end{aligned}
$$

### Algorithm 1: SQP for NMPC at Discrete Time \(i\)

**Input:**
- Current state estimate \(\hat{x}_i\)
- Reference trajectory \((\boldsymbol{x}_i^{\text{ref}}, \boldsymbol{u}_i^{\text{ref}})\)
- Initial guess \((\boldsymbol{x}_i^{\text{guess}}, \boldsymbol{u}_i^{\text{guess}})\)

**Procedure:**

1. **While** the solution has not converged:
   1. Evaluate \(r_{i,k}\), \(h_{i,k}\), and the sensitivities \(A_{i,k}\), \(B_{i,k}\), \(C_{i,k}\), \(D_{i,k}\), \(H_{i,k}\), \(J_{i,k}\) using equation (12).
   2. Construct and solve the quadratic program \(\text{QP}_{\text{NMPC}}(\hat{x}_i, \boldsymbol{x}_i^{\text{guess}}, \boldsymbol{u}_i^{\text{guess}}, \boldsymbol{x}_i^{\text{ref}}, \boldsymbol{u}_i^{\text{ref}})\) as defined in equation (11) to obtain the Newton direction \((\Delta \boldsymbol{x}_i, \Delta \boldsymbol{u}_i)\).
   3. Update the guess with the Newton step:
      \[
      (\boldsymbol{x}_i^{\text{guess}}, \boldsymbol{u}_i^{\text{guess}}) \leftarrow (\boldsymbol{x}_i^{\text{guess}}, \boldsymbol{u}_i^{\text{guess}}) + \alpha (\Delta \boldsymbol{x}_i, \Delta \boldsymbol{u}_i)
      \]
   
2. **End While**

**Output:**
- NMPC solution \((x_i, u_i) = (\boldsymbol{x}_i^{\text{guess}}, \boldsymbol{u}_i^{\text{guess}})\)


### Implementation details: SQP for NMPC at Discrete Time \(i\)
