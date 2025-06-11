# Physics-Informed Neural Networks: Solving PDEs with PINNs

This project applies and extends a **Physics-Informed Neural Network (PINN)** to solve nonlinear partial differential equations (PDEs). PDEs such as the **Burgers' equation** play a fundamental role in modeling fluid dynamics, traffic flow, and nonlinear acoustics.

We reproduce and build upon the classical example from [Raissi et al. (2019)](https://doi.org/10.1016/j.jcp.2018.10.045), demonstrating how PINNs can learn the underlying physics directly from data and differential constraints.

---

## What is a PINN?

A **Physics-Informed Neural Network (PINN)** is a deep learning model that incorporates prior knowledge of the governing physics (typically in the form of PDEs) into the training process.  
This is achieved by adding the residual of the PDE (computed via **automatic differentiation**) to the loss function, alongside losses from boundary and initial conditions.

---

## Project Overview

The repository contains two main experiments:

1. **Reproducing the 1D Burgers' Equation PINN**  
   We re-implement the example from Raissi et al., solving the nonlinear 1D Burgers' equation. This serves as a baseline for understanding the training dynamics and structure of standard PINNs.

2. **Original Extension: Learning a Family of Wave Equations**  
   As a personal extension of the PINN framework, I tried to train a neural network that can solve the wave equation for a **family of Dirichlet boundary conditions**.  
   The idea is to represent the boundary function $ g(x) $ using a truncated sine Fourier series, and to feed the Fourier coefficients $(A_1, \dots, A_n)$  as extra inputs to the network:
   $$
   \text{NN}(x, t; A_1, \dots, A_n) \approx u(x, t)
   $$
   This way, the network learns to approximate a **solution operator** that maps different boundary profiles to their corresponding PDE solutions.  
   The idea was inspired by parameterized PDE models in the literature, but here the focus is on generalizing over boundary conditions rather than physical parameters.


---

## Method

We implement the PINNs using TensorFlow, training the network to minimize a composite loss function consisting of:

- Residual loss of the PDE (via automatic differentiation),
- Boundary condition loss (Dirichlet),
- Initial condition loss.

In the wave-equation extension, the boundary conditions are not fixed; instead, their Fourier coefficients are sampled and provided as inputs during training.

---

## Files

- `Example1BurgersEquation.ipynb` — Reproduces the 1D Burgers' equation example from the original paper.
- `PINN_WaveEq_Family_GitHubReady.ipynb` — Experiments with solving a family of wave equations using the PINNs framwork.
- `README.md` — This document.

---

## Results

- The PINN is able to accurately learn the solution to the 1D Burgers' equation.
- In the extended version, we demonstrate partial success in generalizing to parametric PDEs.
- Plots include predicted vs. ground truth, and error metrics over the domain.

---

## Requirements

```bash
pip install torch numpy matplotlib scipy

---

## References

- M. Raissi, P. Perdikaris, G.E. Karniadakis,  
  *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*,  
  Journal of Computational Physics, 378 (2019): 686–707. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045)

