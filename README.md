# 📘 Material Parameter Identification via FEM + CasADi Optimization

*A modular framework for dynamic FEM simulation, symbolic differentiation, and IPOPT-based material calibration.*

---

## 🚀 Overview

This repository provides a full pipeline for:

* **Finite Element simulation** of deformable objects
* **Symbolic differentiation** of the full dynamic model (via **CasADi**)
* **Nonlinear optimization** of material parameters (Young’s modulus **E** and Poisson ratio **ν**)
* **Trajectory-based material calibration** to match a target deformation sequence

This enables a **new strategy** for calibrating elastic materials:

> Tune FEM material parameters so a simulated object behaves like a reference object —
> either another FEM simulator, a physics engine, or real-world measured motion.

This lets you:

* Match simulation models across engines
* Fit physical materials from recorded motion data
* Validate differentiable simulators
* Perform inverse elasticity estimation
* Build data-driven soft robotics models

---

# 📂 Repository Structure

```
.
├── optimize_material_modular.py     # Main FEM + optimizer module
├── plots_optimization.py            # Plotting & visualization tools
├── fem_comparison_animation.mp4     # Example animation of results
├── README.md                        # This file
└── data/                            # (optional) target trajectories, etc.
```

---

# 🔧 Core Components

## 1️⃣ FEMModel

* Tetrahedral mesh
* Linear elasticity
* Computes element stiffness matrices
* Computes volumes, masses, B‐matrices
* Assembles global matrices (K, M, C)
* Applies boundary constraints

## 2️⃣ FEMSimulator

* Runs a **dynamic FEM simulation**
* Unrolls time steps symbolically using CasADi
* Computes internal/external forces at each step
* Provides symbolic objective:
  $$
  J(E,\nu) = \tfrac12 \sum_t |u_t^{sim}(E,\nu) - u_t^{target}|^2
  $$

## 3️⃣ MaterialOptimizer ⭐ **Main optimizer (recommended)**

Runs IPOPT **once** on the full nonlinear program:

* Uses reparameterizations:

  * $E = \exp(\log E)$
  * $\nu = 0.1\tanh(t_\nu) + 0.4$
* Ensures stable variables during optimization
* Returns optimal material parameters

## 4️⃣ MaterialOptimizer_Verbose 🐞 **Debugging version**

* IPOPT runs in *one-iteration increments*
* Prints detailed logs:

  * Objective value per iteration
  * Gradients
  * Parameter evolution
  * Convergence diagnostics
* Useful for diagnosing:

  * vanishing/exploding gradients
  * long trajectory horizon
  * ill-conditioning

---

# 🧠 Theory

## Dynamic FEM model

The governing equation solved at each timestep:

$$
M\ddot{u} + C\dot{u} + Ku = f_{\text{ext}}
$$

Integrated using:

$$
\dot{u}_{t+1} = \dot{u}_t + \Delta t * a_t \\
u_{t+1} = u_t + \Delta t * \dot{u}_{t+1}
$$

---

## Objective function

Given target displacements $u_t^*$,

$$
J(E,\nu) = \frac{1}{2} \sum_{t=0}^{T} |u_t(E,\nu) - u_t^*|^2
$$

---

## Material reparameterization

Ensures valid domains and stable optimization:

```
E  = exp(logE)
nu = 0.1 * tanh(t_nu) + 0.4
```

Thus:

* $E > 0$
* $\nu \in (0.3, 0.5)$
---

## CasADi + IPOPT

* Entire simulation graph is symbolic
* CasADi provides exact gradients/Hessians
* IPOPT solves the nonlinear program
* Convergence detected by KKT satisfaction

---

# ▶️ Running the Optimization

## Example script

```python
from optimize_material_modular import FEMModel, FEMSimulator, MaterialOptimizer
import numpy as np

# Load mesh + target trajectory
nodes = np.load("nodes.npy")
tets = np.load("tets.npy")
target_us = np.load("target_us.npy")

model = FEMModel(nodes, tets)
sim   = FEMSimulator(model, dt=1e-3, n_steps=len(target_us))

opt = MaterialOptimizer(sim, target_us)

E_opt, nu_opt = opt.solve(E_guess=5e4, nu_guess=0.45)

print("Optimal E:", E_opt)
print("Optimal ν:", nu_opt)
```

---

# 📊 Plotting Results

The script `plots_optimization.py` provides convenience functions for:

### ✔ Objective history

### ✔ Parameter trajectories (E, ν)

### ✔ Final vs. target deformation comparison

### ✔ 3D node trajectories

### ✔ Per-node displacement errors
---
It expects to find in the directory:
```python
nodes_ref = np.load("nodes_ref.npy")
u_target_series = np.load("u_target_series.npy")
u_series_opt = np.load("u_series_opt.npy")
tets = np.load("tets.npy")
F_ext_series = np.load("F_ext_series.npy")
```

# 🎞 Animation

This repository includes an example animation that compares:

* the **target FEM trajectory**
* the **optimized FEM trajectory**

Embed inside README:

## 📹 FEM Comparison Animation

*(Actual animation included in repo)*

![Animation](fem_comparison_animation.mp4)

---

# 🧪 Full Workflow

### 1. Generate or load target trajectories

From simulation, real data, motion capture, etc.

### 2. Run optimization

Choose optimizer type:

* `MaterialOptimizer` → fast, production use
* `MaterialOptimizer_Verbose` → debug mode

### 3. Save results

Both optimizers save:

* optimized parameters
* simulated trajectories
* intermediate files
* error curves

### 4. Plot and animate

Call functions in `plots_optimization.py`.

---

# ⚠️ Troubleshooting

### 🟥 Problem: IPOPT converges but J is large

Likely cause: **vanishing gradients** in long time horizons.

**Fixes:**

* Subsample timesteps
* Increase damping
* Use `MaterialOptimizer_Verbose`
* Use shorter “windows” of simulation
* Normalize objective by number of time steps

---

### 🟥 Problem: Convergence fails for long trajectories

Reason: Jacobian product through many timesteps becomes ill-conditioned.

**Fixes:**

* multiple-shooting optimization
* windowed optimization
* adjoint-based backward integration (coming soon)
* reduce timestep count
* regularize J

---

### 🟥 Problem: Memory usage is high

CasADi graphs grow with number of time steps.

**Fixes:**

* reduce `n_steps`
* enable CasADi JIT compilation
* run in verbose mode to see memory growth

---

# 🧭 Use Cases

### ✔ material calibration from real deformation data

### ✔ matching two FEM simulators (cross-engine calibration)

### ✔ validating differentiable physics models

### ✔ learning soft-robotics material behavior

### ✔ fitting constitutive parameters for biological tissues

---

# 🤝 Contributing

PRs are welcome, especially for:

* new materials (Neo-Hookean, corotated, etc.)
* adjoint-based gradients
* GPU acceleration (planned)
* improved plotting/animation
* integration with PyTorch or JAX

---

# 📜 License

*(Choose one and update this section.)*

---

# 🎉 Final Notes

This framework is meant to be:

* research-grade
* explainable
* modular
* extendable
* educational
* practical for real calibration tasks
