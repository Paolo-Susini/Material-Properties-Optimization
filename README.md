### 📄 **`README.md`**

# 🧠 Dynamic FEM Material Parameter Identification with CasADi

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python Badge">
  <img src="https://img.shields.io/badge/CasADi-yellow?logo=casadi&logoColor=black" alt="CasADi Badge">
  <img src="https://img.shields.io/badge/IPOPT-brightgreen?logo=ipopt&logoColor=black" alt="IPOPT Badge">
  <img src="https://img.shields.io/badge/NumPy-white?logo=numpy&logoColor=blue" alt="NumPy Badge">
  <img src="https://img.shields.io/badge/Matplotlib-grey?logo=matplotlib&logoColor=white" alt="Matplotlib Badge">
</p>

---

### 🧩 Overview

This project performs **dynamic material parameter identification** for a 3D deformable object using **CasADi**.  
It constructs a **non-linear Finite Element Method (FEM)** simulation symbolically, then applies **gradient-based optimization** to identify material properties — **Young’s Modulus** ($E$) and **Poisson’s Ratio** ($\nu$) — that best match a target motion.

> 🧮 In essence: this is an **inverse problem**, identifying the causes (material properties) from observed effects (motion).

---

### 🎥 Demo

You can visualize the optimization with `plots_optimization.py`.  
To export an animation as a GIF, simply add this before `plt.show()`:

```python
ani.save("animation.gif", writer="pillow")
````

> If needed, install Pillow:
>
> ```bash
> pip install pillow
> ```
<p align="center">
  <img src="cube_animation.gif" alt="Dynamic FEM Cube Animation" width="500"/>
</p>
---

### 🎯 Core Concepts

| Concept                            | Description                                                                                           |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Non-Linear FEM**                 | Uses an *updated-Lagrangian* approach, recomputing element stiffness matrices $K_e$ at each timestep. |
| **Parameter Identification**       | Solves an inverse problem to identify $(E, \nu)$ from observed motion data.                           |
| **CasADi Symbolic Graph**          | Builds the entire simulation as a single symbolic expression $J(E,\nu)$.                              |
| **Automatic Differentiation (AD)** | CasADi provides efficient gradients via reverse-mode AD (“adjoint method”).                           |
| **Gradient-Based Optimization**    | The non-linear program is solved by **IPOPT** to find optimal material parameters.                    |

---

### 🚀 Pipeline Overview

The main script `optimize_material_time_series.py` executes the full workflow:

1. **Generate Ground Truth**
   Runs a numerical FEM simulation with known $(E_{true}, \nu_{true})$, saving `u_target_series.npy`.

2. **Build Symbolic Model**
   Recreates the same simulation in CasADi (`ca.SX`), leaving $E$, $\nu$ as symbolic variables.

3. **Define Objective Function**
   Builds cost function
   $$
   J = \sum_t ||u_{\text{sim}}(t) - u_{\text{target}}(t)||^2
   $$

4. **Optimize**
   Solves `minimize J(E, ν)` using CasADi + IPOPT to find $(E_{opt}, \nu_{opt})$.

5. **Generate Optimized Trajectory**
   Runs a final simulation using $(E_{opt}, \nu_{opt})$ and saves `u_series_opt.npy`.

6. **Visualize Results**
   Calls `plots_optimization.py` to display and compare the results interactively.

---

### ⚙️ Requirements

Install dependencies:

```bash
pip install numpy casadi matplotlib scipy
```

| Library        | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| **numpy**      | Numerical computations                                        |
| **casadi**     | Symbolic modeling, automatic differentiation, IPOPT interface |
| **matplotlib** | Visualization and animations                                  |
| **scipy**      | Optional numerical utilities                                  |

---

### 📈 How to Run

Simply run:

```bash
python optimize_material_time_series.py
```

This command will:

* ✅ Generate the ground truth dataset
* 🔍 Run full parameter optimization
* 🧾 Print true vs. identified parameters
* 💾 Save all `.npy` and `.txt` results
* 🎨 Launch the visualizer

**The visualizer displays:**

* Side-by-side 3D animation (Ground Truth vs. Optimized)
* Position error per node over time
* Final error bar chart
* 3D scatter of error distribution on the object

---

### 📁 Project Structure

```
.
├── optimize_material_time_series.py   # MAIN: Full optimization pipeline
├── plots_optimization.py               # VISUALIZER: Animations & analysis
│
├── nodes_ref.npy                       # (Generated) Rest positions of the mesh
├── tets.npy                            # (Generated) Element connectivity
├── u_target_series.npy                 # (Generated) "Ground truth" displacements
├── u_series_opt.npy                    # (Generated) Optimized displacements
└── optimized_params.txt                # (Generated) Final (E_opt, ν_opt)
```

---

### 🧠 Author Notes

This repository shows how **symbolic computation**, **automatic differentiation**, and **non-linear FEM** can combine into a compact yet powerful workflow for solving complex **inverse problems in mechanics**.

---

### 🧩 Keywords

`Finite Element Method (FEM)` • `CasADi` • `IPOPT` • `Automatic Differentiation` • `Inverse Problems` • `Material Parameter Identification`

---

⭐ If you find this useful, please consider **starring the repository** to support further development!

