Dynamic FEM Material Parameter Identification with CasADi

<p align="center">
<img alt="Python" src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3.9%252B-blue%3Flogo%3Dpython%26logoColor%3Dwhite">
<img alt="CasADi" src="https://www.google.com/search?q=https://img.shields.io/badge/CasADi-yellow%3Flogo%3Dcasadi%26logoColor%3Dblack">
<img alt="IPOPT" src="https://www.google.com/search?q=https://img.shields.io/badge/IPOPT-brightgreen%3Flogo%3Dipopt%26logoColor%3Dblack">
<img alt="NumPy" src="https://www.google.com/search?q=https://img.shields.io/badge/NumPy-white%3Flogo%3Dnumpy%26logoColor%3Dblue">
<img alt="Matplotlib" src="https://www.google.com/search?q=https://img.shields.io/badge/Matplotlib-grey%3Flogo%3Dmatplotlib%26logoColor%3Dwhite">
</p>

This project performs dynamic parameter identification for a 3D deformable object. It uses CasADi to build a complete, non-linear Finite Element (FEM) simulation symbolically, then uses gradient-based optimization to find the material properties (Young's Modulus $E$ and Poisson's Ratio $\nu$) that best match a target motion.

This is an inverse problem where we find the causes (material properties) from the effects (motion).

🎥 Demo

Pro-Tip: The visualization script plots_optimization.py shows a Matplotlib animation. To create a GIF like the one below, add ani.save('animation.gif', writer='pillow') (you may need pip install pillow) before plt.show().

🎯 Core Concepts

This repository demonstrates a powerful combination of techniques:

Non-Linear FEM: The simulation is not linear. It uses an "updated-Lagrangian" approach where the element stiffness ($K_e$) is recomputed at every timestep based on the current, deformed node positions.

Parameter Identification: It solves an inverse problem to identify unknown parameters ($E$, $\nu$) from observed data.

CasADi Symbolic Graph: The entire time-stepping simulation is built as a massive symbolic expression in CasADi. The final objective function J (total error) is a single symbolic graph that depends only on the initial guesses for $E$ and $\nu$.

Automatic Differentiation (AD): By building the simulation symbolically, we get the gradient of the total error with respect to the material properties for "free" using CasADi's reverse-mode AD (the "adjoint method").

Gradient-Based Optimization: The high-performance solver IPOPT is used to solve this non-linear program (NLP), finding the optimal $E$ and $\nu$ that minimize the error.

🚀 The Pipeline

The main script optimize_material_time_series.py executes a full, end-to-end pipeline:

Generate Ground Truth: A numerical (NumPy) simulation is run with known "true" properties ($E_{true}$, $\nu_{true}$). The resulting displacement history is saved as u_target_series.npy.

Build Symbolic Model: A second, identical simulation is built using CasADi's symbolic variables (ca.SX). The material properties $E$ and $\nu$ are left as free symbolic variables.

Define Objective Function: The script defines a cost function J as the sum of squared differences between the symbolic simulation's output and the "ground truth" data at each timestep.

Optimize: CasADi and IPOPT work together to solve the optimization problem minimize J(E, nu). This step finds the optimal $E_{opt}$ and $\nu_{opt}$ that make the simulation best match the target.

Generate Optimized Trajectory: A final numerical (NumPy) simulation is run using the found $E_{opt}$ and $\nu_{opt}$. Its history is saved as u_series_opt.npy.

Visualize: The script saves all necessary .npy files and automatically calls Paolo_Sofa/Paolo_codes/plots_optimization.py to launch the final visualization and analysis.

⚙️ Requirements

The project requires the following Python libraries. You can install them using pip:

pip install numpy casadi matplotlib scipy


numpy for all numerical computation.

casadi for the symbolic framework, AD, and IPOPT interface.

matplotlib for all plotting and animation.

scipy (while not used in this specific script, it's a standard part of this ecosystem).

📈 How to Run

No special setup is required. Just run the main optimization script:

python optimize_material_time_series.py


This single command will:

Generate the ground truth data.

Run the full optimization (this may take a few minutes).

Print the true vs. found parameters.

Save all data files (.npy and .txt).

Automatically open the visualization script, which shows four plots:

A side-by-side 3D animation of the Ground Truth vs. Optimized simulation.

A plot of position error for each node over time.

A bar chart of the final position error for each node.

A 3D scatter plot of the final error mapped onto the object's geometry.

📁 File Structure

.
├── optimize_material_time_series.py    # MAIN SCRIPT: Runs the full pipeline
├── plots_optimization.py               # VISUALIZER: Called by the main script
├── nodes_ref.npy                       # (Generated) Rest positions of the mesh
├── tets.npy                            # (Generated) Element connectivity
├── u_target_series.npy                 # (Generated) "Ground truth" displacement history
├── u_series_opt.npy                    # (Generated) Optimized displacement history
└── optimized_params.txt                # (Generated) Final E_opt and nu_opt values
