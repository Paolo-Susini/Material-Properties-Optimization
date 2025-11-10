"""
optimize_material_time_series.py

Dynamic FEM parameter identification using CasADi.
- Mesh: cube split into 5 tetrahedra (small test problem)
- Forward dynamics: updated-Lagrangian-ish approach where element B matrices
  are recomputed from current nodal positions each timestep (so geometry updates).
- Objective: minimize sum over timesteps of squared error between simulated
  nodal displacements and a provided target time-series {u_tar^n}.
- Decision variables: scalar Young's modulus E and Poisson ratio nu (global).
- Optimization via CasADi (IPOPT) using automatic adjoint (reverse-mode AD).
- Note: this is an educational toy example; performance is not optimized.
Requires: casadi, numpy, scipy, matplotlib
Install: pip install casadi numpy scipy matplotlib

Uses CasADi SX (symbolic) instead of MX to avoid
determinant/inverse compilation/evaluation problems in the large symbolic graph.

Note: SX can be slower for very large graphs but is more robust for determinant/inv nodes.
"""

# --- Standard libraries ---
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# --- Libraries for calling the external visualization script ---
import subprocess
import sys

# =============================================================================
# 🟦 1. MESH, MODEL, AND PARAMETER DEFINITION
# =============================================================================

# ----------------------
# Mesh (cube subdivided into 5 tetrahedra)
# ----------------------
# Define the 8 nodes (vertices) of the cube in their resting positions.
nodes_ref = np.array([
    [-5.0, -5.0, 0.0],  # Node 0
    [5.0, -5.0, 0.0],   # Node 1
    [5.0, 5.0, 0.0],    # Node 2
    [-5.0, 5.0, 0.0],   # Node 3
    [-5.0, -5.0, 10.0], # Node 4
    [5.0, -5.0, 10.0],  # Node 5
    [5.0, 5.0, 10.0],   # Node 6
    [-5.0, 5.0, 10.0]   # Node 7
], dtype=float)

# Define the 5 tetrahedral elements by listing their 4 node indices.
tets = [
    [0,2,7,5],
    [1,0,2,5],
    [3,0,2,7],
    [4,0,5,7],
    [6,2,5,7]
]
# Calculate total number of nodes and Degrees of Freedom (DOFs).
# Each node has 3 DOFs (x, y, z).
n_nodes = nodes_ref.shape[0]
ndof = 3 * n_nodes

# ----------------------
# Boundary Conditions (BCs)
# ----------------------
# Define which nodes are fixed (the 4 on the bottom face).
fixed_nodes = [0,1,2,3]
# Convert the list of fixed *nodes* into a list of fixed *DOFs*.
fixed_dofs = []
for n in fixed_nodes:
    # For each node index, add its corresponding x, y, and z DOF indices.
    fixed_dofs += [3*n, 3*n+1, 3*n+2]
fixed_dofs = np.array(fixed_dofs, dtype=int)
# Get the list of "free" DOFs (all DOFs *except* the fixed ones).
free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs) # here I find the set difference between two arrays

# ----------------------
# Simulation / model parameters (base)
# ----------------------
total_mass = 1.0
g = np.array([0.0, -9.81, 0.0]) # Gravity acceleration vector (pointing -Y)
# Proportional (Rayleigh) damping: C = alpha * M
alpha_damping = 0.0  # viscous damping coefficient

# ----------------------
# Time integration parameters
# ----------------------
dt = 1e-4        # Time step size (in seconds)
n_steps = 300    # Total number of steps to simulate
record_every = 1 # How often to save a frame (1 = save every frame)

# ----------------------
# External Forces
# ----------------------
# Apply a constant external force (in addition to gravity).
force_node = 6         # Apply force to node 6
force_magnitude = -15000.0 # Force value in Newtons
F_point = np.zeros(ndof)
# Apply the force to the Y-direction (DOF index 3*n + 1) of the node.
F_point[3*force_node + 1] = force_magnitude

# =============================================================================
# 🟦 2. UTILITY FUNCTIONS AND PRECOMPUTATION
# =============================================================================

# ----------------------
# Utilities
# ----------------------
def compute_B_and_volume_numpy(tet_coords):
    """
    Computes the strain-displacement matrix (B) and volume (V) for a
    single linear 4-node tetrahedron using NumPy.
    
    This is the core of the linear FEM element calculation.
    """
    # Create the 4x4 coordinate matrix [1, x, y, z]
    M = np.ones((4,4))
    M[:,1:] = tet_coords
    
    # Calculate volume: V = det(M) / 6
    detM = np.linalg.det(M)
    vol = detM / 6.0
    
    # Invert the matrix. The rows of Minv contain the coefficients
    # (a, b, c, d) of the linear shape functions N(x,y,z) = a + bx + cy + dz.
    Minv = np.linalg.inv(M)
    
    # Initialize the 6x12 strain-displacement matrix B
    B = np.zeros((6,12))
    for i in range(4): # Loop over the 4 nodes
        # Get shape function derivatives for node i:
        b_i = Minv[1,i] # d(Ni)/dx
        c_i = Minv[2,i] # d(Ni)/dy
        d_i = Minv[3,i] # d(Ni)/dz
        
        # Get the starting column in B for this node's DOFs
        col = 3*i
        
        # Populate the B matrix according to the strain definition
        # (e.g., epsilon_xx = d(u)/dx = b_i * u_i)
        B[0, col+0] = b_i
        B[1, col+1] = c_i
        B[2, col+2] = d_i
        # (e.g., gamma_xy = d(u)/dy + d(v)/dx = c_i * u_i + b_i * v_i)
        # Note: Your script uses a different (but valid) shear definition
        # B[3] -> gamma_yz = d(v)/dz + d(w)/dy = d_i*v_i + c_i*w_i
        B[3, col+1] = d_i 
        B[3, col+2] = c_i
        # B[4] -> gamma_xz = d(u)/dz + d(w)/dx = d_i*u_i + b_i*w_i
        B[4, col+0] = d_i
        B[4, col+2] = b_i
        # B[5] -> gamma_xy = d(u)/dy + d(v)/dx = c_i*u_i + b_i*v_i
        B[5, col+0] = c_i
        B[5, col+1] = b_i
        
    return B, abs(vol)

# ----------------------
# Precompute DOF mapping
# ----------------------
# Create a list `dof_map` where `dof_map[e]` is a 12-element list
# containing the global DOF indices for element `e`.
# This is used for "assembling" the global K matrix.
dof_map = []
for tet in tets:
    dofs = []
    for n in tet:
        dofs += [3*n, 3*n+1, 3*n+2]
    dof_map.append(dofs)

# ----------------------
# Precompute mass and force properties
# ----------------------
# Calculate reference volumes, density, and lumped mass matrix.
B_ref_list = []
V_ref_list = []
elem_masses = []
total_vol = 0.0
# Loop over elements in their *rest state*
for tet in tets:
    coords = nodes_ref[np.array(tet)]
    _B, vol = compute_B_and_volume_numpy(coords)
    B_ref_list.append(_B) # Store the B matrix at rest
    V_ref_list.append(vol) # Store the volume at rest
    total_vol += vol

# Calculate density (rho)
rho = total_mass / total_vol
# Calculate mass for each element
for vol in V_ref_list:
    elem_masses.append(rho * vol)

# "Lump" the mass: Distribute each element's mass to its 4 nodes
node_mass = np.zeros(n_nodes)
for me, tet in zip(elem_masses, tets):
    per_node_mass = me / 4.0 # 1/4 of element mass to each node
    for n in tet:
        node_mass[n] += per_node_mass # Add to the node's total mass

# Build the global diagonal (lumped) mass matrix M
M_lumped = np.zeros((ndof, ndof))
for i in range(n_nodes):
    for d in range(3):
        M_lumped[3*i + d, 3*i + d] = node_mass[i]

# Precompute the inverse of M as a diagonal vector (faster for a=M_inv*F)
diag = np.diag(M_lumped)
M_inv_diag = np.zeros(ndof)
nonzero = diag > 0 # Avoid division by zero (for fixed/massless nodes)
M_inv_diag[nonzero] = 1.0 / diag[nonzero]

# Build the global damping matrix C (C = alpha * M)
C_lumped = alpha_damping * M_lumped

# ----------------------
# Precompute external gravity force vector
# ----------------------
# Distribute gravity force (F_g = m*g) in the same way as mass
F_gravity = np.zeros(ndof)
for me, tet in zip(elem_masses, tets):
    f_e = (me * g) / 4.0 # 1/4 of element's gravity force to each node
    for i, n in enumerate(tet):
        F_gravity[3*n:3*n+3] += f_e

# Combine gravity and the concentrated point force into one constant force vector
F_ext_const_np = F_gravity + F_point

# =============================================================================
# 🟦 3. GROUND TRUTH GENERATION (NUMPY)
# =============================================================================
# This section runs a simulation with *known* material properties
# to create the "target" or "ground truth" data. The goal of the
# optimization is to recover these known properties.

E_true = 4e3  # The "true" Young's Modulus we want to find
nu_true = 0.45 # The "true" Poisson's Ratio we want to find
print("Creating synthetic target trajectory with E_true, nu_true =", E_true, nu_true)

# Initialize lists to store the time series of displacements
u_target_series = []
# Initialize state vectors: u (displacement), v (velocity)
u_tmp = np.zeros(ndof)
v_tmp = np.zeros(ndof)

# --- Start the Numerical Simulation Loop ---
for step in range(n_steps):
    # This is the "Updated Lagrangian" / Corotational part.
    # We *re-calculate the global stiffness K* at *every* timestep
    # based on the *current* deformed state.
    K = np.zeros((ndof, ndof))
    
    # Calculate Lame parameters (mu, lambda) and D matrix from E, nu
    mu = E_true / (2.0*(1.0+nu_true))
    lam = (E_true*nu_true) / ((1.0+nu_true)*(1.0-2.0*nu_true))
    D_true = np.zeros((6,6))
    D_true[0:3,0:3] = lam # Off-diagonal part
    for i in range(3): 
        D_true[i,i] += 2.0*mu       # Diagonal part
    D_true[3,3] = mu; D_true[4,4] = mu; D_true[5,5] = mu # Shear part

    # Loop over all 5 elements to build the global K
    for e_idx, tet in enumerate(tets):
        # Get the CURRENT deformed coordinates of the 4 nodes
        coords = np.zeros((4,3))
        for i, n in enumerate(tet):
            coords[i,:] = nodes_ref[n,:] + u_tmp[3*n:3*n+3]
        
        # Re-compute B matrix and Volume from the *deformed* state
        B_cur, vol = compute_B_and_volume_numpy(coords)
        
        # Compute element stiffness k_e = V * B^T * D * B
        Ke = vol * (B_cur.T @ D_true @ B_cur)
        
        # Assemble Ke into the global K
        dofs = dof_map[e_idx]
        for a in range(12):
            for b in range(12):
                K[dofs[a], dofs[b]] += Ke[a,b]

    # --- Solve Equation of Motion (Explicit) ---
    # F_total = F_ext - F_int - F_damp
    # M*a = F_total
    
    # Calculate internal force (F_int = K(u) * u)
    f_int = K @ u_tmp
    # Calculate damping force (F_damp = C * v)
    f_damp = C_lumped @ v_tmp
    # Calculate residual (total) force
    res = F_ext_const_np - f_int - f_damp
    
    # Calculate acceleration (a = M_inv * F_total)
    a_tmp = M_inv_diag * res
    # Apply BCs: zero out acceleration for fixed nodes
    a_tmp[fixed_dofs] = 0.0
    
    # Time integration (Semi-Implicit Euler)
    # v_new = v_old + a * dt
    v_tmp = v_tmp + a_tmp * dt
    # Apply BCs: zero out velocity for fixed nodes
    v_tmp[fixed_dofs] = 0.0
    # u_new = u_old + v_new * dt
    u_tmp = u_tmp + v_tmp * dt
    # Apply BCs: zero out displacement for fixed nodes
    u_tmp[fixed_dofs] = 0.0
    
    # Save the displacement for this frame
    if step % record_every == 0:
        u_target_series.append(u_tmp.copy())

# Convert the list of results into a single (n_record, ndof) NumPy array
u_target_series = np.array(u_target_series)
n_record = u_target_series.shape[0]
print("Target series generated: n_record =", n_record)

# =============================================================================
# 🟦 4. CASADI SYMBOLIC MODEL DEFINITION
# =============================================================================
# This section re-builds the *entire* simulation loop from above,
# but using CasADi's symbolic variables (SX) instead of NumPy.
# The goal is to create a single, giant symbolic expression `J`
# that represents the *total error* as a function of `E` and `nu`.

# ----------------------
# Define Symbolic Optimization Variables
# ----------------------
# `E` and `nu` are no longer numbers, but symbolic placeholders.
E_sym = ca.SX.sym('E')
nu_sym = ca.SX.sym('nu')

# ----------------------
# Build Symbolic D Matrix
# ----------------------
# Build the D matrix *symbolically* in terms of E_sym and nu_sym.
mu_sym = E_sym / (2*(1+nu_sym))
lam_sym = (E_sym*nu_sym) / ((1+nu_sym)*(1-2*nu_sym))
D_sym = ca.SX.zeros(6,6)
for i in range(3):
    for j in range(3):
        D_sym[i,j] = lam_sym
for i in range(3):
    D_sym[i,i] = D_sym[i,i] + 2*mu_sym
D_sym[3,3] = mu_sym; D_sym[4,4] = mu_sym; D_sym[5,5] = mu_sym

# ----------------------
# Convert NumPy Constants to CasADi format
# ----------------------
# CasADi's symbolic expressions must interact with its own "DM" type.
nodes_ref_dm = ca.DM(nodes_ref)
F_ext_dm = ca.DM(F_ext_const_np)
M_inv_diag_dm = ca.DM(M_inv_diag)
C_dm = ca.DM(C_lumped)
# Convert array of indices to a plain list for CasADi compatibility
fixed_dofs_list = list(map(int, fixed_dofs.tolist()))
dof_map_list = dof_map

# ----------------------
# Define Symbolic Simulation State
# ----------------------
# `u` and `v` are now symbolic vectors representing the *state* of the system.
u = ca.SX.zeros(ndof)
v = ca.SX.zeros(ndof)

# ----------------------
# Define Symbolic Objective Function
# ----------------------
# `J` is the symbolic cost/error, initialized to zero.
J = ca.SX(0)
record_index = 0

# --- Start the SYMBOLIC Simulation Loop ---
# This loop *builds a computational graph*. It does not run a simulation.
print("Building symbolic graph in CasADi...")
for step in range(n_steps):
    
    # Re-build K(u) symbolically, just like in the NumPy loop
    K_sym = ca.SX.zeros(ndof, ndof)
    for e_idx, tet in enumerate(tets):
        # Get symbolic deformed coordinates (depends on state `u`)
        coords = ca.SX.zeros(4,3)
        for i_local, n in enumerate(tet):
            coords[i_local,0] = nodes_ref_dm[n,0] + u[3*n + 0]
            coords[i_local,1] = nodes_ref_dm[n,1] + u[3*n + 1]
            coords[i_local,2] = nodes_ref_dm[n,2] + u[3*n + 2]
        
        # Build symbolic M matrix [1, x, y, z]
        Mmat = ca.SX.ones(4,4)
        Mmat[:,1] = coords[:,0]
        Mmat[:,2] = coords[:,1]
        Mmat[:,3] = coords[:,2]
        
        # Compute symbolic inverse and determinant
        Minv = ca.inv(Mmat)
        vol_e = ca.det(Mmat) / 6.0
        vol_e = ca.fabs(vol_e) # Ensure positive volume
        
        # Compute symbolic B matrix
        B_e = ca.SX.zeros(6,12)
        for i_local in range(4):
            b_i = Minv[1, i_local]
            c_i = Minv[2, i_local]
            d_i = Minv[3, i_local]
            col = 3*i_local
            B_e[0, col+0] = b_i
            B_e[1, col+1] = c_i
            B_e[2, col+2] = d_i
            B_e[3, col+1] = d_i
            B_e[3, col+2] = c_i
            B_e[4, col+0] = d_i
            B_e[4, col+2] = b_i
            B_e[5, col+0] = c_i
            B_e[5, col+1] = b_i

        # Compute symbolic element stiffness Ke
        # Ke is now a function of `u` (from B/vol) AND `E_sym, nu_sym` (from D_sym)
        Ke = vol_e * (B_e.T @ D_sym @ B_e)
        
        # Symbolically assemble Ke into K_sym
        dofs = dof_map_list[e_idx]
        for a in range(12):
            for b in range(12):
                K_sym[dofs[a], dofs[b]] = K_sym[dofs[a], dofs[b]] + Ke[a,b]
    
    # --- Symbolic Equation of Motion ---
    # These lines *define the relationships* in the graph.
    f_int = ca.mtimes(K_sym, u) # Use mtimes for symbolic matrix-vector product
    f_damp = C_dm @ v
    res = F_ext_dm - f_int - f_damp
    
    a_sym = M_inv_diag_dm * res
    for fd in fixed_dofs_list:
        a_sym[fd] = 0 # Symbolically set fixed DOFs' acceleration to 0
    
    # --- Symbolic Time Integration ---
    # `u` and `v` are overwritten with new *symbolic expressions*
    # that represent the state at the *end* of the step.
    v = v + a_sym * dt
    for fd in fixed_dofs_list:
        v[fd] = 0
    u = u + v * dt
    for fd in fixed_dofs_list:
        u[fd] = 0
    
    # --- Update Symbolic Objective Function `J` ---
    # This is the "cost" part of the optimization
    if step % record_every == 0:
        # Load the *numerical* ground truth data for this frame
        u_tar_vec = ca.DM(u_target_series[record_index])
        # Calculate the error (symbolic `u` vs. numerical `u_tar_vec`)
        diff = u - u_tar_vec
        # Add the squared error to the total cost `J`
        J = J + 0.5 * ca.dot(diff, diff)
        record_index += 1
# --- End of Symbolic Loop ---

# At this point, `J` is one giant symbolic expression.
# `J`'s only free variables are `E_sym` and `nu_sym`.
# `u` is also a giant symbolic expression for the *final* displacement.
print("Symbolic graph built.")

# Create a compiled CasADi Function. (Not used for optimization, but good for testing).
# This function maps (E, nu) -> (Total Error, Final Displacement)
f_forward = ca.Function('forward', [E_sym, nu_sym], [J, u])

# =============================================================================
# 🟦 5. OPTIMIZATION SETUP AND EXECUTION
# =============================================================================

# ----------------------
# Reparameterization (Constrained Optimization)
# ----------------------
# We need to ensure E > 0 and 0 < nu < 0.5.
# We do this by optimizing over *transformed* variables.
logE = ca.SX.sym('logE')   # Unconstrained variable
t_nu = ca.SX.sym('t_nu')   # Unconstrained variable

# E = exp(logE) -> This guarantees E is always positive.
E_var = ca.exp(logE)
# nu = 0.49 * tanh(t_nu) -> tanh maps (-inf, inf) to (-1, 1),
# so nu is mapped to (-0.49, 0.49), satisfying the physics constraint.
nu_var = 0.49 * ca.tanh(t_nu)

# Substitute the original E_sym/nu_sym in `J` with our new functions.
# J now becomes a function of `logE` and `t_nu`.
J_sub = ca.substitute(ca.substitute(J, E_sym, E_var), nu_sym, nu_var)
u_sub = ca.substitute(ca.substitute(u, E_sym, E_var), nu_sym, nu_var)

# Create a new function for the *optimizer* to call.
f_J = ca.Function('Jfun', [logE, t_nu], [J_sub, u_sub])

# ----------------------
# Setup NLP Solver (IPOPT)
# ----------------------
# `w` is the vector of *decision variables* we are solving for.
w = ca.vertcat(logE, t_nu)
# `nlp` defines the Non-Linear Program:
# 'x' are the decision variables, 'f' is the objective function to minimize.
nlp = {'x': w, 'f': J_sub}
# Configure the solver (IPOPT)
opts = {'ipopt.print_level': 5, 'print_time': False, 'ipopt.max_iter': 200}
# Create the solver instance
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# ----------------------
# Run the Optimization
# ----------------------
# Provide an initial guess for the optimizer
# Guess E=5e4 -> logE = log(5e4)
# Guess nu=0.0 -> t_nu = atanh(0/0.49) = 0
x0 = [np.log(5e4), 0.0] 
print("Starting optimization...")

# --- THIS IS THE MAGIC LINE ---
# The `solver` call does all the work:
# 1. It calls `f_J` to get the error `J`.
# 2. CasADi *automatically* computes the gradient of `J` w.r.t. `w`
#    using Automatic Differentiation (backpropagation through time).
# 3. IPOPT uses this gradient to take a step towards the minimum.
# 4. Repeat until converged.
sol = solver(x0=x0)

# ----------------------
# Extract Results
# ----------------------
w_opt = sol['x'].full().flatten() # Get the optimized `w` vector
# Back-transform the results to get physical E and nu
logE_opt = float(w_opt[0]); t_nu_opt = float(w_opt[1])
E_opt = float(np.exp(logE_opt))
nu_opt = float(0.49 * np.tanh(t_nu_opt))

print(f"Optimization finished. E_opt = {E_opt:.6e}, nu_opt = {nu_opt:.6e}")

# Run the function one last time with the optimal values
J_final, u_final = f_J(logE_opt, t_nu_opt)
J_final = float(J_final)
u_final = np.array(u_final).flatten()
print("Final objective J =", J_final)

# ----------------------
# Print Diagnostics
# ----------------------
u_num = forward_numpy_updated = None  # Clear/initialize helper variable

print("Selected node displacement norms (target final vs optimized final):")
for n in [4,5,6,7]: # Check displacement of the free top nodes
    tnorm = np.linalg.norm(u_target_series[-1, 3*n:3*n+3])
    onorm = np.linalg.norm(u_final[3*n:3*n+3])
    print(f" Node {n}: target_final ||u||={tnorm:.6e}, optimized_final ||u||={onorm:.6e}")


# =============================================================================
# 🟦 6. VISUALIZATION AND FINAL OUTPUT
# =============================================================================

# ----------------------
# Helper function (copy of the NumPy simulator)
# ----------------------
# This is used to re-run the *entire* simulation history
# using the newly found E_opt and nu_opt.
def forward_numpy_updated(E_val, nu_val, n_steps_local=n_steps, dt_local=dt):
    """
    Runs a full numerical simulation with the given E and nu.
    This is a pure NumPy copy of the Ground Truth generator.
    """
    mu = E_val / (2.0*(1.0+nu_val))
    lam = (E_val*nu_val) / ((1.0+nu_val)*(1.0-2.0*nu_val))
    D = np.zeros((6,6))
    D[0:3,0:3] = lam
    for i in range(3): D[i,i] += 2.0*mu
    D[3,3]=mu; D[4,4]=mu; D[5,5]=mu
    u = np.zeros(ndof)
    v = np.zeros(ndof)
    u_series = []
    for step in range(n_steps_local):
        K = np.zeros((ndof, ndof))
        for e_idx, tet in enumerate(tets):
            coords = np.zeros((4,3))
            for i, n in enumerate(tet):
                coords[i,:] = nodes_ref[n,:] + u[3*n:3*n+3]
            B, vol = compute_B_and_volume_numpy(coords)
            Ke = vol * (B.T @ D @ B)
            dofs = dof_map[e_idx]
            for a in range(12):
                for b in range(12):
                    K[dofs[a], dofs[b]] += Ke[a,b]
        f_int = K @ u
        f_damp = C_lumped @ v
        res = F_ext_const_np - f_int - f_damp
        a = M_inv_diag * res
        a[fixed_dofs] = 0
        v = v + a * dt_local
        v[fixed_dofs] = 0
        u = u + v * dt_local
        u[fixed_dofs] = 0
        u_series.append(u.copy())
    return np.array(u_series)

# ----------------------
# Generate and Save Final Data
# ----------------------
# Run the numerical simulator one last time to get the full
# time series of the *optimized* parameters.
print("Computing optimized trajectory for visualization...")
u_series_opt = forward_numpy_updated(E_opt, nu_opt, n_steps_local=n_steps, dt_local=dt)


# Write the final optimized parameters to a text file
with open('optimized_params.txt', 'w') as f:
    f.write(f"E_opt = {E_opt}\nnu_opt = {nu_opt}\nJ_final = {J_final}\n")

# Save all 4 .npy files required by the visualization script
np.save("nodes_ref.npy", nodes_ref)
np.save("tets.npy", np.array(tets))
np.save("u_target_series.npy", u_target_series) # The "ground truth"
np.save("u_series_opt.npy", u_series_opt)      # The "optimized" result

print("Script finished. Optimized parameters written to optimized_params.txt")
print("All .npy files for visualization have been saved.")

# ----------------------
# Call External Visualization Script
# ----------------------
try:
    # `sys.executable` finds the path to the currently running Python interpreter
    # This ensures the subprocess uses the *same environment* (e.g., venv)
    command_list = [sys.executable, "Paolo_Sofa/Paolo_codes/plots_optimization.py"]
    
    # Run the external script and wait for it to complete.
    # `check=True` will raise an error if the script fails.
    result = subprocess.run(command_list, check=True)
    
    print("--- The other script has finished ---")

except subprocess.CalledProcessError as e:
    # This catches errors *from* the visualization script
    print(f"The visualization script failed with error: {e}")
except FileNotFoundError:
    # This catches the error if the .py file doesn't exist
    print("Error: 'plots_optimization.py' not found.")