#!/usr/bin/env python3
"""
visualize_fem_comparison.py

Utility script to visualize and analyze FEM simulations:
---------------------------------------------------------
- Plots side-by-side 3D animations of "ground truth" vs "optimized" results.
- Computes per-node position errors over time and plots diagnostic figures.

Expected inputs (NumPy .npz or .npy files):
    nodes_ref.npy          : (n_nodes, 3) reference node coordinates
    u_target_series.npy    : (n_steps, ndof) ground-truth displacements
    u_series_opt.npy       : (n_steps, ndof) optimized displacements
    tets.npy               : (n_elems, 4) tetrahedral connectivity list

Optionally, if you have multiple sets (e.g. different experiments),
you can just replace the loaded filenames or arguments.

Run:
    python visualize_fem_comparison.py

Author: ChatGPT (adapted for Paolo's FEM inverse problem) AHAAHHAHAHAHAHA
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # <-- ADD THIS LINE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================
# 🟦 1. Load data
# ==============================================================

# You can adapt these paths to your setup or pass via sys.argv
nodes_ref = np.load("nodes_ref.npy")          # shape (n_nodes, 3)
u_target_series = np.load("u_target_series.npy")  # shape (n_steps, ndof)
u_series_opt = np.load("u_series_opt.npy")        # shape (n_steps, ndof)
tets = np.load("tets.npy")                    # shape (n_elems, 4)

# Time-step size (only used for labeling in animation)
dt = 1e-4
n_steps = min(len(u_target_series), len(u_series_opt))
n_nodes = nodes_ref.shape[0]
ndof = 3 * n_nodes

print(f"Loaded {n_nodes} nodes, {n_steps} frames, {len(tets)} tetrahedra")

# ==============================================================
# 🟧 2. Animation setup
# ==============================================================

def draw_tetra(ax, nodes, tets):
    """Draw mesh edges for visualization."""
    for tet in tets:
        verts = nodes[np.array(tet)]
        for i in range(4):
            for j in range(i+1, 4):
                p1, p2 = verts[i], verts[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        'k-', lw=0.5, alpha=0.6)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def update(frame):
    """Animation callback: update both sides for given frame index."""
    ax1.cla(); ax2.cla()
    ax1.set_title(f"Ground truth  (t={frame*dt*1000:.1f} ms)")
    ax2.set_title(f"Optimized      (t={frame*dt*1000:.1f} ms)")

    u_tar = u_target_series[frame].reshape(n_nodes, 3)
    u_opt = u_series_opt[frame].reshape(n_nodes, 3)
    nodes_tar = nodes_ref + u_tar
    nodes_opt = nodes_ref + u_opt

    ax1.scatter(nodes_tar[:, 0], nodes_tar[:, 1], nodes_tar[:, 2], c='b', s=20)
    ax2.scatter(nodes_opt[:, 0], nodes_opt[:, 1], nodes_opt[:, 2], c='r', s=20)
    draw_tetra(ax1, nodes_tar, tets)
    draw_tetra(ax2, nodes_opt, tets)

    for ax in [ax1, ax2]:
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-2, 12)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=100)
plt.show()

# ==============================================================
# 🟩 3. Node-wise position error computation
# ==============================================================

# Compute Euclidean error per node and timestep
errors = np.zeros((n_steps, n_nodes))
for f in range(n_steps):
    u_tar = u_target_series[f].reshape(n_nodes, 3)
    u_opt = u_series_opt[f].reshape(n_nodes, 3)
    errors[f, :] = np.linalg.norm(u_tar - u_opt, axis=1)

# --------------------------------------------------------------
# Error vs time plot
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
time = np.arange(n_steps) * dt
for i in range(n_nodes):
    plt.plot(time, errors[:, i], label=f'Node {i}')
plt.xlabel('Time [s]')
plt.ylabel('Position error [m]')
plt.title('Node-wise position error over time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Final-step bar plot
# --------------------------------------------------------------
final_err = errors[-1, :]
plt.figure(figsize=(7, 4))
plt.bar(np.arange(n_nodes), final_err)
plt.xlabel('Node index')
plt.ylabel('Final position error [m]')
plt.title('Final position error per node')
plt.show()

# --------------------------------------------------------------
# 3D color-coded error map on final geometry
# --------------------------------------------------------------
plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
nodes_final = nodes_ref + u_target_series[-1].reshape(n_nodes, 3)
p = ax.scatter(nodes_final[:, 0], nodes_final[:, 1], nodes_final[:, 2],
               c=final_err, cmap='inferno', s=60)
for i in range(n_nodes):
    ax.text(nodes_final[i, 0], nodes_final[i, 1], nodes_final[i, 2],
            str(i), fontsize=8)
fig.colorbar(p, ax=ax, label='Error magnitude [m]')
ax.set_title('Final error per node (color = |u_opt - u_target|)')
plt.show()

print("Visualization complete ✅")
