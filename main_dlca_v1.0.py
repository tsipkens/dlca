import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# from autils import autils

import tools


# %%

# Constant:
# KB = 1.380649e-23

# # Physical parameters
# T = 300  # [K]
# p = 1  # [atm]
# dm = 100  # mobility diameter of the primary particles [nm]
# D = KB * T * autils.cc(dm, T, p) / (3 * np.pi * autils.mu(T, p) * dm * 1e-9)

# # Transform units.
# D_transformed = D * Dt / (dm * 1e-9) ** 2


# UNITS BELOW:
# LENGTH = radius
# DT

# DLCA parameters
N_PARTICLES = 50
BOX_SIZE = 40.0
RADIUS = 1
DIFFUSION_COEFF_DT = RADIUS ** 2 / 6 * 0.1  # D * Dt = <RADIUS>^2 / 6 (prev. RADIUS = 0.5, DT = 0.01, DIFF = 0.5)
STEPS = 10000

# Initialize positions randomly
pos = tools.init_particles(N_PARTICLES, BOX_SIZE, RADIUS)
N_PARTICLES = len(pos)  # update N_PARTICLES in case the generator couldn't fit them all

# Track clusters: initially each particle is its own cluster
# clusters[i] = ID of the cluster particle i belongs to
clusters = np.arange(N_PARTICLES)

# Write to XYZ.
tools.write_xyz(pos, RADIUS, c=clusters)

# For live plot. 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scat = tools.init_live_plot(pos, c=clusters, ax=ax)

# -------- Main time integration loop --------
for step in range(STEPS):
    # Generate Brownian motion for each CLUSTER
    unique_clusters = np.unique(clusters)
    for c_id in unique_clusters:
        # Compute timestep by modifying D * Dt based on number of monomers. 
        Np = np.count_nonzero(clusters == c_id)  # number of monomers
        tau = DIFFUSION_COEFF_DT * Np ** (-1/1.78)

        # Calculate random displacement simulating random diffusion.
        displace = np.random.normal(0, np.sqrt(2 * tau), 3)

        # Move all particles belonging to this cluster.
        pos[clusters == c_id] += displace

    # Apply periodic boundary conditions to positions.
    pos = pos % BOX_SIZE

    # Build a KDTree for computing distances efficiently.
    # Find all the pairs within a specified distance. 
    tree = cKDTree(pos, boxsize=BOX_SIZE)
    pairs = tree.query_pairs(r = 2 * RADIUS)

    # Find overlapping pairs. 
    for ii, jj in pairs:
        if clusters[ii] != clusters[jj]:
            clusters[clusters == clusters[jj]] = clusters[ii]  # merge clusters, jj -> ii

    # Outputs at specific intervals. 
    if step % 100 == 0:
        tools.update_live_plot(fig, scat, pos, clusters)  # update live plot in Jupyter
    if step % 50 == 0:
        tools.write_xyz(pos, RADIUS, c=clusters, write_mode='a')  # write XYZ coordinates

    # Check finishing conditions and print corresponding text. 
    if len(np.unique(clusters)) == 2:
        print('Minimum number of clusters reached.')
        break
    if step == STEPS - 1:
        print('Step limit reached.')


print("DONE!")

