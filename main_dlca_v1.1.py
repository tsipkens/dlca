import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
import time
import sys

from IPython.display import clear_output, display

# from autils import autils

import tools


# %%

# ---------- SIMULATION PARAMETERS --------------
# Ouput parameters. 
F_PLOT = False  # whether to generate live plot
F_XYZ = True  # whether to output XYZ files at intermediate points

# DLCA parameters
SEED_DENSITY = 0.0008  # in #/RADIUS^3
RADIUS = 1  # use standard radius of 1
DIFFUSION_COEFF_DT = (0.1*2)**2/2 * RADIUS ** 2  # therefore particle moves ~ 0.1 * RADIUS


# ----- Discrete Set Union (DSU) Helper Class -----
class DSU:
    def __init__(self, n):
        self.parent = np.arange(n)
    
    def find(self, i):
        # Path compression: makes future lookups near O(1)
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True # a merge happened
        return False # already in same agg


# ----- Function to run simulation -----
def run(n_particles, seed_density):

    # Use inputs to set simulation parameters.
    box_size = np.ceil((n_particles / seed_density) ** (1/3))  # compute box size based on seed density
    steps = 1500 * n_particles  # scale total number of steps before break with number of particles
    id = tools.timecode()

    # Print input parameters. 
    print(f'-------------- SIMULATION ({id}) --------------')
    print(f'n_particles={n_particles}, box_size={box_size:2.3f}')


    # --------------- INITIALIZE SIMULATION ---------------
    # Initialize positions randomly
    pos = tools.init_particles(n_particles, box_size, RADIUS)
    n_particles = len(pos)  # update n_particles in case the generator couldn't fit them all

    # Track aggs: initially each particle is its own agg
    # aggs[i] = ID of the agg particle i belongs to
    aggs = np.arange(n_particles)

    # Write to XYZ.
    fn = f'outputs\\agg_{id}_run.xyz'
    tools.write_xyz(pos, RADIUS, c=aggs, filename=fn)

    # For live plot. 
    if F_PLOT:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scat = tools.init_live_plot(pos, c=aggs, ax=ax)


    # ---------------- MAIN TIME LOOP ---------------------
    dsu = DSU(n_particles)  # set up DSU

    for step in (pbar := tqdm(range(steps))):
        # Flatten the DSU structure to get current agg IDs for all particles
        # This is much faster than manual array masking
        for i in range(n_particles): dsu.find(i) 
        aggs = dsu.parent

        # Generate Brownian motion for each agg
        unique_aggs, Npp = np.unique(aggs, return_counts=True)
        n_aggs = len(unique_aggs)

        # Modify the diffusion and time step based on the agg size. 
        # To do this, we modify the diffusion based on the number of monomers, assuming Df = 1.8.
        # NOTE: We then normalize by the minimum modifier. 
        # This effectively increases the time step when the aggregates get bigger, while maintaining the 0.1 * d distance.
        modifier = (Npp / np.min(Npp)) ** (-1/1.8)

        # Get standard deviation of the displacement for each agg.
        stds = np.sqrt(2 * DIFFUSION_COEFF_DT * modifier)

        # Calculate random displacement simulating random diffusion.
        displace = np.random.normal(0, 1, (n_aggs, 3)) * stds[:, np.newaxis]

        # Map agg displacements back to individual agg IDs.
        # We create a lookup table where the index is the agg_id.
        # This is a bit harder on memory, but makes the code faster. 
        lookup = np.zeros((np.max(aggs) + 1, 3))
        lookup[unique_aggs] = displace

        # Move all of the particles in each agg. 
        pos += lookup[aggs]

        # Apply periodic boundary conditions to positions.
        pos = pos % box_size

        # Build a KDTree for computing distances efficiently.
        # Find all the pairs within a specified distance. 
        tree = cKDTree(pos, boxsize=box_size)
        pairs = tree.query_pairs(r = 2 * RADIUS)

        # Find overlapping pairs. 
        for ii, jj in pairs:
            # dsu.union handles "if aggs[ii] != aggs[jj]" check internally
            # Then assigns both particles/aggs to the same agg.
            dsu.union(ii, jj)
        
        # Update progress bar. 
        if step % 20 == 0:
            pbar.set_postfix({
                'Aggs': n_aggs,
                'Merged': f"{(1 - n_aggs/n_particles)*100:.1f}%"
            })

        # Outputs at specific intervals. 
        if F_PLOT:
            if step % 100 == 0:
                tools.update_live_plot(fig, scat, pos, aggs)  # update live plot in Jupyter
        if F_XYZ:
            if step % 50 == 0:
                tools.write_xyz(pos, RADIUS, c=aggs, write_mode='a', filename=fn)  # write XYZ coordinates

        # Check finishing conditions and print corresponding text. 
        if len(unique_aggs) == 1:
            print('Only one agg. remaining. Exiting loop.')
            pbar.set_postfix({  # print final update to progress bar
                'Aggs': 1,
                'Merged': f"{100:.1f}%"
            })
            break
        if step == steps - 1:
            print('Step limit reached.')

        # if step % 10 == 0 and step < 100:
        #     # 1. Clear the current cell output
        #     # wait=True prevents the screen from flickering by waiting 
        #     # until the next piece of content is ready to be shown.
        #     clear_output(wait=True)
            
        #     # 2. Print your 2D grid
        #     grid_str = get_ascii_2d(pos, box_size)
        #     print(f"\033[92m{grid_str}\033[0m")
            
        #     # 3. Manually trigger the tqdm update so it stays below the grid
        #     pbar.update(0)
            
        #     # 4. Your requested pause
        #     time.sleep(0.5)

    pos = tools.unwrap(pos, box_size)  # center agg in box before final write
    tools.write_xyz(pos, RADIUS, c=aggs, write_mode='a', filename=f'outputs\\agg_{id}_final.xyz')  # write XYZ coordinates
    
    print(tools.get_ascii_2d(pos, np.ceil((n_particles / SEED_DENSITY) ** (1/3))))
    
    print("\033[92mDONE\033[0m!")

    return pos

n_particles = np.random.randint(20, 31, 1)
for ii in range(len(n_particles)):
    pos = run(n_particles[ii], SEED_DENSITY)



