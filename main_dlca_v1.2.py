import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

import tools


# %%

# ---------- SIMULATION PARAMETERS --------------
# Ouput parameters. 
F_PLOT = False  # whether to generate live plot
F_XYZ = True  # whether to output XYZ files at intermediate points

# DLCA parameters
SEED_DENSITY = 0.0008  # in #/RADIUS^3
RADIUS = 1  # use standard radius of 1
DIFFUSION_DT = ((0.1*2)**2/2) * RADIUS ** 2  # diffusion coefficient * time step, particle moves ~ 0.1 * DIAM


# ----- Function to run simulation -----
def run(n_particles, seed_density):

    # Use inputs to set simulation parameters.
    box_size = np.ceil((n_particles / seed_density) ** (1/3))  # compute box size based on seed density
    steps = 2000 * n_particles  # scale total number of steps before break with number of particles
    id = tools.timecode()

    # Print input parameters. 
    print('-'*22 + f' SIMULATION ({id}) ' + '-'*22)
    print(f'              n_particles={n_particles}     box_size={int(box_size):2.3f}')
    # print('-'*66)
    print('\n')


    # --------------- INITIALIZE SIMULATION ---------------
    # Initialize positions randomly
    pos = tools.init_particles(n_particles, box_size, RADIUS)
    n_particles = len(pos)  # update n_particles in case the generator couldn't fit them all

    # Initialize DSU, used to track clustering/aggs. 
    # Initially each particle is its own agg. 
    dsu = tools.DSU(n_particles)  # set up DSU to handle agg operation
    aggs = dsu.flatten()  # extract cluster IDs from dsu

    # Write to XYZ.
    fn = f'outputs\\agg_{id}_run.xyz'
    tools.write_xyz(pos, RADIUS, c=aggs, filename=fn)

    # For live plot. 
    if F_PLOT:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scat = tools.init_live_plot(pos, c=aggs, ax=ax)

    # ---------------- MAIN TIME LOOP ---------------------
    for step in (pbar := tqdm(range(steps))):
        # 1. Sync IDs and get counts
        # Using bincount is much faster than np.unique for integer IDs.
        aggs = dsu.flatten()  # extract the agg that each particle is a member of
        counts = np.bincount(aggs)
        unique_aggs = np.where(counts > 0)[0]  # unique agg ids
        npp = counts[unique_aggs]

        # 2. Generate Brownian motion for each agg.
        # Modify the diffusion and time step based on the agg size. 
        # To do this, we modify the diffusion based on the number of monomers, assuming Df = 1.8.
        # NOTE: We then normalize by the minimum modifier. 
        # This effectively increases the time step when the aggregates get bigger, while maintaining the 0.1 * d distance.
        modifier = (npp / np.min(npp)) ** (-1/1.8)  # adjust the distance to travel based on aggregate size
        stds = np.sqrt(2 * DIFFUSION_DT * modifier)  # standard deviation of motion
        displace = np.random.normal(0, 1, (dsu.agg_count, 3)) * stds[:, np.newaxis]  # displacement for random motion

        # 3. Apply Movement
        # Efficient lookup table mapping agg_id -> displacement.
        # Lookup is harder on memory but otherwise is faster. 
        lookup = np.zeros((aggs.max() + 1, 3))
        lookup[unique_aggs] = displace
        pos = (pos + lookup[aggs]) % box_size  # update positions and apply periodic boundary conditions

        # 4. Build a KDTree for computing distances efficiently.
        tree = cKDTree(pos, boxsize=box_size)

        # 5. Consider pairs, backoff to prevent overlap, and then merge. 
        # Find all the pairs within a specified distance. 
        pairs = tree.query_pairs(r = 2 * RADIUS)
        if pairs:
            # Backoff particles to avoid overlap. 
            # Also avoid pairs continually appearing in later steps. 
            pos = tools.backoff(pos, RADIUS, box_size, pairs, aggs)

            # Merge clusters that have collided. 
            for ii, jj in pairs:
                dsu.union(ii, jj)
            
        # 6. UI Updates
        if step % 20 == 0 or dsu.agg_count == 1:
            pbar.set_postfix({
                '\033[92mAggs\033[0m': dsu.agg_count
            })

        # Outputs at specific intervals.
        if F_PLOT:
            if step % 100 == 0:
                tools.update_live_plot(fig, scat, pos, aggs)  # update live plot in Jupyter
        if F_XYZ:
            if step % 50 == 0 or dsu.agg_count == 1:
                tools.write_xyz(pos, RADIUS, c=dsu.flatten(), write_mode='a', filename=fn)  # write XYZ coordinates

        # Check finishing conditions and print corresponding text. 
        if dsu.agg_count == 1:
            break

    pos = tools.unwrap(pos, box_size)  # center agg in box before final write
    tools.write_xyz(pos, RADIUS, c=aggs, filename=f'outputs\\agg_{id}_final.xyz')  # write XYZ coordinates
    
    print('\n' + tools.get_ascii_2d(pos, left=18))
    print('\n' + '-'*66)

    return pos, dsu, box_size

n_particles = np.random.randint(20, 51, 1)
for ii in range(len(n_particles)):
    pos, dsu, box_size = run(n_particles[ii], SEED_DENSITY)



