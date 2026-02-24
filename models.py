
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from tqdm import tqdm

import tools

# DLCA parameters
RADIUS = 1  # use standard radius of 1
DIFFUSION_DT = ((0.1*2)**2/2) * RADIUS ** 2  # diffusion coefficient * time step, particle moves ~ 0.1 * DIAM


# ----- Discrete Set Union (DSU) Helper Class -----
# Used to make cluster operations more efficient. 
class DSU:
    def __init__(self, n):
        """
        Initializes Disjoint Set Union with 'n' individual elements.
        """
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
        self.age = np.zeros((n, 2)).astype(int)  # stores the time when node i was connected to its parent
        self.agg_count = n  # track number of aggs/clusters on merge

    def find(self, i):
        """
        Iterative find with path compression to prevent RecursionError.
        Recursively traces up trees to the root particle.
        """
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        
        # Path compression: Short-circuit the path to the root
        while self.parent[i] != root:
            next_node = self.parent[i]
            self.parent[i] = root
            i = next_node
        return root

    def union(self, i, j, step=0):
        """Union by rank: Attaches the smaller tree to the larger tree during merges."""
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Union by rank logic
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self.age[root_j] = [step, i]
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self.age[root_i] = [step, j]
            else:
                self.parent[root_j] = root_i
                self.age[root_j] = [step, i]
                self.rank[root_i] += 1
            
            # Decrement aggregate count whenever a merge actually happens
            self.agg_count -= 1
            return True
        return False

    def flatten(self):
        """
        Ensures all nodes point directly to their root. 
        Useful for syncing a 'aggs' array in one pass.
        """
        for i in range(len(self.parent)):
            self.find(i)
        return self.parent


# ------ Other helper functions ------
def backoff(pos, radius, box_size, pairs, aggs):
    """
    Back particle off of one another to avoid overlap.
    """
    backoff = np.zeros((aggs.max() + 1, 3))
    for ii, jj in pairs:
        agg_i = aggs[ii]

        # Compute distance vector.
        diff = pos[ii] - pos[jj]  # distance between particle centers
        diff = diff - box_size * np.round(diff / box_size)  # apply periodic boundary conditions
        dist = np.linalg.norm(diff)
        unit_vec = diff / dist  # unit vector spanning particle centers
        
        # Magnitude of correction and multiply by unit vector.
        correct = 2 * radius - dist + 1e-10  # extra 1e-10 add small amount of space
        backoff[agg_i] = correct * unit_vec  # backup along normal vector

    return (pos + backoff[aggs]) % box_size  # update positions and apply periodic boundary conditions

def init_particles(n, box, r):
    """
    Places each particle individually and checks for overlap.
    """
    positions = []
    while len(positions) < n:
        cand = np.random.uniform(0, box, 3)
        if len(positions) == 0:
            positions.append(cand)
        else:
            # Check PBC distances
            diff = np.array(positions) - cand
            diff -= box * np.round(diff / box) # minimum image convention
            dists = np.linalg.norm(diff, axis=1)
            if np.all(dists >= 2 * r):
                positions.append(cand)
    return np.array(positions)


# ----- Main function used to run simulation -----
def run(n_particles, seed_density, f_xyz=1, f_plot=False, output_folder='outputs'):
    """
    Run a single DLCA simulation.

    Parameters:
    -----------
    n_particles : int
        The initial number of particles to spawn.
    seed_density : float
        A number density used to calculate the simulation box size [#/RADIUS^3].
    f_xyz : int, default=1
        If 1, writes final coordinates to an .xyz file.
        If 2, writes .xyz trajectory file at regular intervals.
        Else, does not write .xyz file.
    f_plot : bool, default=False
        If True, initializes and updates a live 3D Matplotlib visualization.

    Returns:
    --------
    pos : ndarray
        The final (N, 3) array of particle positions.
    dsu : DSU
        The Disjoint Set Union object containing the final cluster structure.
    box_size : float
        The side length of the cubic simulation volume.
    """

    # Use inputs to set simulation parameters.
    box_size = np.ceil((n_particles / seed_density) ** (1/3))  # compute box size based on seed density
    steps = 2000 * n_particles  # scale total number of steps before break with number of particles
    id = tools.timecode()

    # Print input parameters. 
    print('-'*22 + f' SIMULATION ({id}) ' + '-'*22)
    print(f'                n_particles={n_particles}     box_size={int(box_size)}')
    # print('-'*66)
    print('\n')


    # --------------- INITIALIZE SIMULATION ---------------
    # Initialize positions randomly
    pos = init_particles(n_particles, box_size, RADIUS)
    n_particles = len(pos)  # update n_particles in case the generator couldn't fit them all

    # Initialize DSU, used to track clustering/aggs. 
    # Initially each particle is its own agg. 
    dsu = DSU(n_particles)  # set up DSU to handle agg operation
    aggs = dsu.flatten()  # extract cluster IDs from dsu

    # Write to XYZ.
    fn = f'{output_folder}\\agg_{id}_run.xyz'
    if f_xyz == 2:
        tools.write_xyz(pos, RADIUS, c=aggs, filename=fn, comment=f'box_size={box_size}')

    # For live plot. 
    if f_plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scat = tools.init_live_plot(pos, c=aggs, ax=ax)

    # ---------------- MAIN TIME LOOP ---------------------
    color = "\033[31m"  # printing color afterwards, red if not converged
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

        # 4. Build a KDTree for computing pairwise distances efficiently.
        # As per below, if particles are far apart, their distance is not queried as a pair. 
        tree = cKDTree(pos, boxsize=box_size)

        # 5. Consider pairs, backoff to prevent overlap, and then merge. 
        # Find all the pairs within a specified distance. 
        pairs = tree.query_pairs(r = 2 * RADIUS)
        if pairs:
            # Backoff particles to avoid overlap. 
            # Also avoid pairs continually appearing in later steps. 
            pos = backoff(pos, RADIUS, box_size, pairs, aggs)

            # Merge clusters that have collided. 
            for ii, jj in pairs:
                dsu.union(ii, jj, step)
            
        # 6. UI Updates
        if step % 20 == 0 or dsu.agg_count == 1:
            pbar.set_postfix({
                '\033[92mAggs\033[0m': dsu.agg_count
            })

        # Outputs at specific intervals.
        if f_plot:
            if step % 100 == 0:
                tools.update_live_plot(fig, scat, pos, aggs)  # update live plot in Jupyter
        if f_xyz == 2:
            if step % 50 == 0 or dsu.agg_count == 1:
                tools.write_xyz(pos, RADIUS, c=dsu.flatten(), write_mode='a', filename=fn, comment=f'box_size={box_size}')  # write XYZ coordinates

        # Check finishing conditions and print corresponding text. 
        if dsu.agg_count == 1:
            color = "\033[92m"
            break

    pos = tools.unwrap(pos, box_size, RADIUS)  # center agg in box before final write
    if f_xyz >= 1:
        tools.write_xyz(pos, RADIUS, c=dsu.age[:,0], filename=f'{output_folder}\\agg_{id}_final.xyz', comment=f'box_size={box_size}')  # write XYZ coordinates
    
    print('\n' + tools.get_ascii_2d(pos, left=18, color=color))
    print('\n' + '-'*66 + '\n')

    return pos, dsu, box_size