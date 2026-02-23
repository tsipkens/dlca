
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from collections import deque
from IPython.display import display, clear_output  # for live plot in Python
import time

from tqdm import tqdm


# ----- Discrete Set Union (DSU) Helper Class -----
# Used to make cluster operations more efficient. 
class DSU:
    def __init__(self, n):
        """
        Initializes Disjoint Set Union with 'n' individual elements.
        """
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
        self.agg_count = n  # track number of aggs on merge

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

    def union(self, i, j):
        """Union by rank: Attaches the smaller tree to the larger tree during merges."""
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Union by Rank logic
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
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



def pos2xyz(pos):
    return pos[:,0], pos[:,1], pos[:,2]

def timecode(length=7):
    """Converts a number to a base-36 string."""
    n = int(time.time())  # get current time
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    arr = []
    while n:
        n, rem = divmod(n, 36)
        arr.append(digits[rem])
    
    arr.reverse()  # reverse order of characters
    res = "".join(arr)
    
    # Pad with '0' to the left to ensure Windows sorts by string length correctly
    return res.zfill(length)

def write_xyz(pos, r, c=None, filename=f"outputs\\aggregate.xyz", comment="DLA Cluster", write_mode='w'):
    """
    Writes particle positions and radii to an XYZ file.

    
    Args:
        particles (list): List of objects with .pos (array-like) and .r (float)
        filename (str): The output file name.
        comment (str): Text for the second line of the file.
    """
    try:
        with open(filename, write_mode) as f:
            # 1. Write the number of atoms/particles
            f.write(f"{len(pos)}\n")
            
            # 2. Write the comment line
            f.write(f"{comment}\n")
            
            # 3. Write each particle: Element (C for Carbon/Default) X Y Z R
            for ii, p in enumerate(pos):
                # We use 'C' as a placeholder for the element name
                # Standard XYZ is: Atom X Y Z. We add R as an extra column.
                if c is not None:
                    f.write(f"{p[0]:12.6e} {p[1]:12.6e} {p[2]:12.6e} {r:12.6e} {c[ii]}\n")
                else:
                    f.write(f"{p[0]:12.6e} {p[1]:12.6e} {p[2]:12.6e} {r:12.6e}\n")
        
    except Exception as e:
        print(f"Error writing XYZ file: {e}")

def read_xyz(filename):
    """
    Reads a single-frame XYZ file.
    Returns: (N, 3) array of positions and a list of atom names.
    """
    with open(filename, 'r') as f:
        n_particles = int(f.readline())
        comment = f.readline()
        
    # Use loadtxt to skip the first two lines
    # usecols=(1, 2, 3) assumes column 0 is the atom name (e.g., 'C')
    read = np.loadtxt(filename, skiprows=2, usecols=(0, 1, 2, 3))
    pos = read[:, :3]
    radius = read[:, 3]
    
    return pos, radius


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




def init_live_plot(pos, c=None, ax=None):
    """
    For plotting a live 3D scatter plot of particle positions. 
    """
    if ax is None:
        ax = plt.gca()

    if c is None:
        c = np.ones(len(pos))

    limit = np.max(pos)
    limit_min = np.min(pos)
    data_range = limit - limit_min

    # It is crucial to set the limits before calculating the scale
    ax.set_xlim(limit_min, limit)
    ax.set_ylim(limit_min, limit)
    ax.set_zlim(limit_min, limit)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]

    scat = ax.scatter(x, y, z, s=60, c=c)

    return scat

def update_live_plot(fig, scat, pos, clusters):
    """Update the live plot."""
    scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
    scat.set_array(clusters)
    clear_output(wait=True)
    display(fig)
    plt.pause(0.001)


def get_ascii_2d(pos, box_size=None, res=(10, 25), left=0, color="\033[92m"):
    """
    Returns a 2D ASCII string projection (XY plane).
    res: (rows, cols)
    """

    # Shrink box to domain of particles for viz if not given.
    if box_size == None:
        pos = pos - np.min(pos, axis=0)  # wrap box up to min
        box_size = np.max(pos)  # get max remaining
        pos = pos + (box_size - np.max(pos, axis=0)) / 2  # center in remaining box
        
        # If small aggregate, scale res down so more representative.
        # Otherwise circles are very far apart in viz. 
        if box_size < np.max(res):
            res = 2 * np.floor(res / np.max(res) * box_size).astype(int)

    rows, cols = res
    grid = np.full((rows, cols), " ")
    
    # Scale XY positions to grid dimensions
    # pos[:, 0] is X, pos[:, 1] is Y
    x_indices = (pos[:, 0] / box_size * (cols - 1)).astype(int)
    y_indices = (res[0] - 1) - (pos[:, 1] / box_size * (rows - 1)).astype(int)
    
    # Fill grid (clipping to ensure indices stay in bounds)
    for x, y in zip(x_indices, y_indices):
        if 0 <= x < cols and 0 <= y < rows:
            if grid[y, x] == " ":
                grid[y, x] = "o"
            elif grid[y, x] == "o":
                grid[y, x] = "#"
            else:
                grid[y, x] = "@"
            
    # Join into a multiline string
    out = " "*left + "╭" + "-"*int(res[1]) + "╮\n"
    out = out + "\033[0m|\n".join([" "*left + "|" + color + "".join(row) for row in grid])
    out = out + "\033[0m|\n" + " "*left + "╰" + "-"*int(res[1]) + "╯"
    return out



def unwrap(pos, box_size, radius, origin=None):
    """
    Unwraps periodic boundary conditions for a connected cluster of particles.
    
    Parameters:
    -----------
    pos : ndarray (N, 3)
        Wrapped x, y, z coordinates.
    box_size : float or array-like
        The dimensions of the simulation box.
    radius : float
        Particle radius used to build r_threshold.
        (threshold usually 2 * RADIUS + small epsilon).
    origin : ndarray (1, 3)
        Origin on which to center particle after unwrapping.
        
    Returns:
    --------
    unwrapped_pos : ndarray (N, 3)
        Coordinates shifted out of the [0, box_size] range to be contiguous.
    """

    # Parse origin input. Used to center cluster below.
    if origin is None:
        origin = box_size/2  # default is center of the box

    r_threshold = 2.05 * radius  # threshold to be considered connected

    n_particles = len(pos)  # number of particles
    unwrapped_pos = np.copy(pos).astype(float) # initialize unwrapped positions
    
    # 1. Build Neighbor List (respecting PBC)
    # + Get all pairs within the connectivity threshold
    tree = cKDTree(pos, boxsize=box_size)
    adj_list = tree.query_ball_tree(tree, r=r_threshold)
    
    # 2. Track visited particles
    visited = np.zeros(n_particles, dtype=bool)
    
    # Iterate through all particles to handle multiple disconnected aggregates
    for seed_idx in range(n_particles):
        if visited[seed_idx]:  # already considered, so skip
            continue
            
        # BFS Queue: (current_particle_index)
        queue = deque([seed_idx])
        visited[seed_idx] = True
        
        while queue:
            curr = queue.popleft()
            
            for neighbor in adj_list[curr]:
                if not visited[neighbor]:
                    # 3. Calculate the periodic shift
                    # 'diff' is the vector from current to neighbor
                    diff = unwrapped_pos[neighbor] - unwrapped_pos[curr]
                    
                    # Find the nearest image shift
                    # If diff is > box/2, it means it wrapped; this pulls it back.
                    shift = np.round(diff / box_size) * box_size
                    unwrapped_pos[neighbor] -= shift
                    
                    visited[neighbor] = True
                    queue.append(neighbor)

    # 4. Recenter agg at specified origin.
    unwrapped_pos = unwrapped_pos - np.mean(unwrapped_pos, axis=0) + origin
    
    return unwrapped_pos


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



def com(pos, box_size):
    """
    Computes COM for equivalent particles under periodic boundary conditions.
    pos: (N, 3) array of XYZ coordinates
    box_length: (3,) array or float representing box dimensions
    """
    # 1. Scale coordinates to [0, 2*pi]
    theta = (pos / box_size) * 2 * np.pi
    
    # 2. Transform to unit circle (cos, sin)
    xi = np.cos(theta)
    zeta = np.sin(theta)
    
    # 3. Average the circle coordinates
    mean_xi = np.mean(xi, axis=0)
    mean_zeta = np.mean(zeta, axis=0)
    
    # 4. Map back to angle space [0, 2*pi]
    mean_theta = np.arctan2(-mean_zeta, -mean_xi) + np.pi
    
    # 5. Transform back to box coordinates
    com = (mean_theta / (2 * np.pi)) * box_size
    return com

def Rg(pos):
    """
    pos: (N, 3) numpy array of x, y, z centers
    """
    dists_sq = np.sum((pos - com(pos))**2, axis=1)  # square distance to CoM
    return np.sqrt(np.mean(dists_sq))  # root mean square distance


def projected_area(pos, radius=1, view=[1,0,0], n_samples=int(1e6)):
    """
    Computes projected area using optimized Monte Carlo integration.
    """
    v = np.asarray(view) / np.linalg.norm(view)
    z_axis = np.array([0, 0, 1])
    
    # 1. Faster Rotation check
    if not np.allclose(v, z_axis):
        # Using a more direct rotation calculation if possible
        rot_axis = np.cross(v, z_axis)
        norm = np.linalg.norm(rot_axis)
        if norm > 1e-9:
            rot_axis /= norm
            angle = np.arccos(np.clip(np.dot(v, z_axis), -1.0, 1.0))
            projected_pos = R.from_rotvec(rot_axis * angle).apply(pos)
        else: # Case where v is [0,0,-1]
            projected_pos = pos * np.array([1, 1, -1])
    else:
        projected_pos = pos

    xy = projected_pos[:, :2]

    # 2. Bounding Box
    mins = np.min(xy, axis=0) - radius
    maxs = np.max(xy, axis=0) + radius
    box_area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])

    # 3. Generate Random Samples
    samples = np.random.uniform(mins, maxs, (n_samples, 2))

    # 4. Optimized Hit Check
    # Instead of query_ball_point (which returns lists), use query (returns distances)
    tree = cKDTree(xy)
    dists, _ = tree.query(samples, k=1, workers=-1) # k=1 is the nearest neighbor
    
    # Hits are where the nearest neighbor is closer than radius
    num_hits = np.count_nonzero(dists <= radius)

    area = box_area * (num_hits / n_samples)
    da = 2 * np.sqrt(area / np.pi)

    return area, da


def random_view(n_proj):
    """
    Get random views for projection calculations.
    Getting views prior to calling projected area calculations can make calculations repeatable.
    """
    vec = np.random.standard_normal((n_proj, 3))  # random direction
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)  # normalize to unit vector

def ave_projected_area(pos, n_proj=200, radius=1, n_samples=int(1e5), views=None):
    """
    Wrapper for projected_area to average over directions.

    n_proj : int
        Number of direction to use when averaging
    n_samples : int
        Number of samples per view/projection.
        Total number of samples = n_proj * n_samples.
    """

    # If views not provided, compute new random ones. 
    # Giving a views input allows for repeatable/comparable calculations.
    if views is None:
        views = random_view(n_proj)
    
    # Initialize array to contain areas.
    areas = np.zeros(n_proj)
    
    for ii in tqdm(range(n_proj), desc="Projecting"):
        areas[ii], _ = projected_area(pos, radius, view=views[ii], n_samples=n_samples)
    
    mean_a = np.mean(areas)
    mean_da = 2 * np.sqrt(mean_a / np.pi)

    return mean_a, mean_da


def scale_agg(pos, radius=1, rho_100=510, zet=2.48, rho_m=1860, rho_gsd=1, da1=None):
    """
    Scale agg to lie on a specified mass-mobility relation. 
    NOTE: Projected area is taken as equal to the mobility. 

    Parameters:
    -----------
    pos : ndarray (N, 3)
        x, y, z coordinates
    radius : float
        Radius of monomers in the simulation.
    rho_gsd : float
        Geometric standard deviation (GSD) on rho_100 to randomly perturb from relation.
        Assumes lognormal conditional distribution about main relation. 
    """

    # If not provided, get projected area diameter for mobility using random views.
    if da1 is None:
        da1 = ave_projected_area(pos)[1]

    npp = len(pos)  # number of monomers

    # Randomly perturb rho_100 based on provided GSD.
    if rho_gsd > 1:
        rho_100 = rho_100 * np.exp(np.log(rho_gsd) * np.random.standard_normal(1))

    d100 = 100  # reference point for relation (assumed nm)

    # dimensionless combination of diameters with various powers. 
    diam_fact = d100**(zet-3) * (2 * radius)**3 / (da1**zet)

    scale = (diam_fact * npp * rho_m / rho_100) ** (1 / (zet - 3))  # dimensionless scale factor

    radius = scale * radius
    pos = scale * pos

    return pos, radius





