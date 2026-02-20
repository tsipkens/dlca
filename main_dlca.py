
import numpy as np
import models

# %%
n_particles = np.random.randint(20, 51, 1)
for ii in range(len(n_particles)):
    pos, dsu, box_size = models.run(n_particles[ii], seed_density=0.0008, f_xyz=True)

