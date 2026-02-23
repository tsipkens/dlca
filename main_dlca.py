
import numpy as np
import models
import tools

# %%

pos = []

n_particles = np.random.randint(20, 31, 5)
for ii in range(len(n_particles)):
    pos_ii, dsu, box_size = models.run(n_particles[ii], seed_density=0.0008, f_xyz=2)

    pos.append(pos_ii)


# %%

pos_sc = []
da = []
dpp = []
rho = []

for ii in range(len(n_particles)):
    sc = tools.scale_agg(pos[ii])

    # Append outputs to respective lists.
    pos_sc.append(sc[0])
    dpp.append(sc[1] * 2)
    da.append(sc[2])

    rho.append(tools.rho_eff(pos[-1], da=da[-1], dpp=dpp[-1]))


# %%

img = tools.plot_projection(pos[-1], type='tem', tem_args={'noise_floor': 0.15})

