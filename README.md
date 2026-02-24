# DLCA

This repository contains relatively simple Python code to simulate diffusion-limited cluster aggregation (DLCA). 

Some complexity is added via a discrete set union (DSU) class. This class controls information surrounding clusters/agglomerates, making union of and referencing to cluster information far more efficient. A traditional array containing the cluster identifier for each monomer is accessible via the dsu.flatten() method. 

Otherwise, the models module is relatively straightforward, generating a series on monomers in a box before looping forward in time, randomly moving each aggregate in each timestep and then checking for collisions. 

The code otherwise uses fairly standard packages: numpy, scipy, matplotlib, and tqdm (for progress bars). 

Simulations are performed in radius units, where **1 length unit = 1 monomer radius**. Translation to a meaningful scale is described below. Size is otherwise unimportant, only influencing how far a cluster/agglomerate can move in a given timestep relative to other clusters (for monomers, they all move the same amount). A simplification is made to slow down the agglomerates/clusters as they get larger. 


### Scaling to mass-mobility relations

Agglomerates that come out of the DLCA simulations of this kind must be scaled to be representative of aggregates we see in real aerosol populations. This can be accomplished by scaling the aggregates to match mass-mobility relations (Nikookar et al., 2025). 

We define the mass-mobility relation using the following expressions:

$$m = m_{100} (d_{A} / d_{100}) ^ {\zeta}$$

or, equivalently, 

$$\rho_{eff} = \rho_{100} (d_{A} / d_{100}) ^ {\zeta - 3},$$

where $d_A$ is the projected area diameter, $\zeta$ is the mass-mobility exponent, $d_{100}$ is a reference diameter typically taken as 100 nm, and $\rho_{100}$ is the effective density at $d_{100}$. 

To estimate the mobility diameter of an agglomerate, we equate the mass from the mass-mobility relation and that by summing the constituent monomers. he scaling factor for the entire aggregate is determined by:

$$\frac{d_{A}}{d_{A,1}} = \left[ \frac{n_{pp} \rho_m}{\rho_{100}} \left( \frac{2 r_{pp}}{d_{100}} \right) ^3  \left( \frac{d_{A,1}}{d_{100}} \right) ^{-\zeta} \right] ^ {1 / (\zeta - 3)}.$$

where $d_{A,1}$ is the estimated projected area diameter in radius units, $n_{pp}$ is the number of primary particles, $r_{pp}$ is the monomer radius from the simulation (by default, $r_{pp} = 1$), and $\rho_m$ is the monomer density. 

To account for physical variability, dispersion is achieved by sampling $\rho_{100}$ from a **lognormal distribution**. 
