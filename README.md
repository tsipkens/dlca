# DLCA

This repository contains relatively simple Python code to simulate diffusion-limited cluster aggregation (DLCA). 

Simulations are performed in radius units, where **1 length unit = 1 monomer radius**. Translation to a meaningful scale is described below. 


### Scaling to mass-mobility relations

We define the mass-mobility relation using the following expressions:

$$m = m_{100} (d_{A} / d_{100}) ^ {\zeta}$$

or, equivalently, 

$$\rho_{eff} = \rho_{100} (d_{A} / d_{100}) ^ {\zeta - 3},$$

where $d_A$ is the projected area diameter, $\zeta$ is the mass-mobility exponent, $d_{100}$ is a reference diameter typically taken as 100 nm, and $\rho_{100}$ is the effective density at $d_{100}$. 

To estimate the mobility diameter of an agglomerate, we equate the mass from the mass-mobility relation and that by summing the constituent monomers. From this, we obtain a scaling factor consistent with Nikookar et al. (2025). The scaling factor for the entire aggregate is determined by:

$$\frac{d_{A}}{d_{A,1}} = \left( \frac{n_{pp} \rho_m}{\rho_{100}} \frac{d_{100}^{\zeta-3} (2 r_{pp})^3}{d_{A,1}^{\zeta}} \right) ^ {1 / (\zeta - 3)}.$$

where $d_{A,1}$ is the estimated projected area diameter in radius units, $n_{pp}$ is the number of primary particles, and $\rho_m$ is the monomer density. 

To account for physical variability, dispersion is achieved by sampling $\rho_{100}$ from a **lognormal distribution**. 
