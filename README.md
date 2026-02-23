
# dlca

A relatively simple code to perform diffusion-limited cluster agglomeration (DLCA).


### Scaling aggregates to mass-mobility relation

Simulations occur in radius units (i.e., 1 length unit = 1 MONOMER RADIUS). 

Consider a mass-mobility relation of

$m = m_{100} (d_{A} / d_{100}) ^ {\zeta}$

or, equivalently, 

${\rho}_{eff} = {\rho}_{100} (d_{A} / d_{100}) ^ {\zeta - 3}$

Under these conditions, one can compute an estimate of the mobility diameter of the agglomerate (e.g., from projected area diameter) in radius units, which we define as $d_{g,1}$. Then, equating the mass from summing the mass of the monomers in the agglomerate

$\frac{d_{A}}{d_{A,1}} = \left( \frac{n_{pp} \rho_m}{{\rho}_{100}} \frac{d_{100}^{{\zeta}-3} (2 r_{pp})^3}{d_{A,1}^{\zeta}} \right) ^ {1 / (\zeta - 3)}$

The quantity on the left acts to scale the entire aggregate up to match the given expression. 

Dispersion can be achieved by sampling $m_{100}$ from a lognormal distribution, centered on an expected geometric mean for $m_{100}$. 

