
# dlca

A relatively simple code to perform diffusion-limited cluster agglomeration (DLCA).


### Scaling aggregates to mass-mobility relation

Simulations occur in radius units (i.e., 1 length unit = RADIUS) of the monomers. 

Consider a mass-mobility relation of

$m = m_{100} (d / d_{100}) ^ {\zeta}$

Under these conditions, one can compute an estimate of the mobility diameter of the agglomerate (e.g., from projected area diameter) in radius units, which we define as $d_{g,1}$. Then, equating the mass from summing the mass of the monomers in the agglomerate

$\frac{d_{g}}{d_{g,1}} = d_{g,1} ^ {-\zeta / (\zeta - 3)} \left(\frac{N \rho_m \pi}{6 m_{100}} d_{100}^{\zeta}\right) ^ {1 / (\zeta - 3)}$

The quantity on the left acts to scale the entire aggregate up to match the given expression. 

Dispersion can be achieved by sampling $m_{100}$ from a lognormal distribution, cenetered on an expected geometric mean for $m_{100}$. 

