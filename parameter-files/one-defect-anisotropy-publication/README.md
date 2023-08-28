---
author: "Lucas Myers"
date: 2023-08-22
---

# Anisotropic disclination structure publication

For sake of reproducibiliy, I'm listing information here that one would need to recreate all of the figures that appear in this publication.
The `.toml` files are configuration files that the `NematicSystemMPISim` executable takes to run a simulation.
The python script commands used to actually create the plots are listed below.

## Plotting script commands

Single-defect eigenvalue core structure:
```
python -m analysis.plotting.plot_eigenvalue_fourier_vs_r --data_folder ~/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-one-defect-anisotropy-publication/L3_4_5 --input_filename far_structure.h5 --plot_prefix far_fourier --n_modes 4 --data_key timestep_5500/Q_vec --r_cutoff 0.2
```
