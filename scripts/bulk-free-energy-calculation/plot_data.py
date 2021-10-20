import h5py
import numpy as np
import matplotlib.pyplot as plt


filename = r"/home/lucasmyers97/maier-saupe-lc-hydrodynamics/data/bulk-free-energy-calculation/2021-10-08/data.h5"
f = h5py.File(filename, 'r')

free_energy = f.get('free_energy')[...]
S_vals = f.get('S_val')[...]

plt.plot(S_vals, free_energy)
plt.show()