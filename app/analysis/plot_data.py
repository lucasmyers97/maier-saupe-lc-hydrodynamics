"""
This script does essentialy the same thing as `plot_bulk_free_energy`, though
I think I recorded the S-values this time around (or maybe not).
Not sure why this file exists.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt


data_filename = r"/home/lucas/Documents/grad-work/research/maier-saupe-lc-hydrodynamics/data/simulations/bulk-free-energy-calculation/2021-11-10/data.h5"
f = h5py.File(data_filename, 'r')

key_1 = "free_energy_3.300000"
key_2 = "free_energy_3.404900"
key_3 = "free_energy_3.500000"

free_energy_1 = f.get(key_1)[...]
free_energy_2 = f.get(key_2)[...]
free_energy_3 = f.get(key_3)[...]
S_vals = f.get('S_val')[...]

x_pixels = (318, 973)
y_pixels = (712, 18)
x_range = (0, 0.6)
y_range = (-0.02, 0.02)
pixels_tot = (1104, 844)

x_unit_per_pixel = (x_range[1] - x_range[0]) / (x_pixels[1] - x_pixels[0])
x_left = x_unit_per_pixel * (-x_pixels[0])
x_right = x_unit_per_pixel * pixels_tot[0] + x_left

y_unit_per_pixel = np.abs((y_range[1] - y_range[0]) / (y_pixels[1] - y_pixels[0]))
y_down = y_range[0] - (pixels_tot[1] - y_pixels[0]) * y_unit_per_pixel
y_up = y_range[1] + y_pixels[1] * y_unit_per_pixel

image_filename = "/home/lucas/Documents/grad-work/research/maier-saupe-lc-hydrodynamics/data/simulations/bulk-free-energy-calculation/cody-data/bulk-free-energy.png"
img = plt.imread(image_filename)

plt.imshow(img, extent=[x_left, x_right, y_down, y_up], aspect="auto")

linewidth = 0.5
plt.plot(S_vals, free_energy_1, linewidth=linewidth)
plt.plot(S_vals, free_energy_2, linewidth=linewidth)
plt.plot(S_vals, free_energy_3, linewidth=linewidth)
plt.xlim((-0.1, 0.7))
plt.ylim((-0.02, 0.02))
plt.xlabel("S")
plt.ylabel("f_b")

save_folder = r"/home/lucas/Documents/grad-work/research/maier-saupe-lc-hydrodynamics/data/simulations/bulk-free-energy-calculation/2021-11-10/bulk_energy_comparison.png"
plt.savefig(save_folder, dpi=1500)
plt.show()
