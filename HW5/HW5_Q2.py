import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor

df = pd.read_csv('HW5/particles.dat', header=None, names=['x', 'y'])
x_coords = df['x'].values
y_coords = df['y'].values

M = 100  

def cloud_in_cell(x_coords, y_coords, M):
    grid = np.zeros((M, M))

    for particle in range(len(x_coords)):   
        x = x_coords[particle]
        y = y_coords[particle]

        # lower-left grid indices
        i = floor(x)
        j = floor(y)

        # fractional distances
        dx = x - i
        dy = y - j

        if 0 <= i < M and 0 <= j < M:
            grid[i, j] += (1 - dx) * (1 - dy)
        if i+1 < M and 0 <= j < M:
            grid[i+1, j] += dx * (1 - dy)
        if 0 <= i < M and j+1 < M:
            grid[i, j+1] += (1 - dx) * dy
        if i+1 < M and j+1 < M:
            grid[i+1, j+1] += dx * dy
    return grid

density = cloud_in_cell(x_coords, y_coords, M)

plt.figure(figsize=(6, 5))
plt.imshow(density.T, origin='lower', cmap='viridis')
plt.colorbar(label="Charge Density")
plt.title("Cloud-in-Cell Density Grid")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
