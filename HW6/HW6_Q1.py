import numpy as np
import matplotlib.pyplot as plt


J = 1.0       
k_B = 1.0     
N = 20        # Lattice dimensions (20x20)
STEPS = 1000000 # Total Monte Carlo steps for the main run

# Part A: Function to calculate total energy using vectorized arithmetic
def calculate_total_energy(spins):
    """
    Calculates total energy E = -J * sum(s_i * s_j) for neighbor pairs.
    Uses np.roll to vectorize the sum over neighbors (up, down, left, right).
    """
    neighbors = (
        np.roll(spins, 1, axis=0) +  # Down
        np.roll(spins, -1, axis=0) + # Up
        np.roll(spins, 1, axis=1) +  # Right
        np.roll(spins, -1, axis=1)   # Left
    )
    return -J * np.sum(spins * neighbors) / 2.0

# Part B: Metropolis Algorithm
def run_metropolis(T, steps, snapshot_indices=None):
    """
    Runs the simulation for a specific temperature T.
    """
    spins = np.random.choice([-1, 1], size=(N, N))
    current_mag = np.sum(spins)
    mag_history = np.zeros(steps)
    snapshots = []
    
    # Pre-calculate Boltzmann factors
    w = {
        4.0: np.exp(-4.0 * J / (k_B * T)),
        8.0: np.exp(-8.0 * J / (k_B * T))
    }

    if snapshot_indices is not None and 0 in snapshot_indices:
        snapshots.append(spins.copy())

    for t in range(1, steps + 1):
        i, j = np.random.randint(0, N, 2)
        s = spins[i, j]
        
        # Calculate local energy change dE
        nb_sum = (spins[(i+1)%N, j] + spins[(i-1)%N, j] + 
                  spins[i, (j+1)%N] + spins[i, (j-1)%N])
        
        dE = 2 * J * s * nb_sum
        
        accept = False
        if dE <= 0:
            accept = True 
        else:
            if np.random.random() < w.get(dE, 0): 
                accept = True
        
        if accept:
            spins[i, j] *= -1
            current_mag += 2 * (-s) 
            
        mag_history[t-1] = current_mag
        
        if snapshot_indices is not None and t in snapshot_indices:
            snapshots.append(spins.copy())
            
    return mag_history, snapshots

print("Running simulation...")

# Part C & D: Magnetization vs Time at T=1
print("1. Simulating Magnetization evolution at T=1...")
mag_hist, _ = run_metropolis(T=1.0, steps=STEPS)

# Check for symmetry breaking (Part D)
print("2. Checking symmetry breaking (final M of 5 short runs at T=1):")
for k in range(5):
    m_check, _ = run_metropolis(T=1.0, steps=50000)
    print(f"   Run {k+1}: Final Magnetization = {m_check[-1]}")

# Part E: Visualizing evolution at T=1, T=2, T=3
print("3. Generating snapshots for T = 1, 2, 3...")
temps = [1.0, 2.0, 3.0]
snap_points = [0, 100, 1000, 10000, 100000, 1000000]
results_visuals = {}

for T in temps:
    _, snaps = run_metropolis(T, STEPS, set(snap_points))
    results_visuals[T] = snaps

print("Simulation complete. Preparing report plots...")

# Plot 1: Magnetization vs Time (Part C)
fig1, ax1 = plt.subplots(figsize=(10, 6))
# Changed: increased linewidth to 2.5
ax1.plot(mag_hist, linewidth=2.5, alpha=0.9, color='#1f77b4') 

# Changed: increased font sizes for labels, title, and ticks
ax1.set_xlabel("Time (Monte Carlo Steps)", fontsize=14)
ax1.set_ylabel("Total Magnetization $M$", fontsize=14)
ax1.set_title("Magnetization vs. Time (T=1.0)", fontsize=16, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Plot 2: Grid Snapshots (Part E)
rows = len(temps)
cols = len(snap_points)
fig2, axes = plt.subplots(rows, cols, figsize=(14, 8), sharex=True, sharey=True)

for r, T in enumerate(temps):
    snaps = results_visuals[T]
    for c, img in enumerate(snaps):
        ax = axes[r, c]
        ax.imshow(img, cmap='binary', interpolation='none', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Changed: increased font sizes for headers
        if r == 0:
            ax.set_title(f"Step {snap_points[c]}", fontsize=12)
        if c == 0:
            ax.set_ylabel(f"T = {T}", fontsize=14, fontweight='bold')

fig2.suptitle("Ising Model Evolution: Grid Snapshots", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

print("All plots generated.")