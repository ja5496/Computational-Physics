import numpy as np
import matplotlib.pyplot as plt
import random

L = 50  # Lattice size
steps = 300000  # Total steps for the main visualization run
tau_vis = 10000  # Time constant for the visualization run
T_start = 2.0   # Starting temperature

def run_dimer_annealing(tau, total_steps, L=50):
    # state grid: 0 = empty, 1 = occupied
    # partner grid: stores tuple of partner coordinates for each site
    
    grid = np.zeros((L, L), dtype=int)
    partners = np.full((L, L, 2), -1, dtype=int) 
    
    dimer_count = 0
    dimer_history = []
    
    early_state = None
    snapshot_step = int(total_steps * 0.05)
    
    for t in range(total_steps):
        # Temperature schedule
        T = T_start * np.exp(-t / tau)
        
        # Choose a site and a random neighbor
        x, y = np.random.randint(0, L), np.random.randint(0, L)
        
        # Pick a neighbor
        direction = np.random.randint(0, 4)
        dx, dy = 0, 0
        if direction == 0: dy = 1
        elif direction == 1: dy = -1
        elif direction == 2: dx = -1
        elif direction == 3: dx = 1
        
        nx, ny = x + dx, y + dy
        
        # Check boundary conditions
        if 0 <= nx < L and 0 <= ny < L:
            
            # Both sites are currently empty -> Add Dimer
            if grid[x, y] == 0 and grid[nx, ny] == 0:
                # Always accept addition 
                grid[x, y] = 1
                grid[nx, ny] = 1
                partners[x, y] = [nx, ny]
                partners[nx, ny] = [x, y]
                dimer_count += 1
                
            # These two sites form a single dimer -> Remove Dimer
            elif grid[x, y] == 1 and grid[nx, ny] == 1:
                # Check if partners
                if np.array_equal(partners[x, y], [nx, ny]):
                    # Calculate acceptance probability
                    if np.random.random() < np.exp(-1.0 / T):
                        grid[x, y] = 0
                        grid[nx, ny] = 0
                        partners[x, y] = [-1, -1]
                        partners[nx, ny] = [-1, -1]
                        dimer_count -= 1
        
        # Save history
        if t == snapshot_step:
            early_state = grid.copy()
            
    return dimer_count, grid, early_state

print(f"Running visualization simulation (tau={tau_vis})...")
final_count, final_grid, early_grid = run_dimer_annealing(tau_vis, steps)
print(f"Final Dimer Count: {final_count}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(early_grid, cmap='binary', interpolation='nearest')
axes[0].set_title(f"Early State (t={int(steps*0.05)})", fontsize=20)
axes[0].set_xlabel("Lattice X", fontsize=16)
axes[0].set_ylabel("Lattice Y", fontsize=16)
axes[0].tick_params(labelsize=14)

axes[1].imshow(final_grid, cmap='binary', interpolation='nearest')
axes[1].set_title(f"Final State (Dimers={final_count})", fontsize=20)
axes[1].set_xlabel("Lattice X", fontsize=16)
axes[1].set_ylabel("Lattice Y", fontsize=16)
axes[1].tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('dimer_plot.png')
plt.show()

# --- 3. Run Cooling Schedule Comparison (for Table) ---
taus = [100, 1000, 10000, 50000]
print("\n--- Cooling Schedule Comparison ---")
print(f"{'Tau':<10} | {'Final Dimers':<15} | {'Fill Fraction (%)':<20}")
print("-" * 50)

for t_val in taus:
    # Scale total steps slightly for very long taus to ensure cooling
    run_steps = max(300000, t_val * 20) 
    count, _, _ = run_dimer_annealing(t_val, run_steps)
    fill_frac = (count * 2) / (L * L) * 100
    print(f"{t_val:<10} | {count:<15} | {fill_frac:.2f}%")