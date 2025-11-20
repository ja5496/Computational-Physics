import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor

df = pd.read_csv('particles.dat', header=None, delim_whitespace=True, names=['x', 'y'])
x_coords = df['x'].values
y_coords = df['y'].values

M = 100   # grid size

# Part A

def cloud_in_cell(x_coords, y_coords, M):
    grid = np.zeros((M, M))

    for particle in range(len(x_coords)):   
        x = float(x_coords[particle])
        y = float(y_coords[particle])

        i = floor(x)
        j = floor(y)

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
plt.title("Cloud-in-Cell Density Grid", fontsize=18)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Part B

def solve_poisson_relaxation(rho, max_iter=50000, tol=1e-4):
    phi = np.zeros_like(rho)
    phi_new = np.zeros_like(rho)

    for it in range(max_iter):

        # Jacobi update
        phi_new[1:-1, 1:-1] = 0.25 * (
            phi[2:, 1:-1] + phi[:-2, 1:-1] +
            phi[1:-1, 2:] + phi[1:-1, :-2] +
            rho[1:-1, 1:-1]
        )

        # Check convergence
        diff = np.max(np.abs(phi_new - phi))
        if it % 100 == 0:
            print(f"Iteration {it}, max difference = {diff:.6f}")

        if diff < tol:
            print(f"\nConverged in {it} iterations.")
            break

        phi, phi_new = phi_new, phi  # swap references
        iteration = it
    print 

    return phi, iteration

# Solve for potential
phi, _ = solve_poisson_relaxation(-density)  # solves Laplacian(phi) = -rho

plt.figure(figsize=(6, 5))
plt.imshow(phi.T, origin='lower', cmap='magma')
plt.colorbar(label="Potential φ")
plt.title("Electric Potential", fontsize=18, fontweight='bold')
plt.xlabel("x",fontsize=18)
plt.ylabel("y",fontsize=18)
plt.show()

# Part C

def sor_iteration(phi, rho, omega):
    """
    Perform ONE Gauss–Seidel SOR sweep.
    Updates phi in-place and returns max update magnitude.
    Poisson equation: ∇²φ = -rho.
    """
    M = phi.shape[0]
    max_diff = 0.0

    for i in range(1, M-1):
        for j in range(1, M-1):

            # Standard Gauss–Seidel update (without relaxation)
            phi_gs = 0.25 * (
                phi[i+1, j] + phi[i-1, j] +
                phi[i, j+1] + phi[i, j-1] +
                rho[i, j]
            )

            # Apply relaxation
            new_val = (1 - omega) * phi[i, j] + omega * phi_gs

            diff = abs(new_val - phi[i, j])
            if diff > max_diff:
                max_diff = diff

            phi[i, j] = new_val

    return max_diff


def solve_poisson_sor(rho, omega, max_iter=5000, tol=1e-4):
    """
    Solve Poisson with SOR using a fixed relaxation parameter omega.
    Returns converged phi and iteration count.
    """
    phi = np.zeros_like(rho)

    for it in range(max_iter):
        diff = sor_iteration(phi, rho, omega)

        if it % 100 == 0:
            print(f"[ω={omega:.3f}] Iter {it}, max diff = {diff:.6e}")

        if diff < tol:
            return phi, it

    return phi, max_iter


# Golden Ratio Search

def golden_section_optimize(rho, max_iter=30):
    """
    Find the best ω in [1, 2] for fastest convergence using golden-section search.
    Minimizes number of iterations required for convergence.
    """
    phi_test = None

    # Search interval
    a, b = 1.0, 2.0
    gr = (np.sqrt(5) - 1) / 2  # golden ratio factor

    # Evaluate function = iteration count for SOR(omega)
    def f(omega):
        _, iters = solve_poisson_sor(rho, omega, max_iter=2000, tol=1e-3)
        return iters

    # Initial interior points
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    fc = f(c)
    fd = f(d)

    print("\nStarting Golden Section Optimization...\n")

    for step in range(max_iter):
        print(f"Step {step}: interval [{a:.4f}, {b:.4f}]")

        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, fc = c, fd
            c = d
            d = a + gr * (b - a)
            fd = f(d)

    omega_opt = (a + b) / 2
    print(f"\nOptimal ω ≈ {omega_opt:.5f}")
    return omega_opt

# Compute optimal relaxation parameter
omega_opt = golden_section_optimize(-density)

# Solve Poisson using optimal parameter
phi_sor, iters = solve_poisson_sor(-density, omega_opt)

print(f"\nSOR converged in {iters} iterations using ω = {omega_opt:.5f}")

plt.figure(figsize=(6, 5))
plt.imshow(phi_sor.T, origin='lower', cmap='magma')
plt.colorbar(label="Potential (SOR)")
plt.title(f"Potential with Gauss-Seidel (ω={omega_opt:.3f})", fontsize=18, fontweight='bold')
plt.xlabel("x", fontsize=18)
plt.ylabel("y", fontsize=18)
plt.show()
