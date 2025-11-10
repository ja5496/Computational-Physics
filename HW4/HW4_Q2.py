import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

G = 1.0
M = 1.0
mu = G*M/4.0
r0 = np.array([1.0, 0.0])
d = 1e-8
t_0 = 0
t_f = 20
h_0 = 0.1
#A = B = 1

r_a = 1.0
r_p = 1e-7
a = 0.5 * (r_a + r_p)
v_a = np.sqrt(mu * (2/r_a - 1/a))

def circ_speed(r):
    R = np.linalg.norm(r)  
    return np.sqrt(mu/R)

def f(t, y, A, B):
    r = y[:2]
    v = y[2:]
    R = np.linalg.norm(r)
    a = -mu * r / (R**3)        # gravitational acceleration
    return np.hstack([v, a])    # dy/dt = [vx, vy, ax, ay]

def f2(t, y, A, B):
    r = y[:2]
    v = y[2:]
    R = np.linalg.norm(r)
    V = np.linalg.norm(v)
    a = -mu * r / (R**3) - A*v/(V**3 + B)
    return np.hstack([v, a])

def rk4_adaptive(f, y0, t0, t1, h0, A, B, tol=d, hmin=1e-8, hmax=1e-1):
    """
    Adaptive RK4 (step-doubling) for vector ODEs (e.g., 2D).
    f(t, y): returns dy/dt as np.array([dxdt, dydt, ...])
    """
    t = t0
    y = np.array(y0, dtype=float)
    h = h0
    T = [t]
    Y = [y.copy()]
    p = 4  # RK4 order

    def step(t, y, h):
        k1 = f(t, y, A, B)
        k2 = f(t + 0.5*h, y + 0.5*h*k1, A, B)
        k3 = f(t + 0.5*h, y + 0.5*h*k2, A, B)
        k4 = f(t + h,     y + h*k3, A, B)
        return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    while t < t1:
        # make sure step isn't too tiny or overshooting
        if h <= 0: break

        r = np.sqrt(y[0]**2 + y[1]**2)

        if r <= 1e-7:
            break

        # step-doubling error estimate
        y_big  = step(t, y, h)
        y_half = step(t, y, 0.5*h)
        y_two  = step(t + 0.5*h, y_half, 0.5*h)
        err = np.linalg.norm(y_two - y_big, ord=np.inf)

        if err <= tol or not np.isfinite(err):
            # accept step
            t += h
            y = y_two
            T.append(t)
            Y.append(y.copy())

            # scale step size (order +1 = 5 → exponent 1/5)
            fac = 2.0 if err == 0.0 else np.clip(0.9*(tol/err)**0.2, 0.5, 2.0)
            h *= fac
        else:
            # reject and shrink step
            h *= 0.5

        # float floor so t always advances
        if t + h == t:
            h = max(np.nextafter(t, np.inf) - t, hmin)


    return np.array(T), np.vstack(Y)

# Part A:

y0 = np.array([1.0, 0.0, 0.0, v_a]) 
T, Y = rk4_adaptive(f, y0, t_0, t_f, h_0, 1, 1)
x, y = Y[:,0], Y[:,1]
vx, vy = Y[:,2], Y[:,3]
r = np.sqrt(x**2 + y**2)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x, y)
plt.title("Orbital Trajectory (No Friction)", fontsize=18)
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)

plt.subplot(1,2,2)
plt.plot(T, r)
plt.yscale('log')
plt.xlabel("Time", fontsize=16) 
plt.ylabel("radius (log scale)", fontsize=16)
plt.title("Radius over time", fontsize=18)

plt.tight_layout()
plt.show()

# Part B

y0 = np.array([1.0, 0.0, 0.0, 0.8*circ_speed(1)]) 
T, Y = rk4_adaptive(f2, y0, t_0, t_f, h_0, 1, 1)
x, y = Y[:,0], Y[:,1]
vx, vy = Y[:,2], Y[:,3]
r = np.sqrt(x**2 + y**2)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x, y)
plt.title("Orbital Trajectory (Friction)", fontsize=18)
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)

plt.subplot(1,2,2)
plt.plot(T, r)
plt.yscale('log')
plt.xlabel("Time", fontsize=16) 
plt.ylabel("Radius (log scale)", fontsize=16)
plt.title("Radius over time", fontsize=18)

plt.tight_layout()
plt.show()

#Part C:

y0 = np.array([1.0, 0.0, 0.0, 0.8*circ_speed(1)]) 
A_arr = np.linspace(0.5, 10, 5)
B_arr = np.linspace(0.5, 10, 5)

heatmap_data = []

for i in A_arr:
    for j in B_arr:
        print('A: ', i, ';  B: ', j)
        T, Y = rk4_adaptive(f2, y0, t_0, t_f, h_0, i, j)
        heatmap_data.append([i,j,T[-1]])

heatmap_data = np.array(heatmap_data)
x = heatmap_data[:, 0]
y = heatmap_data[:, 1]
z = heatmap_data[:, 2]

# define grid resolution (increase for smoother heatmap)
bins = 5

# compute mean z-value in each (x, y) bin
heatmap, xedges, yedges, _ = plt.hist2d(
    x, y, bins=bins, weights=z, cmap='viridis'
)
counts, _, _ = np.histogram2d(x, y, bins=bins)
heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts!=0)

# plot
plt.figure(figsize=(6,5))
plt.imshow(
    heatmap.T, origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect='auto', cmap='viridis'
)
plt.colorbar(label='Collapse Time')
plt.xlabel('A', fontsize=18)
plt.ylabel('B', fontsize=18)
plt.title('Binary BH Collapse', fontsize=20, fontweight='bold')
