import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # Import the animation function

L = 1e-8 # width of square well
x_0 = L/2
sigma = 1e-10
k = 5e10
N = 1000 # number of spacial slices 
hbar = 1 # Using reduced units for simplicity
m = 1    # Using reduced units for simplicity
dt = 1e-18

# Part A: Write C-N code to develop the state forward in time

def psi_0(L, N, k, x_0, sigma):
    a = L/N
    psi = []
    for i in range(1,N+1): # goes from 1 to 1000
        x = i*a
        psi.append(np.exp(-(x-x_0)**2/(2*sigma**2)) * np.exp(1j*k*x))
    psi = np.array(psi)
    return psi

def matrices(L, N, dt): # construct the A and B matrices
    a = L/N
    a_1 = complex(1,dt*hbar/(2*m*a**2))
    a_2 = complex(0,-dt*hbar/(4*m*a**2))
    b_1 = complex(1,-dt*hbar/(2*m*a**2))
    b_2 = complex(0,dt*hbar/(4*m*a**2))
    A = np.eye(N)*a_1
    B = np.eye(N)*b_1
    for i in range(N-1):
        A[i][i+1] = A[i+1][i] = a_2 # Corrected: only one assignment
        B[i][i+1] = B[i+1][i] = b_2 # Corrected: only one assignment
    return A, B

def Crank_Nicolson(A, B, psi_0, dt, time_steps):
    psi_memory = [psi_0] # keep track for plotting
    current_psi = psi_0
    for t in range(time_steps):
        print(t)
        v = B @ current_psi 
        new_psi = np.linalg.solve(A,v)
        psi_memory.append(new_psi)
        current_psi = new_psi # Update for the next loop
    
    # The last element is 'new_psi', which is already in psi_memory
    return psi_memory[-1], psi_memory

print("Running simulation...")
A, B = matrices(L,N,dt)
psi_initial = psi_0(L, N, k, x_0, sigma)
time_steps = 3000 # Number of steps to simulate and animate

# Run the simulation and get the history of psi
final_psi, psi_memory = Crank_Nicolson(A, B, psi_initial, dt, time_steps)
print("Simulation complete. Preparing animation...")

# Create the x-axis points for plotting
a = L/N
x_points = np.arange(1, N + 1) * a

# Find the maximum probability density to set a fixed y-axis
max_prob = 0
for psi in psi_memory:
    max_prob = max(max_prob, np.max(np.abs(psi)**2))


fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, max_prob * 1.1) # Add 10% padding

ax.set_xlabel("Position (m)")
ax.set_ylabel("Probability Density |ψ|²")
ax.set_title("Wave Packet in a Square Well")
line, = ax.plot(x_points, np.abs(psi_memory[0])**2, lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Initialization function: sets the initial frame
def init():
    line.set_ydata(np.abs(psi_memory[0])**2)
    time_text.set_text('')
    return line, time_text

# Animation function: this is called sequentially for each frame
def animate(i):
    # Update the y-data of the plot line
    prob_density = np.abs(psi_memory[i])**2
    line.set_ydata(prob_density)
    
    # Update the time text
    time_text.set_text(f'Time Step: {i}')
    
    return line, time_text # Return the artists that were updated

frame_skip = 10
animation_frames = len(psi_memory) // frame_skip

ani = FuncAnimation(
    fig, 
    animate, 
    frames=lambda: (i * frame_skip for i in range(animation_frames)),
    init_func=init,
    blit=True,      
    interval=30,    # Delay between frames in milliseconds
    save_count=animation_frames # Ensure all frames are processed
)

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Animation window closed.")