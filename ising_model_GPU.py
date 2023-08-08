# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d, conv3d
from scipy.ndimage import convolve, generate_binary_structure

# Set the device to CUDA if available, otherwise CPU. This helps in accelerating the computations using GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the size of the lattice (grid). This lattice represents a 2D Ising model.
N = 500

# Create a random tensor of size NxN with values between 0 and 1.
init_random = torch.rand((N,N)).to(device)

# Initialize the lattice for spins which will start with 75% negative spins.
lattice_n = torch.zeros((N, N)).to(device)
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

# Create another random tensor for the second lattice.
init_random = torch.rand((N,N)).to(device)

# Initialize the lattice for spins which will start with 75% positive spins.
lattice_p = torch.zeros((N, N)).to(device)
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1

# Plot the lattice_n to visualize the initial configuration of spins.
plt.pcolormesh(lattice_n.cpu())
plt.show()

# Create a 3x3 kernel that represents the nearest neighbors in a 2D lattice.
kern = generate_binary_structure(2, 1) 
kern[1][1] = False

# Convert the kernel to a PyTorch tensor and move it to the specified device.
kern = torch.tensor(kern.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

# Define a function to calculate the energy of the lattice.
# This uses the Ising model's energy calculation which involves summing over nearest neighbors.
def get_energy_arr(lattices):
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    kern = torch.tensor(kern.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    arr = -lattices * conv2d(lattices, kern, padding='same')
    return arr
    
def get_energy(lattices):
    return get_energy_arr(lattices).sum(axis=(1,2,3))

# Define a function to get the change in energy when a spin is flipped.
def get_dE_arr(lattices):
    return -2*get_energy_arr(lattices)

# Stack the two lattices (lattice_n and lattice_p) to process them together.
lattices = torch.stack([lattice_n, lattice_p]).unsqueeze(dim=1)

# Define the Metropolis algorithm which is a Monte Carlo method used to sample equilibrium states of the Ising model.
def metropolis(spin_tensor_batch, times, BJs):
    energies = []
    avg_spins = []
    spin_tensor_batch = torch.clone(spin_tensor_batch)
    BJs = BJs.reshape([-1,1,1,1])
    for t in range(times):
        # Randomly choose a position in the lattice.
        i = np.random.randint(0,2)
        j = np.random.randint(0,2)
        # Calculate the change in energy if the spin at the chosen position is flipped.
        dE = get_dE_arr(spin_tensor_batch)[:,:,i::2,j::2]
        # Decide whether to flip the spin or not based on Metropolis criteria.
        change = (dE>=0)*(torch.rand(dE.shape).to(device) < torch.exp(-BJs*dE)) + (dE<0)
        spin_tensor_batch[:,:,i::2,j::2][change] *=-1
        # Store the energy and average spin for each iteration.
        energies.append(get_energy(spin_tensor_batch))
        avg_spins.append(spin_tensor_batch.sum(axis=(1,2,3))/N**2)
    return torch.vstack(avg_spins), torch.vstack(energies)

# Set the initial values for BJs which is the inverse temperature times the interaction strength.
BJs = 0.5*torch.ones(lattices.shape[0]).to(device)

# Run the Metropolis algorithm for both lattices.
spins, energies = metropolis(lattices, 1000, BJs)

# Plot the average spin and energy vs. time steps for visualization.
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins[:,1].cpu())
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Average Spin $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energies[:,1].cpu())
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
plt.show()

# Define a function to calculate the average spin and energy for a given lattice over a range of temperature values.
def get_spin_energy(lattice, BJs):
    lattices = lattice.unsqueeze(dim=0).repeat(len(BJs),1,1,1)
    spins, energies = metropolis(lattices, 1000, BJs)
    spins_avg = torch.mean(spins[-400:], axis=0)
    energies_avg = torch.mean(energies[-400:], axis=0)
    energies_std = torch.std(energies[-400:], axis=0)
    return spins_avg, energies_avg, energies_std

# Set a range of temperature values.
BJs = 1/torch.linspace(1, 3, 20).to(device)

# Get the average spin and energy for both lattices over the temperature range.
spins_avg_n, E_means_n, E_stds_n = get_spin_energy(lattice_n, BJs)
spins_avg_p, E_means_p, E_stds_p = get_spin_energy(lattice_p, BJs)

# Plot the average spin vs. temperature for both lattices.
plt.figure(figsize=(8,5))
plt.plot(1/BJs.cpu(), spins_avg_n.cpu(), 'o--', label='75% of spins started negative')
plt.plot(1/BJs.cpu(), spins_avg_p.cpu(), 'o--', label='75% of spins started positive')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$')
plt.ylabel(r'$\bar{m}$')
plt.legend(facecolor='white', framealpha=1)
plt.show()

'''
Here's a brief overview of the physics:

The code simulates the 2D Ising model, a classic model in statistical mechanics.
In the Ising model, a lattice (or grid) is populated by spins that can be either up (+1) or down (-1).
The energy of the system is determined by the interactions between neighboring spins. Opposite neighboring spins increase the energy, while like neighboring spins decrease it.
The Metropolis algorithm is a method to simulate the evolution of this system over time and temperature, helping to understand phenomena like phase transitions.
The code runs the algorithm on two initial configurations: one with 75% negative spins and another with 75% positive spins.
The final plots give insights into how the average spin and energy change over time and temperature.
'''
