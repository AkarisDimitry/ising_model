import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d, conv3d
from scipy.ndimage import convolve, generate_binary_structure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 50

init_random = torch.rand((N,N,N)).to(device)
lattice_p = torch.zeros((N,N,N)).to(device)
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1

def get_energy_arr(lattice):
    # applies the nearest neighbours summation
    kern = generate_binary_structure(3, 1) 
    kern[1][1][1] = False
    kern = torch.tensor(kern.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    arr = -lattice * conv3d(lattice, kern, padding='same')
    return arr
    
def get_energy(lattice):
    # applies the nearest neighbours summation
    return get_energy_arr(lattice).sum(axis=(2,3,4)).squeeze()

def get_dE_arr(lattice):
    return -2*get_energy_arr(lattice)

lattices = torch.stack([lattice_p, lattice_p]).unsqueeze(dim=1)

def metropolis(spin_tensor_batch, times, BJs):
    energies = []
    avg_spins = []
    spin_tensor_batch = torch.clone(spin_tensor_batch)
    BJs = BJs.reshape([-1,1,1,1,1])
    for t in range(times):
        i = np.random.randint(0,2)
        j = np.random.randint(0,2)
        k = np.random.randint(0,2)
        dE = get_dE_arr(spin_tensor_batch)[:,:,i::2,j::2,k::2]
        change = (dE>=0)*(torch.rand(dE.shape).to(device) < torch.exp(-BJs*dE)) + (dE<0)
        spin_tensor_batch[:,:,i::2,j::2,k::2][change] *=-1
        energies.append(get_energy(spin_tensor_batch))
        avg_spins.append(spin_tensor_batch.sum(axis=(1,2,3,4))/N**3)
    return torch.vstack(avg_spins), torch.vstack(energies)

def get_spin_energy(lattice, BJs):
    lattices = lattice.unsqueeze(dim=0).repeat(len(BJs),1,1,1,1)
    spins, energies = metropolis(lattices, 1000, BJs)
    spins_avg = torch.mean(spins[-400:], axis=0)
    energies_avg = torch.mean(energies[-400:], axis=0)
    energies_std = torch.std(energies[-400:], axis=0)
    return spins_avg, energies_avg, energies_std

BJs = 1/torch.linspace(3, 5.5, 20).to(device)
spins_avg_p, E_means_p, E_stds_p = get_spin_energy(lattice_p, BJs)

# According to https://arxiv.org/abs/2205.12357 (page 5) the accepted value for the critical temperature in the 3D Ising model is 


plt.figure(figsize=(8,5))
plt.plot(1/BJs.cpu(), spins_avg_p.cpu(), 'o--')
plt.axvline(1 / 0.221654626, color='k', ls='--')
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











