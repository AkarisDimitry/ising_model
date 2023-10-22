# 2D Ising Model Simulation with Metropolis Algorithm

![image](https://github.com/AkarisDimitry/ising_model/assets/34775621/21308bbf-2e7b-4c96-97fa-45b66e2d7a26)

This repository contains a PyTorch-based simulation of the 2D Ising model using the Metropolis algorithm. The code provides insights into how the average spin and energy of the system change over time and temperature.

## Table of Contents
1. [Background](#background)
2. [Physics Behind](#physics-behind)
3. [Usage](#usage)
4. [Visualization](#visualization)
5. [Dependencies](#dependencies)

## Background

The Ising model is a mathematical model in statistical mechanics. It describes a system of spins that can be in one of two states (+1 or -1). The model can be used to explain ferromagnetism, where atomic spins align such that the material exhibits a net magnetic moment.

## Physics Behind

- **Ising Model**: A lattice (or grid) is populated by spins that can either be up (+1) or down (-1). The energy of the system is determined by the interactions between these neighboring spins. Like spins decrease the system's energy, while opposite spins increase it.
  
- **Metropolis Algorithm**: It's a Monte Carlo method used to simulate the evolution of the Ising model system over time. The algorithm decides whether to flip a spin or not based on certain criteria that depend on the change in energy due to the flip and the temperature of the system.

- **Phase Transitions**: As temperature varies, the system can undergo phase transitions, e.g., from a ferromagnetic phase where spins are aligned to a phase where they are random.

## Usage

1. Ensure you have the necessary dependencies installed (listed below).
2. Clone the repository.
3. Run the main script to see the visualization of the lattice and plots of average spin and energy vs. temperature.

```bash
python ising_model_CPU.py
python ising_model_3D_GPU.py
python ising_model_GPU.py
```

## Visualization
The code provides two main visualizations:

1. **Initial Lattice Configuration**: This shows the initial configuration of spins on the lattice. The code initializes two lattices: one with 75% negative spins and another with 75% positive spins.

2. **Average Spin & Energy Plots**: After running the Metropolis algorithm over a range of temperatures, the code plots how the average spin and energy change. This gives insights into the behavior of the system, including phase transitions.

## Dependencies
- Python (>=3.6)
- PyTorch
- NumPy
- Matplotlib
- SciPy
