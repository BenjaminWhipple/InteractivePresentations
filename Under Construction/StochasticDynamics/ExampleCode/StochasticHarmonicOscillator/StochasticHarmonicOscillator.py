#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:06:56 2024

@author: benjaminwhipple
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0  # mass
gamma = 0.1  # damping coefficient
k = 1.0  # spring constant
F0 = 0.5  # Amplitude of the deterministic driving force
omega = 1.0  # Frequency of the deterministic driving force
sigma = 5.0  # Standard deviation of the stochastic force

# Simulation parameters
dt = 0.01  # timestep
T = 40  # total time
N = int(T / dt)  # number of timesteps

# Initial conditions
x = 0.0  # initial position
v = 0.0  # initial velocity

# Time vector for plotting
t = np.arange(0, T, dt)


# Stochastic simulation (Euler-Murayama)
x_stoc_arr = np.zeros(N)
v_stoc_arr = np.zeros(N)

noises = []

for i in range(N):
    # Generate stochastic force
    xi = np.random.normal(0, sigma / np.sqrt(dt))
    noises.append(xi)
    
    # Update velocity and position using Euler-Maruyama method
    v = v + (-gamma/m * v - k/m * x + F0/m * np.cos(omega * i * dt) + xi/m) * dt
    x = x + v * dt
    
    # Store position and velocity
    x_stoc_arr[i] = x
    v_stoc_arr[i] = v

# Deterministic

x_det_arr = np.zeros(N)
v_det_arr = np.zeros(N)


for i in range(N):
    
    # Update velocity and position using Euler method
    v = v + (-gamma/m * v - k/m * x + F0/m * np.cos(omega * i * dt)) * dt
    x = x + v * dt
    
    # Store position and velocity
    x_det_arr[i] = x
    v_det_arr[i] = v

# Plotting
# Plot stochastic
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_stoc_arr, label='Position, Stochastic')
plt.ylabel('p')
plt.title('Stochastic Harmonic Oscillator')
plt.xticks(ticks=[])

plt.subplot(2, 1, 2)
plt.plot(t, v_stoc_arr, color='red', label='Velocity, Stochastic')
plt.xlabel('Time')
plt.ylabel('v')

plt.tight_layout()
plt.savefig("StochasticHarmonicOscillator.png")

# Plot deterministic
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_det_arr, label='Position, Stochastic')
plt.ylabel('p')
plt.title('Harmonic Oscillator')
plt.xticks(ticks=[])

plt.subplot(2, 1, 2)
plt.plot(t, v_det_arr, color='red', label='Velocity, Stochastic')
plt.xlabel('Time')
plt.ylabel('v')

plt.tight_layout()
plt.savefig("DeterministicHarmonicOscillator.png")

# Plot noise
plt.figure(figsize=(6,4))
plt.title("Noise Signal")
plt.plot(t,noises,color='green',alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Noise")
plt.tight_layout()
plt.savefig("Noise.png")