#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:38:49 2024

@author: jbarrieu
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up Parameters
Ngrid = 100
Nsteps = 500
dt = 0.01
dx = 2.0

# Set Initial conditions
x = np.arange(Ngrid) * dx
f1, f2, f3 = np.ones(Ngrid), np.zeros(Ngrid), np.ones(Ngrid)
u = np.zeros(Ngrid+1)
P = np.zeros(Ngrid)
c = np.zeros(Ngrid)

#Define Advection function to Use in Loop
def advection(f, u, dt, dx):
    J = np.zeros(len(f)+1)
    J[1:-1] = np.where(u[1:-1] > 0, f[:-1] * u[1:-1], f[1:] * u[1:-1])
    f = f - (dt / dx) * (J[1:] - J[:-1])
    return f

# Apply initial Gaussian perturbation
Amp = 5000
sigma = Ngrid/10
f3 = f3 + Amp * np.exp(-(x - x.max()/2) ** 2 / sigma ** 2)

# Set Up Plot Conditions/Axis
plt.ion()
fig, ax = plt.subplots(2, 1)
x1, = ax[0].plot(x, f1, 'r-')
x2, = ax[1].plot(x, f2, 'b-')
ax[0].set_xlim([0, dx*Ngrid+1])
ax[0].set_ylim([0, 5])
ax[1].set_xlim([0, dx*Ngrid+1])
ax[1].set_ylim([0, 10])
ax[0].set_xlabel('Position')
ax[1].set_xlabel('Position')
ax[0].set_ylabel('Density')
ax[1].set_ylabel('Mach number')
ax[0].set_title('Density and Mach number of an Adiabatic Shock')
fig.canvas.draw()

# Main loop
for ct in range(Nsteps):
    u[1:-1] = 0.5 * ((f2[:-1] / f1[:-1]) + (f2[1:] / f1[1:]))
    
    f1 = advection(f1, u, dt, dx)
    f2 = advection(f2, u, dt, dx)
    f3 = advection(f3, u, dt, dx)
    
    P = 2/5 * (f3 - ((f2**2) / (2*f1)))
    
    f2[1:-1] = f2[1:-1] - 0.5 * (dt / dx) * (P[2:] - P[:-2])
    f2[0] -= 0.5 * (dt / dx) * (P[1] - P[0])
    f2[-1] -= 0.5 * (dt / dx) * (P[-1] - P[-2])
    
    f3[1:-1] = f3[1:-1] - 0.5 * (dt / dx) * ((f2[2:]/f1[2:]) * P[2:] - (f2[:-2]/f1[:-2]) * P[:-2])
    f3[0] -= 0.5 * (dt / dx) * ((f2[1]/f1[1]) * P[1] - (f2[0]/f1[0]) * P[0])
    f3[-1] -= 0.5 * (dt / dx) * ((f2[-1]/f1[-1]) * P[-1] - (f2[-2]/f1[-2]) * P[-2])
    
    
    c = np.sqrt((5/3) * (P / f1))
    x1.set_ydata(f1)
    x2.set_ydata(np.abs(f2) / c)
    fig.canvas.draw()
    plt.pause(0.001)