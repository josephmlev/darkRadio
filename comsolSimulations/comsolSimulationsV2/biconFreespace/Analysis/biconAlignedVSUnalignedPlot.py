# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 04:57:46 2021

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

freq = np.loadtxt("bicon4mmUnalignedArmsAF.txt", skiprows=5, usecols=0)
AF = np.loadtxt("bicon4mmUnalignedArmsAF.txt", skiprows=5, usecols=1)

freq2 = np.loadtxt('bicon4mmAlignedArmsAF.txt', skiprows=5, usecols=0)
AF2 = np.loadtxt('bicon4mmAlignedArmsAF.txt', skiprows=5, usecols=1)

BAF = AF+5.6
BAF2 = AF2+5.6
plt.figure(1)
plt.plot(freq, BAF, label='Unaligned')
plt.plot(freq2, BAF2, label='Aligned')
plt.title('Bicon Unaligned VS Aligned')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()