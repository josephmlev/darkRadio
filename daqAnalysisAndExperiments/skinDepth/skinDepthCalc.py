#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:30:56 2021

@author: joseph

This script calculates the skin depth as a function of conductivitity of the walls. 
"""
import numpy as np
import matplotlib.pyplot as plt 

    
def skin(s, freq):
    return (np.pi * freq * 12.57e-7 * s)**(-1/2)

#this is only needed in very hight (optical) frequencies, where epsilon and mu are complex
def skinFull(s, freq):
    return (np.real(np.sqrt(1j * freq * 12.57e-7 * 1 * (s + 1j * 1e6 * 8.85e-12))))**-1

sigArr = np.linspace(1e5, 1e8, 1000)
plt.close('all')


freqs = np.linspace(50*10**6, 300*10**6, 3)
for freq in freqs:
    plt.loglog(sigArr, skinFull(sigArr, freq), label = '%i' %(freq/10**6) + 'MHz')

plt.xlabel('sigma (S/m)')
plt.ylabel('skin depth (m)')
plt.title('Skin depth as a function of conductivity')
#plt.loglog(sigArr, skinFull(sigArr))
plt.legend()
