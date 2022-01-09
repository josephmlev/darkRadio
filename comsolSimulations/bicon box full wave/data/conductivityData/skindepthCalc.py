#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:30:56 2021

@author: joseph

This script calculates the skin depth as a function of conductivitity of the walls. 
"""
import numpy as np
import matplotlib.pyplot as plt 

    
def skin(s):
    return (np.pi * 10e6 * 12.57e-7 * s)**(-1/2)

#this is only needed in very hight (optical) frequencies, where epsilon and mu are complex
def skinFull(s):
    return (np.real(np.sqrt(1j * 10e6 * 12.57e-7 * 1 * (s + 1j * 1e6 * 8.85e-12))))**-1

sigArr = np.linspace(1, 1e6, 1000)
plt.close('all')

plt.xlabel('sigma (S/m)')
plt.ylabel('skin depth (m)')
plt.title('Skin depth as a function of conductivity (at 10 MHz)')
plt.loglog(sigArr, skin(sigArr))

