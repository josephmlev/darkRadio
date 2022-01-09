# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:22:52 2021

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

freq = np.loadtxt("bicon4mmRerunRealAF.txt", skiprows=5, usecols=0)
AF = np.loadtxt("bicon4mmRerunRealAF.txt", skiprows=5, usecols=1)

freq2 = np.loadtxt('Bicon_freespace_remove_line_11.23.20.200ishpoints4.001mmGapJoseph.txt', skiprows=5, usecols=0)
AF2 = np.loadtxt('Bicon_freespace_remove_line_11.23.20.200ishpoints4.001mmGapJoseph.txt', skiprows=5, usecols=1)
                   
BAF = AF+5.6
BAF2 = (2*AF2)+5.6


plt.figure(1)

plt.plot(freq, BAF, label='new bicon AF')
plt.plot(freq2, BAF2, label='old 4mm')
plt.title('new vs old 4mm gap')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log10(1/V)]+5.6')
plt.legend()