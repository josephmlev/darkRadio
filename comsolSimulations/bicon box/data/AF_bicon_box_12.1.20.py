# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:00:15 2019

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#import frequency and lumped port volatge from text files, and convert imaginary LPV to real
comFilename = "AF_bicon_box_orientation_11.17.20_180ohm_noLine_2.94mmGap.txt"
comFreq = np.loadtxt(comFilename, skiprows = 5, usecols = 0)
comAf = np.loadtxt(comFilename, skiprows=5, usecols=1) + 5.6
#com5Af = comAf + 5.6

comInfFilename = "bicon_box_orientation_11.14.20_180ohm.txt"
comInfFreq = np.loadtxt(comInfFilename, skiprows = 5, usecols = 0)
comInfAf = np.loadtxt(comInfFilename, skiprows=5, usecols=1) + 5.6
#com4mm5Af = comAf + 5.6

cstFilename = "CST_nores.txt"
cstFreq = np.loadtxt(cstFilename, delimiter=',', usecols = 0)
cstAf = np.loadtxt(cstFilename, delimiter=',', usecols=1) + 5.6

#plot AF versus frequency
plt.figure(1)

plt.plot(comFreq, comAf, label = "steel box, no line, 2.94mm")
#plt.plot(comInfFreq, comInfAf, label = 'perfect conducting box, with line, 3.7mm')
plt.plot(cstFreq, cstAf, label = 'CST, no line, 4mm')


plt.title('')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()
plt.ylim(-10, 85)
