# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:00:15 2019

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#import frequency and lumped port volatge from text files
#assumes measured at same frequencies 
Filename = "2.5mm.txt"
freq = np.loadtxt(Filename, skiprows = 5, usecols = 0) *1000
lpv25 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "5mm.txt"
lpv5 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "7.5mm.txt"
lpv75 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "10mm.txt"
lpv10 = np.loadtxt(Filename, skiprows=5, usecols=1)





#plot AF versus frequency
plt.figure(1)

plt.plot(freq, 10*np.log10(1/lpv25), label = "2.5mm")
plt.plot(freq, 10*np.log10(1/lpv5), label = "5mm")
plt.plot(freq, 10*np.log10(1/lpv75), label = "7.5mm")
plt.plot(freq, 10*np.log10(1/lpv10), label = "10mm")


plt.title('AF vs freq for different gap sizes using dipole in free space')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()

#plot voltage at 83mhz sweeping gap size

Filename = "83mhzSweepGap.txt"
gapsize = np.loadtxt(Filename, skiprows = 5, usecols = 0) * 1000
lpv = np.loadtxt(Filename, skiprows=5, usecols=2)


plt.figure(2)

plt.plot(gapsize,lpv)



plt.title('LPV vs gap size using dipole in free space at 83MHz')
plt.xlabel('Gap size [mm]')
plt.ylabel('LPV')
