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
Filename = "magLPV_bicon_box_3mm_steelWalls_25__300_1_12.6.20.txt"
freq = np.loadtxt(Filename, skiprows = 5, usecols = 0)
lpv = np.loadtxt(Filename, skiprows=5, usecols=1)


Filename = "magEx_empty_box_steelWalls_25__300_1_12.6.20.txt"
Ex = np.loadtxt(Filename, skiprows=5, usecols=1)
#com4mm5Af = comAf + 5.6

Filename = "magEx_empty_box_steelWalls_100Vm_25__300_1_12.6.20.txt"
Ex100 = np.loadtxt(Filename, skiprows=5, usecols=1)

af = 10 * np.log10(Ex/lpv) + 5.6 


notNormalAf = 10 * np.log10(1/lpv) + 5.6


Filename = 'AF_bicon_box_orientation_11.17.20_180ohm_noLine_2.94mmGap.txt'
oldFreq = np.loadtxt(Filename, skiprows = 5, usecols = 0)
oldAf = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "CST_nores.txt"
cstFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstAf = np.loadtxt(Filename, delimiter = ',', usecols=1)

Filename = "E_x_center_Db_CST.txt"
cstEFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstE = np.loadtxt(Filename, delimiter = ',', usecols=1)



#plot AF versus frequency
plt.figure(1)
plt.plot(cstFreq, cstAf, label = "cst")
plt.plot(freq, af, label = "comsol 10log(Ex/lpv) + 5.6")
plt.plot(freq, notNormalAf, label = "comsol 10log(1/lpv)+ 5.6")

plt.title('AF bicon in box (no res)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()

plt.figure(2)
plt.title('X component of E field at center of room, dB')
plt.plot(freq, 20*np.log10(Ex), label = 'comsol 20log(E_x) for 1V/m')
plt.plot(freq, 20*np.log10(Ex100), label = 'comsol 20log(E_x) for 100V/m')
plt.plot(cstEFreq, cstE, 'g', label = 'cst')
plt.xlabel('Frequency [MHz]')
plt.ylabel('E_x')
plt.legend()

plt.figure(3)
plt.plot(freq, Ex)