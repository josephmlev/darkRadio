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
Filename = "magEx_empty_box_steelWalls_fullwave_10__300_.3_2.23.21.txt"
freq = np.loadtxt(Filename, skiprows = 5, usecols = 0)
Ex = np.loadtxt(Filename, skiprows=5, usecols=1)



Filename = "magLPV_bicon_box_3mm_steelWalls_fullwave_30_300_.3_2.23.21.txt"
lpv = np.loadtxt(Filename, skiprows=5, usecols=1)

af = 20 * np.log10(Ex/lpv) + 5.6 



Filename = "CST_af_noRes_12.21.20.txt"
cstFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstAf = np.loadtxt(Filename, delimiter = ',', usecols=1)


Filename = "CST_nores_20log10(mag(lpv)).txt"
cstLpvFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstLpv = np.loadtxt(Filename, delimiter = ',', usecols=1)


Filename = "E_x_center_Db_CST.txt"
cstEFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstE = np.loadtxt(Filename, delimiter = ',', usecols=1)


#plot AF versus frequency
plt.figure(1)
plt.plot(cstFreq, cstAf, label = "cst 20 log(mag(Ex/lpv)) + 5.6")
plt.plot(freq, af, label = "comsol 20 log(mag(Ex/lpv)) + 5.6")
#plt.plot(freq, notNormalAf, label = "comsol 10log(1/lpv)+ 5.6")

plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()

plt.figure(2)
plt.plot(cstEFreq, cstE, label = 'cst')
plt.title('X component of E field at center of room, dB (1000 pts)')
plt.plot(freq, 20*np.log10(Ex), label = 'comsol 20log(E_x) for 1V/m')
#plt.plot(freq, 20*np.log10(Ex100), label = 'comsol 20log(E_x) for 100V/m')
plt.xlabel('Frequency [MHz]')
plt.ylabel('E_x')
plt.legend()

plt.figure(3)
plt.plot(cstLpvFreq, cstLpv, label = '20*log(mag(LPV)) (cst)')
plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV))(comsol)')
plt.ylabel('20*log(LPV)')
plt.xlabel('Frequency (MHz)')
plt.title('Lumped port voltage bicon in no res room (1000 pts)')
plt.legend()


plt.figure(4)
plt.plot(cstLpvFreq, cstLpv + 170, label = '20*log(mag(LPV)) + 170 (cst) ' )
plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV)) (comsol)' )
plt.ylabel('20*log(LPV)')
plt.xlabel('Frequency (MHz)')
plt.title('Lumped port voltage bicon in no res room (1000 pts)')
plt.legend()




