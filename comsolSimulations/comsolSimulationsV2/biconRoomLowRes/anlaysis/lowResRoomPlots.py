# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 04:57:41 2021

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
'''cstfreq = np.loadtxt("cstLPVLowRes.txt", usecols=0)
cstLPV = np.loadtxt("cstLPVLowRes.txt", usecols=1)
cstAF = np.loadtxt("cstLowResAF.txt", usecols=3)
cstAFreal = cstAF * (-1)
'''
comfreq = np.loadtxt("comsolLPVLowRes.txt", skiprows = 5, usecols = 0)
comLPV = np.loadtxt("comsolLPVLowRes.txt", skiprows = 5, usecols = 1)
comEx = np.loadtxt("lowResExCutPoint.txt", skiprows = 5, usecols = 1)
comExAvg = np.loadtxt("lowResExSurfaceAvg.txt", skiprows = 5, usecols = 1)

comAF = 20*np.log10(comEx/comLPV)+5.6
comAF2 = 20*np.log10(comExAvg/comLPV)+5.6

noResfreq = np.loadtxt('EfieldInBoxCutPoint.txt', skiprows=5, usecols=0)
noResEx = np.loadtxt('EfieldInBoxCutPoint.txt', skiprows=5, usecols=1)
noResExSurf= np.loadtxt('surfaceAvg300cmEmptyBox.txt', skiprows=5, usecols=1)
noReslpv = np.loadtxt('biconInBoxlpv.txt', skiprows=5, usecols=1)
noResAF = 20 * np.log10(noResEx/noReslpv) + 5.6
noResAFSurf = 20 * np.log10(noResExSurf/noReslpv) + 5.6

plt.figure(1)
plt.plot(comfreq, comAF, label = 'COMSOL AF cut point')
plt.plot(comfreq, comAF2, label = 'COMSOL AF surface average')
#plt.plot(cstfreq, cstAFreal, label = 'CST AF')
plt.title('Bicon in lowres room')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()

plt.figure(2)
plt.plot(comfreq, comAF2, label = 'COMSOL AF (surface average)')
plt.title('Bicon AF in lowres room')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()

plt.figure(3)
plt.plot(comfreq, comAF2, label = 'low res AF')
plt.plot(noResfreq, noResAFSurf, label = 'no res AF')
plt.title('Comsol Low Res vs. No Res (surface average)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()
