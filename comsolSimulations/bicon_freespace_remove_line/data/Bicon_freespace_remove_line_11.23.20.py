# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:00:15 2019

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#import frequency and lumped port volatge from text files, and convert imaginary LPV to real
com9mmFilename = "Bicon_freespace_remove_line_11.23.20.100ishpoints8.94mmGap.txt"
com9mmFreq = np.loadtxt(com9mmFilename, skiprows = 5, usecols = 0)
com9mmAf = np.loadtxt(com9mmFilename, skiprows=5, usecols=1) + 5.6
#com5Af = comAf + 5.6

com4mmFilename = "Bicon_freespace_remove_line_11.23.20.200ishpoints4.001mmGap.txt"
com4mmFreq = np.loadtxt(com4mmFilename, skiprows = 5, usecols = 0)
com4mmAf = np.loadtxt(com4mmFilename, skiprows=5, usecols=1) + 5.6
#com4mm5Af = comAf + 5.6

comFilename = "Bicon_freespace_remove_line_11.23.20.100ishpoints8.94mmGapAir.txt"
comAirFreq = np.loadtxt(comFilename, skiprows = 5, usecols = 0)
comAirAf = np.loadtxt(comFilename, skiprows=5, usecols=1) + 5.6
#com5AirAf = comAf + 5.6

comTlFilename = "Bicon_freespace_AF_9.28.txt"
comTlFreq = np.loadtxt(comTlFilename, skiprows = 5, usecols = 0)
comTlAf = np.loadtxt(comTlFilename, skiprows=5, usecols=1) + 5.6
#com5TlAf = comTlAf + 5.6

com3mmFilename = "Bicon_freespace_remove_line_11.23.20.100ishpoints2.94mmGap.txt"
com3mmFreq = np.loadtxt(com3mmFilename, skiprows = 5, usecols = 0)
com3mmAf = np.loadtxt(com3mmFilename, skiprows=5, usecols=1) + 5.6
#com5TlGAf = comTlGAf + 5.6

cstFilename = "biconFreespace.cst"
cstFreq = np.loadtxt("paulFreq.txt", usecols=0)
cstAf = np.loadtxt("biconFreespaceCST.txt", usecols=0)

manFilename = "Bicon_AntennaFactor10m_manufacturer.txt"
manFreq = np.loadtxt(manFilename, delimiter=',', usecols=0)
manAf = np.loadtxt(manFilename, delimiter=',', usecols=1)


#plot AF versus frequency
plt.figure(1)

plt.plot(com3mmFreq, com3mmAf, label = "p = 2.94mm")
plt.plot(com4mmFreq, com4mmAf, label = "p = 4.001mm")
plt.plot(com9mmFreq, com9mmAf, label = "p = 8.94mm")

#plt.plot(comTlFreq, com5TlAf, label = "comsol (with trans line) + 5.6db")
#plt.plot(cstFreq, cstAf, label = "cst")
#plt.plot(manFreq, manAf, label = "manufacturer")
plt.title('AF, bicon in free space NO line, 180 ohms, comsol, p = port gap (mm), +5.6dB included')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()

#plot AF versus frequency
plt.figure(2)

plt.plot(com3mmFreq, com3mmAf, label = "comsol NO line p=2.94mm")
#plt.plot(com9mmFreq, com9mmAf, label = "comsol NO line p=8.94mm")
plt.plot(com4mmFreq, com4mmAf, label = "comsol NO line p=4.001mm")
plt.plot(comTlFreq, comTlAf, label = "comsol WITH line p = 3.7mm")
plt.plot(cstFreq, cstAf, label = "cst, NO line p = 4mm")
plt.plot(manFreq, manAf, label = "manufacturer")
plt.title('AF, bicon in free space, 180 ohms, p = port gap (mm)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()

plt.figure(3)

#plt.plot(comFreq, comAf - comAirAf, label = "comsol NO line p=12.24mm")

plt.title('AF, bicon in free space, 180 ohms, p = port gap (mm)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()

