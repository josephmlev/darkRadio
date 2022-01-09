# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 06:56:26 2021

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

freq = np.loadtxt("bicon4mmRerunRealAF.txt", skiprows=5, usecols=0)
AF = np.loadtxt("bicon4mmRerunRealAF.txt", skiprows=5, usecols=1)

freq2 = np.loadtxt('bicon8mm94AF.txt', skiprows=5, usecols=0)
AF2 = np.loadtxt('bicon8mm94AF.txt', skiprows=5, usecols=1)

freq3 = np.loadtxt('bicon2mm94AF.txt', skiprows=5, usecols=0)
AF3 = np.loadtxt('bicon2mm94AF.txt', skiprows=5, usecols=1)

freqMan = np.loadtxt('Bicon_AntennaFactor10m_manufacturer.txt', delimiter = ',',  skiprows=0, usecols=0)
AFMan = np.loadtxt('Bicon_AntennaFactor10m_manufacturer.txt', delimiter = ',', skiprows=0, usecols=1)

cstFilename = "biconFreespace.cst"
cstFreq = np.loadtxt("paulFreq.txt", usecols=0)
cstAf = np.loadtxt("biconFreespaceCST.txt", usecols=0)
                   

                   
                   
BAF = AF+5.6
BAF2 = AF2+5.6
BAF3 = AF3+5.6


plt.figure(1)

plt.plot(freq, BAF, label='comsol, gap=4mm, (+5.6dB)')
#plt.plot(freq2, BAF2, label='gap=8.94mm')
#plt.plot(freq3, BAF3, label='gap=2.94mm')
plt.plot(cstFreq, cstAf, label = "cst, gap=4mm, (+5.6dB)")
plt.plot(freqMan, AFMan, label='manufacturer')

plt.title('Bicon Freespace Antenna Factor COMSOl vs CST vs Manufacturer')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log10(1/V)]')
plt.legend()