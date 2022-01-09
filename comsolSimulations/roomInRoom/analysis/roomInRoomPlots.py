# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:32:58 2021

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

plt.close('all')

featFreq = np.loadtxt("cstFeatureAF1.txt", usecols=0)
#featFreq = featFreq[137:]
featAf = np.loadtxt("cstFeatureAF1.txt", usecols=1) #point avg
#featAfS = featAf[137:]

filename = "cst_af_noRes_12.21.20.txt"
noResFreq = np.loadtxt(filename, usecols = 0, delimiter = ',')
#noResFreq = noResFreq[137:]
noResAf = np.loadtxt(filename, usecols = 1)
#noResAfS = noResAf[137:]

filename = "ExRoomInRoom_000point_9.30.21.txt"
smallFreq = np.loadtxt(filename, skiprows = 5, usecols = 0)
#smallFreqS = smallFreq[137:]
smallEx = np.loadtxt(filename, skiprows = 5, usecols = 1)

smallLpv = np.loadtxt("LPVRoomInRoom_9.30.21.txt", skiprows = 5, usecols = 1)

smallAf = 20 * np.log10(smallEx/smallLpv) + 5.6
#smallAfS = smallAf[137:]
#noResAf[220] = 1000 #test effect of large peaks

Filename = "EfieldInBoxCutPoint_v2.txt"
noResFreqCom = np.loadtxt(Filename, skiprows = 5, usecols = 0)
noResExCom = np.loadtxt(Filename, skiprows = 5, usecols = 1)

Filename = "biconInBoxlpv_v2.txt"
lpvCom = np.loadtxt(Filename, skiprows = 5, usecols=1)

noResAfCom = 20 * np.log10(Ex_v2/lpv_v2) + 5.6


w = 101 #window to average 
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
featAfAvg = moving_average(featAf, w)
noResAfAvg = moving_average(noResAf, w)
smallAfAvg =  moving_average(smallAf, w)

print(noResAf.mean(), 'mean AF no res')
print(featAf.mean(), 'mean AF feature')
print(smallAf.mean(), 'mean AF room in room')
print('')
print(noResAf[137:].mean(), 'mean AF no res, 50-300MHz')
print(featAf[137:].mean(), 'mean AF feature, 50-300MHz')
print(smallAf[137:].mean(), 'mean AF room in room, 50-300MHz')
print('')
print(np.median(noResAf), 'median AF no res')
print(np.median(featAf), 'median AF feature')
print(np.median(smallAf), 'median AF room in room')

#featured cst vs no res cst (avg)
plt.figure()
#plt.plot(featFreq, featAf, label = 'Featured Room')
plt.plot(featFreq[(w-1)//2:-(w-1)//2], featAfAvg, label = 'Featured Room, CST (running average)')
plt.plot(featFreq[(w-1)//2:-(w-1)//2], noResAfAvg, label = 'No Res Room, CST (running average)')
#plt.plot(noResFreq, noResAf, label = 'No Res Room')
plt.title('No Res (Full Size) vs Featured Room, Running Average AF  (window size = %f MHz)' % (w * .2902))
plt.xlabel('freq (MHz)')
plt.ylabel('AF (10log10(E/V) + 5.6)')
plt.legend()


#featured cst vs no res cst
plt.figure()
plt.plot(featFreq, featAf, label = 'Featured Room, CST')
plt.plot(noResFreq, noResAf, label = 'No Res Room, CST')
plt.title('No Res (Full Size) vs Featured Room, AF')
plt.xlabel('freq (MHz)')
plt.ylabel('AF (10log10(E/V) + 5.6)')
plt.legend()
plt.legend()


#featured cst vs room in room comsol avg
plt.figure()
#plt.plot(featFreq, featAf, label = 'Featured Room')
plt.plot(featFreq[(w-1)//2:-(w-1)//2], featAfAvg, label = 'Featured Room (running average), CST')
plt.plot(smallFreq[(w-1)//2:-(w-1)//2], smallAfAvg, label = 'Room in Room (running average), Comsol')
#plt.plot(noResFreq, noResAf, label = 'No Res Room')
plt.title('Room in Room vs Featured Room, Running Average AF (window size = %f MHz)' % (w * .2902))
plt.xlabel('freq (MHz)')
plt.ylabel('AF (10log10(E/V) + 5.6)')
plt.legend()

#featured cst vs room in room comsol avg
plt.figure()
plt.plot(featFreq, featAf, label = 'Featured Room, CST, 1000pts')
plt.title('Room in Room vs Featured Room, AF')
plt.plot(smallFreq, smallAf, label = 'Room in Room AF, Comsol, 1000pts')
#plt.plot(noResFreq, noResAf, label = 'No Res Room, CST')
plt.xlabel('freq (MHz)')
plt.ylabel('AF (10log10(E/V) + 5.6)')
plt.legend()


plt.figure()
plt.title('Room in Room vs No Res (Full Size), AF')
plt.plot(smallFreq, smallAf, label = 'Room in Room AF, Comsol, 1000pts')
#plt.plot(noResFreq, noResAf, label = 'No Res Room, CST')
plt.plot(noResFreqCom, noResAfCom, label = 'No Res Room, Comsol, 500pts')
plt.xlabel('freq (MHz)')
plt.ylabel('AF (10log10(E/V) + 5.6)')
plt.legend()

plt.figure()
plt.title('AF Difference (Room in Room vs Full Size No Res)')
plt.plot(smallFreq, smallAf - noResAf, label = 'Room in Room AF - No Res AF')
plt.xlabel('freq (MHz)')
plt.ylabel('AF (10log10(E/V) + 5.6)')
plt.legend()

plt.figure()
plt.title('Ex')
plt.plot(smallFreq, 20 * np.log10(smallEx), label = 'Room in Room Ex')
plt.xlabel('freq (MHz)')
plt.ylabel('20 log10(Ex)')
plt.legend()

'''
plt.figure()
plt.plot(noResFreq, noResAf-featAf, label = 'No Res Room')
plt.legend()
'''