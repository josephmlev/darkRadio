# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 23:12:17 2021

@author: phys-simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

cstFeatFreq = np.loadtxt("cstFeatureAF1.txt", usecols=0)
cstFeatAF1 = np.loadtxt("cstFeatureAF1.txt", usecols=1) #point avg
cstFeatAF2 = np.loadtxt("cstFeatureAF2.txt", usecols=1) #vol avg

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

compowerFreq = np.loadtxt('10m Compower AF-Table 1.csv', skiprows=1, usecols=0, delimiter=',')
compowerAf = np.loadtxt('10m Compower AF-Table 1.csv', skiprows=1, usecols=1,  delimiter=',')


#cacluclate af with high density 
Filename = "comsolExCutPointFullFeature_150-180mhz.txt"
fullFeatFreqDens = np.loadtxt(Filename, skiprows = 5, usecols = 0)
fullFeatExDens = np.loadtxt(Filename, skiprows = 5, usecols = 1)
fullFeatExDensCyl = np.loadtxt('comsolExCutPointFullFeature_150-180mhz_withCylinder.txt', skiprows = 5, usecols = 1) #cut point from model with cylinder
fullFeatLPVDens = np.loadtxt('comsolLPVFullFeature_150-180mhz.txt', skiprows = 5, usecols = 1)
fullFeatAfDens = 20 * np.log10(fullFeatExDens/fullFeatLPVDens) + 5.6
fullFeatAfDensCyl = 20 * np.log10(fullFeatExDensCyl/fullFeatLPVDens) + 5.6

#low dens
Ex4 = np.loadtxt(Filename, skiprows=5, usecols=1)
fullFeatFreq = np.loadtxt('comsolExCutPointFullFeature_55pts.txt', skiprows=5, usecols=0)
fullFeatEx = np.loadtxt('comsolExCutPointFullFeature_55pts.txt', skiprows=5, usecols=1)
fullFeatExSurf= np.loadtxt('comsolExSurfaceAvgFullFeature_55pts.txt', skiprows=5, usecols=1)
fullFeatLPV = np.loadtxt('comsolLPVFullFeature_55pts.txt', skiprows=5, usecols=1)

fullFeatAF = 20 * np.log10(fullFeatEx/fullFeatLPV) + 5.6
fullFeatAFSurf = 20 * np.log10(fullFeatExSurf/fullFeatLPV) + 5.6

#create arrays of mixed density
temp = np.append(fullFeatFreq[0:15], fullFeatFreqDens) 
fullFeatFreqMixed = np.append(temp, fullFeatFreq[21:]) 
temp = np.append(fullFeatEx[0:15], fullFeatExDens)
fullFeatExMixed = np.append(temp, fullFeatEx[21:]) 
temp = np.append(fullFeatLPV[0:15], fullFeatLPVDens)
fullFeatLPVMixed = np.append(temp, fullFeatLPV[21:])
fullFeatAFMixed = 20 * np.log10(fullFeatExMixed/fullFeatLPVMixed) + 5.6

plt.figure(2)
plt.plot(cstFeatFreq, cstFeatAF1, label = 'CST 1')
plt.plot(fullFeatFreq, fullFeatAFSurf, label = 'Comsol (55pts)')
plt.scatter(fullFeatFreq, fullFeatAFSurf, color = "red", label = 'Comsol (55pts)')
plt.title('Bicon AF in Full Feature room (CST 1)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()

'''
plt.figure(3)
plt.plot(comfreq, comAF2, label = 'low res AF')
plt.plot(noResfreq, noResAFSurf, label = 'no res AF')
plt.title('Comsol Low Res vs. No Res (surface average)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()
'''
#compare cst vol and point
plt.figure(4)
plt.plot(cstFeatFreq, cstFeatAF1, label = 'CST E point')
plt.plot(cstFeatFreq, cstFeatAF2,color = "green", label = 'CST E vol avg')
plt.title('CST 1 VS 2')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()

#comsol low dens vs cst
plt.figure(5)
plt.plot(cstFeatFreq, cstFeatAF2,color = "green", label = 'CST 2')
plt.scatter(fullFeatFreq, fullFeatAFSurf, color = "red", label = 'Comsol (55pts)')
plt.plot(fullFeatFreq, fullFeatAFSurf,color = "orange", label = 'Comsol (55pts)')
plt.title('Bicon AF in Full Feature room (CST 2)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()

#plot mixed density
plt.figure(6)
plt.plot(cstFeatFreq, cstFeatAF1,color = "green", label = 'CST')
plt.scatter(fullFeatFreq[0:15], fullFeatAF[0:15], color = "red", label = 'Comsol')
plt.scatter(fullFeatFreq[21:], fullFeatAF[21:], color = "red")
plt.plot(fullFeatFreqMixed, fullFeatAFMixed,color = "orange", label = 'Comsol')
plt.title('Bicon AF in Full Feature room, mixed point density (E-field point avg)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
x = [150, 180]
y = [60, 60]
plt.fill_between(x, y,  -30,
                 color = 'blue',
                 alpha = .2)
plt.legend()

#close up on high dens
plt.figure()
plt.plot(cstFeatFreq[483:586], cstFeatAF1[483:586],color = "green", label = 'CST')
plt.plot(fullFeatFreqDens, fullFeatAfDens, color = 'orange', label = 'Comsol')
plt.title('Bicon AF in Full Feature room, high point density (E-field point avg)')
plt.legend()

#comsol point avg with and without cylinder
plt.figure()
plt.title('Ex evaluated at point')
plt.plot(fullFeatFreqDens, fullFeatExDensCyl, label = "with cylinder")
plt.plot(fullFeatFreqDens, fullFeatExDens, label = "without")
plt.xlabel('Freq (MHz)')
plt.ylabel('Ex (V/m)')
plt.legend()

plt.figure()
plt.title('Ex evaluated at point (dB)')
plt.plot(fullFeatFreqDens, np.log10(fullFeatExDensCyl), label = "with cylinder")
plt.plot(fullFeatFreqDens, np.log10(fullFeatExDens), label = "without")
plt.xlabel('Freq (MHz)')
plt.ylabel('dB_Ex (log(V/m)')
plt.legend()

plt.figure()
plt.title('AF with and without cylinder')
plt.plot(fullFeatFreqDens, fullFeatAfDensCyl, label = "with cylinder")
plt.plot(fullFeatFreqDens, fullFeatAfDens, label = "without")
plt.xlabel('Freq (MHz)')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')
plt.legend()

plt.figure()
plt.plot(fullFeatFreq,fullFeatAF, label = 'Ex point')
plt.plot(fullFeatFreq,fullFeatAFSurf, label = 'surface avg Ex')
plt.plot(fullFeatFreqDens,fullFeatAfDens)
plt.xlim(150,180)
plt.legend()


#plot mixed density with AF
plt.figure()
plt.plot(cstFeatFreq, cstFeatAF1,color = "green", label = 'CST')

plt.plot(fullFeatFreqMixed, fullFeatAFMixed,color = "orange", label = 'Comsol')
plt.title('Bicon AF in Full Feature room, mixed point density (E-field point avg)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log(Ex/lpv)+5.6]')

plt.plot(compowerFreq, compowerAf, label = 'Compower Free Space AF')
plt.legend()
