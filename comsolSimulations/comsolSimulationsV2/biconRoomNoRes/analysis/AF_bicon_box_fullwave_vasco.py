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
Filename = "empty_box_6wall_Es_1VM_x_1000pts_.001error.txt"
freq = np.loadtxt(Filename, skiprows = 5, usecols = 0)
Ex = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "bicon_box_6wall_Es_1VM_x_1000pts_.001error.txt"
lpv = np.loadtxt(Filename, skiprows=5, usecols=1)

af = 20 * np.log10(Ex/lpv) + 5.6 

Filename = "empty_box_4wall_Es_1VM_x_1000pts_.001error.txt"
freq = np.loadtxt(Filename, skiprows = 5, usecols = 0)
Ex4 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "bicon_box_4wall_Es_1VM_x_1000pts_.001error.txt"
lpv4 = np.loadtxt(Filename, skiprows=5, usecols=1)

af4 = 20 * np.log10(Ex4/lpv4) + 5.6 

Filename = "CST_af_noRes_12.21.20.txt"
cstFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstAf = np.loadtxt(Filename, delimiter = ',', usecols=1)

Filename = "CST_nores_20log10(mag(lpv)).txt"
cstLpvFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstLpv = np.loadtxt(Filename, delimiter = ',', usecols=1)

Filename = "E_x_center_Db_CST.txt"
cstEFreq = np.loadtxt(Filename, delimiter = ',', usecols=0)
cstE = np.loadtxt(Filename, delimiter = ',', usecols=1)

vascFreq = np.loadtxt('EfieldInBoxCutPoint.txt', skiprows=5, usecols=0)
vascEx = np.loadtxt('EfieldInBoxCutPoint.txt', skiprows=5, usecols=1)
vascExSurf= np.loadtxt('surfaceAvg300cmEmptyBox.txt', skiprows=5, usecols=1)
vasclpv = np.loadtxt('biconInBoxlpv.txt', skiprows=5, usecols=1)
vascAF = 20 * np.log10(vascEx/vasclpv) + 5.6
vascAFSurf = 20 * np.log10(vascExSurf/vasclpv) + 5.6


Filename = "EfieldInBoxCutPoint_v2.txt"
freq_v2 = np.loadtxt(Filename, skiprows = 5, usecols = 0)
Ex_v2 = np.loadtxt(Filename, skiprows = 5, usecols = 1)

Filename = "biconInBoxlpv_v2.txt"
lpv_v2 = np.loadtxt(Filename, skiprows = 5, usecols=1)

af_v2 = 20 * np.log10(Ex_v2/lpv_v2) + 5.6




font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


'''
#plot AF versus frequency comsol vs cst. make difference sub plot
#plt.figure()
fig1 = plt.figure(1)
difference = np.abs(af4 - cstAf)
frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(freq,difference, 'r', label = 'abs(comsol AF - cst AF)')
plt.legend()
plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
frame2=fig1.add_axes((.1,.1,.8,.2))        
plt.plot(cstFreq, cstAf, label = "cst 20 log(mag(Ex/lpv)) + 5.6")
plt.plot(freq, af4, label = "comsol 20 log(mag(Ex/lpv)) + 5.6")
plt.legend()
'''
'''
#plt af comsol 4 wall vs 6 wall drive
fig1 = plt.figure(2)
difference = np.abs(af4 - af)
frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(freq,difference, 'r', label = 'abs(AF 4 wall -  AF 6 wall)')
plt.legend()
plt.title('AF bicon in box (no res, galvanized steel, 1000 pts) drive Ex on 4 vs 6 walls')
plt.ylabel('difference in AF')
frame2=fig1.add_axes((.1,.1,.8,.2))        
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.plot(freq, af, label = "comsol 20 log(mag(Ex/lpv)) + 5.6 (6 wall drive)")
plt.plot(freq, af4, label = "comsol 20 log(mag(Ex/lpv)) + 5.6 (4 wall drive")
plt.legend()
'''



#plot comsol vs cst AF
plt.figure()
plt.legend()
plt.title('AF bicon in box (no res, galvanized steel)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF =  20log(mag(Ex/lpv)) + 5.6')      
plt.plot(cstFreq, cstAf, label = "CST")
plt.plot(freq_v2, af_v2, label = 'Comsol')
#plt.plot(freq, af4, label = "old comsol" )
#plt.plot(vascFreq, vascAF, label = 'Vasco comsol 20 log(mag(Ex/lpv)) + 5.6')

plt.legend()

#plot E field in empty room comsol vs cst
plt.figure()
plt.plot(cstEFreq, cstE, label = 'cst')
plt.title('X component of E field at center of room, dB (1000 pts)')
plt.plot(freq, 20*np.log10(Ex), label = 'comsol 20log(E_x) for 1V/m')
#plt.plot(freq, 20*np.log10(Ex100), label = 'comsol 20log(E_x) for 100V/m')
plt.plot(vascFreq, 20*np.log10(vascEx), label='Vasco comsol 20log(E_x) for 1V/m')
plt.xlabel('Frequency [MHz]')
plt.ylabel('E_x')
plt.legend()

#plot lpv comsol vs cst
plt.figure()
plt.plot(cstLpvFreq, cstLpv, label = '20*log(mag(LPV)) (cst)')
plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV))(comsol)')
plt.plot(vascFreq, 20*np.log10(vasclpv), label = 'Vasco 20*log(mag(LPV))(comsol)')
plt.ylabel('20*log(LPV)')
plt.xlabel('Frequency (MHz)')
plt.title('Lumped port voltage bicon in no res room (1000 pts)')
plt.legend()

#plot lpv comsol vs cst SHIFTED
plt.figure()
plt.plot(cstLpvFreq, cstLpv + 170, label = '20*log(mag(LPV)) + 170 (cst) ' )
plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV)) (comsol)' )
plt.plot(vascFreq, 20*np.log10(vasclpv), label = 'Vasco 20*log(mag(LPV))(comsol)')
plt.ylabel('20*log(LPV)')
plt.xlabel('Frequency (MHz)')
plt.title('Lumped port voltage bicon in no res room (1000 pts)')
plt.legend()


#vasco cut point af vs surface avg af
plt.figure()
plt.legend()
plt.title('AF bicon in box (no res, structural steel, 500 pts)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.plot(vascFreq, vascAF, label = 'Vasco cut point 20 log(mag(Ex/lpv)) + 5.6')
#plt.plot(vascFreq, vascAFSurf, label = 'Vasco surface average 20 log(mag(Ex/lpv)) + 5.6')
plt.plot(freq_v2, af_v2, label = 'comsol cut point 20log(mag(Ex/lpv + 5.6')
plt.legend()


'''
plt.figure()
plt.plot(freq, af - af4)
'''
'''
plt.figure()
plt.plot(freq,af, label = 'abs(AF)')
plt.legend()
plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
frame2=fig1.add_axes((.1,.1,.8,.2))        

plt.legend()
'''





