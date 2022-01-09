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
Filename = "biconInBoxlpv_v2.txt"
freqOld = np.loadtxt(Filename, skiprows = 5, usecols = 0)
lpvOld = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "EfieldInBoxCutPoint_v2.txt"
ExOld = np.loadtxt(Filename, skiprows=5, usecols=1)

afOld = 20*np.log10(ExOld/lpvOld)


plt.figure()
plt.plot(freqOld,lpvOld)
plt.title('LPV')
plt.xlabel('Freq (MHz)')
plt.ylabel('Lumped port voltage')


plt.figure()
plt.plot(freqOld, ExOld)
plt.title('Ex')
plt.xlabel('Freq (MHz)')
plt.ylabel('E_x field')

plt.figure()
plt.plot(freqOld, afOld)
plt.title('AF')
plt.xlabel('Freq (MHz)')
plt.ylabel('20 log(Ex/LPV)')
         


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


#plot comsol vs cst AF

plt.figure()
plt.legend()
plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')      
plt.plot(cstFreq, cstAf, label = "cst 20 log(mag(Ex/lpv)) + 5.6")
plt.plot(freq, af4, label = "comsol 20 log(mag(Ex/lpv)) + 5.6")
plt.legend()

#plot E field in empty room comsol vs cst
plt.figure()
plt.plot(cstEFreq, cstE, label = 'cst')
plt.title('X component of E field at center of room, dB (1000 pts)')
plt.plot(freq, 20*np.log10(Ex), label = 'comsol 20log(E_x) for 1V/m')
#plt.plot(freq, 20*np.log10(Ex100), label = 'comsol 20log(E_x) for 100V/m')
plt.xlabel('Frequency [MHz]')
plt.ylabel('E_x')
plt.legend()

#plot lpv comsol vs cst
plt.figure()
plt.plot(cstLpvFreq, cstLpv, label = '20*log(mag(LPV)) (cst)')
plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV))(comsol)')
plt.ylabel('20*log(LPV)')
plt.xlabel('Frequency (MHz)')
plt.title('Lumped port voltage bicon in no res room (1000 pts)')
plt.legend()

#plot lpv comsol vs difference = np.abs(af4 - cstAf)cst SHIFTED
plt.figure()
plt.plot(cstLpvFreq, cstLpv + 170, label = '20*log(mag(LPV)) + 170 (cst) ' )
plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV)) (comsol)' )
plt.ylabel('20*log(LPV)')
plt.xlabel('Frequency (MHz)')
plt.title('Lumped port voltage bicon in no res room (1000 pts)')
plt.legend()

plt.figure()
plt.plot(freq, af - af4)

plt.figure()
plt.plot(freq,af, label = 'abs(AF)')
plt.legend()
plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
frame2=fig1.add_axes((.1,.1,.8,.2))        

plt.legend()


'''



