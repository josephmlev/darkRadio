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


Filename = "empty_1e4.txt"
Ex_1e4 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "bicon_1e4.txt"
lpv_1e4 = np.loadtxt(Filename, skiprows=5, usecols=1)

af_1e4 = 20 * np.log10(Ex_1e4/lpv_1e4) + 5.6 


Filename = "empty_5.9e5.txt"
freq = np.loadtxt(Filename, skiprows = 5, usecols = 0)
Ex_5p9e5 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "bicon_5.9e5.txt"
lpv_5p9e5 = np.loadtxt(Filename, skiprows=5, usecols=1)

af_5p9e5 = 20 * np.log10(Ex_5p9e5/lpv_5p9e5) + 5.6 


Filename = "empty_5.9e6.txt"
Ex_5p9e6 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "bicon_5.9e6.txt"
lpv_5p9e6 = np.loadtxt(Filename, skiprows=5, usecols=1)

af_5p9e6 = 20 * np.log10(Ex_5p9e6/lpv_5p9e6) + 5.6 




Filename = "empty_5.9e12.txt"
Ex_5p9e12 = np.loadtxt(Filename, skiprows=5, usecols=1)

Filename = "bicon_5.9e5.txt"
lpv_5p9e12 = np.loadtxt(Filename, skiprows=5, usecols=1)

af_5p9e12 = 20 * np.log10(Ex_5p9e12/lpv_5p9e12) + 5.6 



#difference 5.9e5 and 5.9e12
if 0:
    fig1 = plt.figure(1)
    difference = np.abs(af_5p9e5 - af_5p9e12)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.plot(freq,difference, 'r', label = 'abs(sigma2 - sigma1)')
    plt.legend()
    plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(freq, af_5p9e5, label = "sigma1 = 5.9e5")
    plt.plot(freq, af_5p9e12, label = "sigma2 = 5.9e12")
    plt.legend()
    
    
    plt.figure()
    plt.legend()
    plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')      
    plt.plot(freq, af_5p9e5, 'g', label = "sigma = 5.9e5")
    plt.plot(freq, af_5p9e12, 'c', label = "sigma = 5.9e12")
    plt.legend()

#difference 1e4 and 5.9e5
if 0:
    #af
    fig1 = plt.figure()
    difference = np.abs(af_1e4 - af_5p9e5)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.plot(freq,difference, 'r', label = 'abs(sigma2 - sigma1)')
    plt.legend()
    plt.title('Difference  AF (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(freq, af_1e4, label = "sigma1 = 1e4")
    plt.plot(freq, af_5p9e5, label = "sigma2 = 5.9e5")
    plt.legend()

    plt.figure()
    plt.legend()
    plt.title('AF (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')      
    plt.plot(freq, af_1e4,'b', label = "sigma = 1e4")
    plt.plot(freq, af_5p9e5, 'g', label = "sigma = 5.9e5")
    plt.legend()
    
    #E field 
    fig1 = plt.figure()
    difference = np.abs(20*np.log10(Ex_1e4) - 20*np.log10(Ex_5p9e5))
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.plot(freq,difference, 'r', label = 'abs(sigma2 - sigma1)')
    plt.legend()
    plt.title('Difference E field (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('20 log (Ex)')
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(freq, 20*np.log10(Ex_1e4), label = "sigma1 = 1e4")
    plt.plot(freq, 20*np.log10(Ex_5p9e5), label = "sigma2 = 5.9e5")
    plt.legend()
    
    plt.figure()
    plt.legend()
    plt.title('E field (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('20 log (Ex)')      
    plt.plot(freq, 20*np.log10(Ex_1e4), 'b', label = "sigma = 1e4")
    plt.plot(freq, 20*np.log10(Ex_5p9e5), 'g', label = "sigma = 5.9e5")
    plt.legend()
    
    
    
#difference 1e4 and 5.9e6
if 1:
    #af
    fig1 = plt.figure()
    difference = np.abs(af_1e4 - af_5p9e6)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.plot(freq,difference, 'r', label = 'abs(sigma2 - sigma1)')
    plt.legend()
    plt.title('Difference AF (conductivity) ')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(freq, af_1e4, label = "sigma1 = 1e4")
    plt.plot(freq, af_5p9e6, label = "sigma2 = 5.9e6")
    plt.legend()

    plt.figure()
    plt.legend()
    plt.title('AF (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')      
    plt.plot(freq, af_1e4,'b', label = "sigma = 1e4")
    plt.plot(freq, af_5p9e6, 'g', label = "sigma = 5.9e6")
    plt.legend()
    
    #E field 
    fig1 = plt.figure()
    difference = np.abs(20*np.log10(Ex_1e4) - 20*np.log10(Ex_5p9e6))
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.plot(freq,difference, 'r', label = 'abs(sigma2 - sigma1)')
    plt.legend()
    plt.title('Difference E field (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('20 log (Ex)')
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(freq, 20*np.log10(Ex_1e4), label = "sigma1 = 1e4")
    plt.plot(freq, 20*np.log10(Ex_5p9e6), label = "sigma2 = 5.9e6")
    plt.legend()
    
    plt.figure()
    plt.legend()
    plt.title('E field (conductivity)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('20 log (Ex)')      
    plt.plot(freq, 20*np.log10(Ex_1e4), 'b', label = "sigma = 1e4")
    plt.plot(freq, 20*np.log10(Ex_5p9e6), 'g', label = "sigma = 5.9e6")
    plt.legend()



#plot AF all
if 0:
    plt.figure()
    plt.legend()
    plt.title('AF bicon in box (no res, galvanized steel, 1000 pts)')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('AF')      
    plt.plot(freq, af_1e4,'b', label = "sigma = 1e4")
    plt.plot(freq, af_5p9e5, 'g', label = "sigma = 5.9e5")
    plt.plot(freq, af_5p9e6, 'r', label = "sigma = 5.9e6")
    plt.plot(freq, af_5p9e12, 'c', label = "sigma = 5.9e12")
    plt.legend()

#plot E all
if 0:
    plt.figure()
    plt.plot(freq, 20*np.log10(Ex_1e4), 'b', label = 'sigma = 1e4')
    plt.plot(freq, 20*np.log10(Ex_5p9e5), 'g', label = 'sigma = 5.9e5')
    plt.plot(freq, 20*np.log10(Ex_5p9e6), 'r', label = 'sigma = 5.9e6')
    plt.plot(freq, 20*np.log10(Ex_5p9e12), 'c', label = 'sigma = 5.9e12')
    plt.title('X component of E field at center of room, dB (1000 pts)')
    #plt.plot(freq, 20*np.log10(Ex), label = 'comsol 20log(E_x) for 1V/m')
    #plt.plot(freq, 20*np.log10(Ex100), label = 'comsol 20log(E_x) for 100V/m')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('E_x')
    plt.legend()





#plot lpv comsol vs cst
if 0:
    plt.figure()
    plt.plot(cstLpvFreq, cstLpv, label = '20*log(mag(LPV)) (cst)')
    plt.plot(freq, 20*np.log10(lpv), label = '20*log(mag(LPV))(comsol)')
    plt.ylabel('20*log(LPV)')
    plt.xlabel('Frequency (MHz)')
    plt.title('Lumped port voltage bicon in no res room (1000 pts)')
    plt.legend()

#plot lpv comsol vs cst SHIFTED
if 0:
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






