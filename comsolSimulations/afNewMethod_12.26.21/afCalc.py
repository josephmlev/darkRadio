#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 18:39:18 2021

@author: dradmin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

#import frequency and lumped port volatge from text files
#assumes measured at same frequencies 

df_empty_box = pd.DataFrame()
'''
Filename = "absLPV_biconAndDipole_12.26.21.txt"
df_empty_box['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
df_empty_box['LPV'] = np.loadtxt(Filename, skiprows = 5, usecols = 1)
'''
Filename = "absEx_52CutPoints_emptyBox_1VmDriveWall_12.26.21.txt"
df_empty_box['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
pointArr = np.loadtxt(Filename, usecols=range(1,53), skiprows=4, max_rows = 1, delimiter = 'abs(emw.Ex) (V/m), Point: ', dtype = 'str')
for i, point in enumerate(pointArr):
    df_empty_box[point] = np.loadtxt(Filename, skiprows=5, usecols=i+1)


df_empty_box['Average Ex'] = df_empty_box.iloc[:,2:].mean(axis = 1)





df_bicon_box = pd.DataFrame()

Filename = "absLpv_biconBox_1VmDriveWall_12.26.21.txt"
df_bicon_box['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
df_bicon_box['LPV'] = np.loadtxt(Filename, skiprows = 5, usecols = 1)

Filename = "absEx_52CutPoints_biconBox_1VmDriveWall_12.26.21.txt"
df_bicon_box['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
pointArr = np.loadtxt(Filename, usecols=range(1,53), skiprows=4, max_rows = 1, delimiter = 'abs(emw.Ex) (V/m), Point: ', dtype = 'str')
for i, point in enumerate(pointArr):
    df_bicon_box[point] = np.loadtxt(Filename, skiprows=5, usecols=i+1)


df_bicon_box['Average Ex'] = df_bicon_box.iloc[:,2:].mean(axis = 1)

df_bicon_box['AF'] = 20 * np.log10(df_bicon_box['Average Ex']/df_bicon_box['LPV']) + 5.6 
#df_bicon_box['AF no avg'] = 20 * np.log10(df_empty_box['(-0.5, 0, 0.4)          ']/df_empty_box['LPV']) + 5.6 
#df_bicon_box['AF no avg2'] = 20 * np.log10(df_empty_box['(-0.1, 0, 0.4)          ']/df_empty_box['LPV']) + 5.6 



df_empty_ff = pd.DataFrame()

Filename = "absEx_52CutPoints_emptyFullFeature_1VmDriveWall_12.26.21.txt"
df_empty_ff['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
pointArr = np.loadtxt(Filename, usecols=range(1,53), skiprows=4, max_rows = 1, delimiter = 'abs(emw.Ex) (V/m), Point: ', dtype = 'str')
for i, point in enumerate(pointArr):
    df_empty_ff[point] = np.loadtxt(Filename, skiprows=5, usecols=i+1)

df_empty_ff['Average Ex'] = df_empty_ff.iloc[:,2:].mean(axis = 1)


df_bicon_ff = pd.DataFrame()

Filename = "absLpv_biconFullFeature_1VmDriveWall_12.26.21.txt"
df_bicon_ff['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
df_bicon_ff['LPV'] = np.loadtxt(Filename, skiprows = 5, usecols = 1)

Filename = "absEx_52CutPoints_biconFullFeature_1VmDriveWall_12.26.21.txt"
df_bicon_ff['Frequency'] = np.loadtxt(Filename, skiprows = 5, usecols = 0)
pointArr = np.loadtxt(Filename, usecols=range(1,53), skiprows=4, max_rows = 1, delimiter = 'abs(emw.Ex) (V/m), Point: ', dtype = 'str')
for i, point in enumerate(pointArr):
    df_bicon_ff[point] = np.loadtxt(Filename, skiprows=5, usecols=i+1)


df_bicon_ff['Average Ex'] = df_bicon_ff.iloc[:,2:].mean(axis = 1)

df_bicon_ff['AF'] = 20 * np.log10(df_bicon_ff['Average Ex']/df_bicon_ff['LPV']) + 5.6 
#df_bicon_ff['AF no avg'] = 20 * np.log10(df_empty_box['(-0.5, 0, 0.4)          ']/df_empty_box['LPV']) + 5.6 
#df_bicon_ff['AF no avg2'] = 20 * np.log10(df_empty_box['(-0.1, 0, 0.4)          ']/df_empty_box['LPV']) + 5.6 



#Free space AF
freq = np.loadtxt("bicon4mmRerunRealAF.txt", skiprows=5, usecols=0)
AF = np.loadtxt("bicon4mmRerunRealAF.txt", skiprows=5, usecols=1)
comAf = AF + 5.6

freqMan = np.loadtxt('Bicon_AntennaFactor10m_manufacturer.txt', delimiter = ',',  skiprows=0, usecols=0)
AFMan = np.loadtxt('Bicon_AntennaFactor10m_manufacturer.txt', delimiter = ',', skiprows=0, usecols=1)

cstFilename = "biconFreespace.cst"
cstFreq = np.loadtxt("paulFreq.txt", usecols=0)
cstAf = np.loadtxt("biconFreespaceCST.txt", usecols=0)




plt.figure()
plt.plot(freq, comAf, label='comsol, gap=4mm, (+5.6dB)')
plt.plot(cstFreq, cstAf, label = "cst, gap=4mm, (+5.6dB)")
plt.plot(freqMan, AFMan, label='manufacturer')
plt.title('Bicon Freespace Antenna Factor COMSOl vs CST vs Manufacturer')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log10(1/V)]')
plt.legend()


plt.figure()
plt.title('AF COMSOL Bicon Free Space vs. Bicon in Box')
plt.plot(freq[5:], comAf[5:], label='Bicon Free Space')
plt.plot(df_bicon_box['Frequency'], df_bicon_box['AF'], label = 'Bicon in Box')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log10(1/V)]')
plt.legend()


plt.figure()
plt.title('AF, Old vs New Method COMSOL Bicon in Box')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_bicon_box['Average Ex']/df_bicon_box['LPV']) + 5.6, label = 'E(with bicon)/V')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_empty_box['Average Ex']/df_bicon_box['LPV']) + 5.6, label = 'E(without bicon)/V')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()


plt.figure()
plt.title('AF, Old vs New Method COMSOL Bicon Full Feature Room')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_bicon_ff['Average Ex']/df_bicon_ff['LPV']) + 5.6, label = 'E(with bicon)/V')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_empty_box['Average Ex']/df_bicon_ff['LPV']) + 5.6, label = 'E(without bicon)/V')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF')
plt.legend()


plt.figure()
plt.title('AF, Box vs Full Feature')
plt.plot(df_bicon_ff['Frequency'], df_bicon_ff['AF'] , label = 'Full Feature')
plt.plot(df_bicon_box['Frequency'], df_bicon_box['AF'], label = 'Box')
plt.xlabel('Frequency [MHz]')
plt.ylabel('AF [20*log10(Ex/V)]')
plt.legend()

plt.figure()
plt.title('Ex Volume Average With and Without Bicon (in Box, dB)')
plt.plot(df_empty_box['Frequency'], 20*np.log10(df_empty_box['Average Ex']), label='Empty Box')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_bicon_box['Average Ex']), label = 'Bicon in Box')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Ex_dB (20*log10(abs(Ex)))')
plt.legend()


plt.figure()
plt.title('Ex Volume Average With and Without Bicon (in Box, Linear)')
plt.plot(df_empty_box['Frequency'], df_empty_box['Average Ex'], label='Empty Box')
plt.plot(df_bicon_box['Frequency'], df_bicon_box['Average Ex'], label = 'Bicon in Box')
plt.xlabel('Frequency [MHz]')
plt.ylabel('abs(Ex)')
plt.legend()

plt.figure()
plt.title('Ex Volume Average in Box vs. Full Feature, Linear')
plt.plot(df_empty_box['Frequency'], df_empty_box['Average Ex'], label='Empty Box')
plt.plot(df_empty_ff['Frequency'], df_empty_ff['Average Ex'], label = 'Empty Full Feature')
plt.xlabel('Frequency [MHz]')
plt.ylabel('abs(Ex)')
plt.legend()

plt.figure()
plt.title('Ex Volume Average vs LPV (in Box, dB)')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_bicon_box['Average Ex']), label='Ex')
plt.plot(df_bicon_box['Frequency'], 20*np.log10(df_bicon_box['LPV']), label = 'LPV')
plt.xlabel('Frequency [MHz]')
plt.ylabel('abs(Ex)')
plt.legend()

'''
df_empty_box['AF'] = 20 * np.log10(df_empty_box['Average Ex']/df_empty_box['LPV']) + 5.6 
df_empty_box['AF no avg'] = 20 * np.log10(df_empty_box['(-0.5, 0, 0.4)          ']/df_empty_box['LPV']) + 5.6 
df_empty_box['AF no avg2'] = 20 * np.log10(df_empty_box['(-0.1, 0, 0.4)          ']/df_empty_box['LPV']) + 5.6 


plt.figure()
plt.title('AF (bicon, full feature, tickle 30cm dipole 60W 60N 57V)')
plt.xlabel('Freq (MHz)')
plt.ylabel('20log10(Average_Ex/LPV) + 5.6' )
plt.plot(df_empty_box['Frequency'], df_empty_box['AF'])
#plt.plot(df_empty_box['Frequency'], df_empty_box['AF no avg'], label = 'no avg')
#plt.plot(df_empty_box['Frequency'], df_empty_box['AF no avg2'], label = 'no avg')

plt.figure()
plt.xlabel('Freq (MHz)')
plt.ylabel('Ex Avg (V/m')
plt.plot(df_empty_box['Frequency'], df_empty_box['Average Ex'])

plt.figure()
plt.xlabel('Freq (MHz)')
plt.plot(df_empty_box['Frequency'], df_empty_box['Average Ex'], label = 'Ex avg')
plt.legend()


plt.figure()
plt.title('Ex_average in dB')
plt.xlabel('Freq (MHz)')
plt.ylabel('dB')
plt.plot(df_empty_box['Frequency'], 10*np.log10(df_empty_box['Average Ex']), label = 'Ex avg')
plt.legend()
'''


















