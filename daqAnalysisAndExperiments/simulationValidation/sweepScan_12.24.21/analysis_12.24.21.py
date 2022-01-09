#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:00:34 2021

@author: dark-radio
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.interpolate import interp1d


def interpDf(dfPath, freqArr):
    freq = pd.read_pickle(dfPath).index.values
    power = pd.read_pickle(dfPath)['Power (dBm)'].to_numpy()
    interpPowerArr = interp1d(freq, power)
    return interpPowerArr(freqArr)

def interpTxt(txtPath, freqArr):
    vswrfreq, vswr= np.loadtxt(txtPath, delimiter = ',', unpack = True)
    interpVswrArr = interp1d(vswrfreq,vswr)
    return interpVswrArr(freqArr)

def interpComsolTxt(txtPath, freqArr):
    freq = np.loadtxt(txtPath, skiprows = 5, usecols = 0)
    value = np.loadtxt(txtPath, skiprows = 5, usecols = 1)
    interpArr = interp1d(freq,value)
    return interpArr(freqArr)


def processPowerDf(freqArr, interpPowerArr, interpFGenCalArr, interpRLArr):
    dfName = pd.DataFrame()

    dfName['Frequency (MHz)'] = freqArr
    dfName['Raw Antenna Power (dBm)'] = interpPowerArr
    dfName['Return Loss'] = interpRLArr
    dfName['F. Gen Output (dBm)'] = interpFGenCalArr
    dfName['Matching Loss'] = -10 * np.log10(1 - 10**(-dfName['Return Loss']/10)) 
    dfName['Power to Antenna'] = dfName['F. Gen Output (dBm)'] - dfName['Matching Loss']
    dfName['Normilized Antenna Power (dBm)'] = dfName['Raw Antenna Power (dBm)'] - dfName['Power to Antenna']
    return dfName

freqArr = np.arange(50,300,0.1)
    

dip52cm_60_60_57_PowerArr = interpDf('./52cmDip_pwr_n34dBm_60N_60W_57V.pkl', freqArr)
dip52cm_60_60_57_fGenCalArr = interpDf('./fGenCalHiRes.pkl', freqArr)
dip52cm_60_60_57_RLArr = interpTxt('52cmDip_RL_60N_60W_57V.txt', freqArr)

dip52cm_60_60_57_powerDf = processPowerDf(freqArr, dip52cm_60_60_57_PowerArr, dip52cm_60_60_57_fGenCalArr, dip52cm_60_60_57_RLArr)




dip30cm_60_60_57_PowerArr = interpDf('./30cmDip_pwr_n20dbm_60N_60W_57V.pkl', freqArr)
dip30cm_60_60_57_fGenCalArr = interpDf('./fGenCal_30cmDip_n20dbm_60N_60W_57V.pkl', freqArr)
dip30cm_60_60_57_RLArr = interpTxt('30cmDip_RL_60N_60W_57V.txt', freqArr)

dip30cm_60_60_57_powerDf = processPowerDf(freqArr, dip30cm_60_60_57_PowerArr, dip30cm_60_60_57_fGenCalArr, dip30cm_60_60_57_RLArr)




'''
'Frequency (MHz)', 'Raw Antenna Power (dBm)', 'VSWR',
'F. Gen Output (dBm)', 'Reflection Coefficient', 'Return Loss (dB)',
'Power to Antenna', 'Normilized Antenna Power (dBm)'
'''
plt.close('all')


plt.figure()
plt.title('Normilized Antenna Output Power')
plt.plot(freqArr, dip30cm_60_60_57_powerDf['Normilized Antenna Power (dBm)'], label = '30cm dip')
plt.plot(freqArr, dip52cm_60_60_57_powerDf['Normilized Antenna Power (dBm)'], label = '52cm dip')
plt.legend()
plt.xlabel('Freq (MHz)')
plt.ylabel('Power (dBm)')

plt.figure()
plt.title('Raw Antenna Output Power')
plt.plot(freqArr, dip30cm_60_60_57_powerDf['Raw Antenna Power (dBm)'], label = '30cm dip')
#plt.plot(freqArr, dip52cm_60_60_57_powerDf['Raw Antenna Power (dBm)'], label = '52cm dip')
plt.legend()
plt.xlabel('Freq (MHz)')
plt.ylabel('Power (dBm)')


plt.figure()
plt.title('Return Loss')
plt.plot(freqArr, dip30cm_60_60_57_powerDf['Return Loss'], label = '30cm dip')
plt.plot(freqArr, dip52cm_60_60_57_powerDf['Return Loss'], label = '52cm dip')
plt.legend()
plt.xlabel('Freq (MHz)')
plt.ylabel('dB')


plt.figure()
plt.title('Matching Loss')
plt.plot(freqArr, dip30cm_60_60_57_powerDf['Matching Loss'], label = '30cm dip')
plt.plot(freqArr, dip52cm_60_60_57_powerDf['Matching Loss'], label = '52cm dip')
plt.legend()
plt.xlabel('Freq (MHz)')
plt.ylabel('dB')


plt.figure()
plt.title('Normilized Antenna Output Power vs COMSOL')
plt.plot(freqArr, dip30cm_60_60_57_powerDf['Normilized Antenna Power (dBm)'], label = '30cm dip')
#plt.plot(freqArr, dip52cm_60_60_57_PowerArr + 20, label = '30cm dip + const 20dB')
#plt.plot(freqArr, interpComsolTxt('30cmDip_60_60_57_simulation_powerDbm_12.25.21.txt', freqArr), label = 'COMSOL')
plt.plot(freqArr, interpComsolTxt('30cmDip_60_60_57_simulation_powerDbm_hiRes_12.26.21.txt', freqArr), label = 'COMSOL')
plt.legend()
plt.xlabel('Freq (MHz)')
plt.ylabel('Power (dBm)')

'''
dip52CmFreq = pd.read_pickle('./52cmDip_pwr_n34dBm_60N_60W_57V.pkl').index.values
dip52CmPower = pd.read_pickle('./52cmDip_pwr_n34dBm_60N_60W_57V.pkl')['Power (dBm)'].to_numpy()
dip52CmInterp = interp1d(dip52CmFreq, dip52CmPower)

fGenCalFreq = pd.read_pickle('./fGenCalHiRes.pkl').index.values
fGenCalPower = pd.read_pickle('./fGenCalHiRes.pkl')['Power (dBm)'].to_numpy()
fGenCalInterp = interp1d(fGenCalFreq, fGenCalPower)

vswrfreq, vswr= np.loadtxt('52cmDip_vswr_60N_60W_57V.txt', delimiter = ',', unpack = True)
vswrInterp = interp1d(vswrfreq,vswr)

df_52cmDip_60N_60W_57V = pd.DataFrame()

df_52cmDip_60N_60W_57V['Frequency (MHz)'] = freqArr
df_52cmDip_60N_60W_57V['Dipole 52 CM Power (dBm)'] = dip52CmInterp(freqArr)
df_52cmDip_60N_60W_57V['VSWR'] = vswrInterp(freqArr)
df_52cmDip_60N_60W_57V['F. Gen Output (dBm)'] = fGenCalInterp(freqArr)
df_52cmDip_60N_60W_57V['Reflection Coefficient'] = (df_52cmDip_60N_60W_57V['VSWR'] - 1) /(df_52cmDip_60N_60W_57V['VSWR'] + 1)
df_52cmDip_60N_60W_57V['Return Loss (dB)'] = -10 * np.log10(df_52cmDip_60N_60W_57V['Reflection Coefficient'])
df_52cmDip_60N_60W_57V['Power to Antenna'] = df_52cmDip_60N_60W_57V['F. Gen Output (dBm)'] - df_52cmDip_60N_60W_57V['Return Loss (dB)']
df_52cmDip_60N_60W_57V['Normilized Dipole Power (dBm)'] = df_52cmDip_60N_60W_57V['Dipole 52 CM Power (dBm)'] - df_52cmDip_60N_60W_57V['Power to Antenna']

df_52cmDip_60N_60W_57V.plot('Frequency (MHz)', 'Normilized Dipole Power (dBm)')
'''
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 