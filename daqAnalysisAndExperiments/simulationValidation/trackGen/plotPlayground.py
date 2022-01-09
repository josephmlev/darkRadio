#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:02:58 2021

@author: joseph
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.close('all')

startFreq = 5 * 10**6 #Hz
stopFreq = 300 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq/1e6, stopFreq/1e6, nPoints)

df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4 = pd.read_pickle('30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl')

df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4 = pd.read_pickle('30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl')

df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2 = pd.read_pickle('30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2.pkl')

df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2 = pd.read_pickle('30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2.pkl')

df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4 = pd.read_pickle('30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl')

df_30cmDipBananna_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4 = pd.read_pickle('30cmDipBananna_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl')

df_30cmDipBananna_noChokes_60N60E57V_20dBattAtFeed_rbw10k_DANLn100pm3dB_avg4 = pd.read_pickle('30cmDipBananna_noChokes_60N60E57V_20dBattAtFeed_rbw10k_DANLn100pm3dB_avg4.pkl')

df_30cmDipBananna_5chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4 = pd.read_pickle('30cmDipBananna_5Chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl')

comsolFreq, comsolPower = np.loadtxt('../comsolOutput/30cmDip_60_60_57_power_fullFeature_0.5MHzRes_12.25.21.txt', skiprows = 5, unpack = True)

comsolFreqCables, comsolPowerCables = np.loadtxt('../comsolOutput/30cmDip_60_60_57_power_addCables_fullFeature_0.25MHzRes_12.29.21.txt', skiprows = 5, unpack = True)

#Raw_2chokesDiff_v2 = df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw130cmDipBananna_5Chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl'0k_DANLn100pm3dB_avg4_v2['Raw Antenna Power (dBm)'] - df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)']
'''
plt.figure()
plt.title('No chokes Raw power v1 - v2')
plt.plot(freqArr, Raw_2chokesDiff_v2)

plt.figure()
plt.title('2 chokes Raw power v1 vs v2')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'], 'r')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2['Raw Antenna Power (dBm)'])
'''



Raw_noChokesDiff_v2 = df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'] - df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2['Raw Antenna Power (dBm)']

plt.figure()
plt.title('No chokes Raw power v1 - v2')
plt.plot(freqArr, Raw_noChokesDiff_v2)

plt.figure()
plt.title('No chokes Raw power v1 vs v2')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2['Raw Antenna Power (dBm)'], 'r')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'])

plt.figure()
plt.title('No chokes Matching Loss v1 vs v2')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2['Matching Loss'], 'r')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Matching Loss'])

plt.figure()
plt.title('No chokes NOrmilized power v1 vs v2')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2['Normilized Antenna Power (dBm)'], 'r')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'])

plt.figure()
plt.title('2 chokes N0rmilized power v1 vs v2')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2['Normilized Antenna Power (dBm)'], 'r')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'])

plt.figure()
plt.title('No chokes vs 2 vs All,  N0rmilized power')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'none')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = '2')
plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'all')
plt.legend()


plt.figure()
plt.title('No chokes vs 2 vs All, raw power')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'], label = 'none')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'], label = '2')
plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'], label = 'all')
plt.legend()

plt.figure()
plt.title('No chokes vs 2 vs All, Mismatch Loss')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Matching Loss'], label = 'none')
plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Matching Loss'], label = '2')
plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Matching Loss'], label = 'all')
plt.legend()


plt.figure()
plt.title('Raw power, bananna attenuator at feed vs on spec analizer')
plt.plot(freqArr, df_30cmDipBananna_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'], label = 'spec')
plt.plot(freqArr, df_30cmDipBananna_noChokes_60N60E57V_20dBattAtFeed_rbw10k_DANLn100pm3dB_avg4['Raw Antenna Power (dBm)'], label = 'at feed')
plt.legend()


plt.figure()
plt.title('Raw power, bananna attenuator at feed vs on spec analizer')
plt.plot(freqArr, df_30cmDipBananna_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'spec')
plt.plot(freqArr, df_30cmDipBananna_noChokes_60N60E57V_20dBattAtFeed_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'at feed')
plt.legend()


plt.figure()
plt.title('Normilized power, bananna vs clip no choke not att.')
plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'clip')
plt.plot(freqArr, df_30cmDipBananna_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'bananna')
plt.legend()




plt.figure()
plt.title('No chokes vs 2 vs All chokes vs COMSOL, Normilized Power')
#plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'none')
#plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = '2')
#plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_D30cmDipBananna_5Chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4.pkl'ANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'All chokes, Normilized')
plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'].rolling(500).mean(), label = 'All chokes, Normilized, rolling avg')
plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'All chokes, Normilized, rolling avg')
plt.plot(comsolFreq, comsolPower, label = 'Comsol')
plt.legend()

plt.figure()
plt.title('No chokes vs 2 vs All chokes vs COMSOL, Normilized Power')
#plt.plot(freqArr, df_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'none')
#plt.plot(freqArr, df_30cmDipClip_2chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = '2')
#plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'All chokes, Normilized')
plt.plot(freqArr, df_30cmDipBananna_5chokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'Bananna')
plt.plot(freqArr, df_30cmDipClip_allChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4['Normilized Antenna Power (dBm)'], label = 'clip')
plt.plot(comsolFreq, comsolPower, label = 'Comsol')
plt.legend()


plt.figure()
plt.plot(comsolFreqCables, comsolPowerCables, label = 'cables')
plt.plot(comsolFreq, comsolPower, label = 'no cables')
plt.legend()











































