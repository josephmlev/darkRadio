#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:08:01 2022

@author: joseph
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close('all')


coldFreq_1_31_22 = np.array([50,80,110,140,170,200, 230, 260, 290, 300])
coldNf_1_31_22 = np.array([.34, .29, .4, .43, .39, .51, .61, .62, .51, .48])

df_coldData = pd.read_csv('coldData_1-6-22.txt', sep= "		", header = None, skiprows=1)
df_coldData.columns = ['Frequency', 'Temp', 'Gain', 'NF', 'Noise Temp']

manFreq, manNf = np.loadtxt('PE15A1012_ManNF.txt', skiprows = 1, unpack = True, delimiter= ',')

aproxTempArr = np.array([296,296,296,296,296,296,208,208,208,208,208,208,165,165,165,165,165,165])
df_coldData['Aprox Temp'] = aproxTempArr

freqArr = np.array([50, 100,150,200,250,300])

group_df = df_coldData.set_index('Aprox Temp', append=True).swaplevel(1,0).sort_index(level=0)

plt.figure()
plt.title('Noise Temperture vs. Frequency')
plt.xlabel('Frequency (MHz))')
plt.ylabel('Noise Temperture (K)')

tempSet = [296, 208, 165]
label = [ '296 K', '208 K', '165K']

markerStyles = ['o', 'v', 'p', '*']
markerColors = ['blue', 'green', 'red', 'black']
for counter, tempSet in enumerate(tempSet):
    noiseTemps = [(10**(x/10.)-1)*296.5 for x in group_df.loc[tempSet]['NF']]
    plt.scatter(freqArr, noiseTemps, label = label[counter])
    plt.plot(freqArr, noiseTemps, '--')

plt.plot(coldFreq_1_31_22, (10**(coldNf_1_31_22/10.)-1)*296.5, '--')  
plt.scatter(coldFreq_1_31_22, (10**(coldNf_1_31_22/10.)-1)*296.5, label = '134 K (1/30/22)')
#plt.errorbar(coldFreq_1_31_22, (10**(coldNf_1_31_22/10.)-1)*296.5, ((10**(0.1/10.)-1)*296.5), 0)

plt.plot(manFreq, (10**(manNf/10.)-1)*300, '--')  
plt.scatter(manFreq, (10**(manNf/10.)-1)*300, label = 'Manufacturer Data')



plt.xlim(45, 310)
plt.ylim(20, 90)

plt.legend()