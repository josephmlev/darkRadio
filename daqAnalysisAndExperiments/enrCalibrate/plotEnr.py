#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 22:38:08 2022

@author: joseph
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close('all')

df = pd.read_pickle('ENR_30avg_1.30.22.pkl')
freqArr = df['Frequency (MHz)']

manFreq = np.array([10, 50, 100, 200, 500])
manEnr = np.array([14.7, 15.5, 15.7, 16.1, 16.1])


plt.figure()
plt.title('Measured vs Manufacturer ENR')
plt.xlabel('Frequency (MHz))')
plt.ylabel('ENR')
plt.plot(freqArr[200:],df['log ENR'][200:].rolling(100).median(), label = 'Measured ENR')
plt.scatter(manFreq, manEnr, label = 'Manufacurer ENR', c = 'r')
plt.legend()
plt.xlim(30, 315)
plt.ylim(15.2, 18)