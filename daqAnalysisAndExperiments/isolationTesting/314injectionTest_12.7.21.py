# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:53:16 2021

@author: phys-simulation
"""

import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

freqArr = np.array([50, 150.01, 299.99])
connectordBArr = np.array([94.01, 86.47, 103.17]) * -1
openHoledBArr = np.array([94.73, 79.49, 105.06]) * -1
tapedBArr = np.array([97.96, 84.14, 104.33]) * -1

plt.figure()
plt.title('314 Isolation Test For Different Patch Pannel Configurations')
xplt.scatter(freqArr, connectordBArr, label = 'Connector')
plt.scatter(freqArr, openHoledBArr, label = 'Open Hole')
plt.scatter(freqArr, tapedBArr, label = 'Tape')
plt.ylabel('Difference (dB)')
plt.xlabel('Frequency (MHz)')
plt.legend()
