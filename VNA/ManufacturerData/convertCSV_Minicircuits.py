import numpy as np


import numpy as np 
import pandas as pd  


def convVSWR(arr):
	return 20*np.log10((arr - 1) / (arr + 1))

dataFile = 'ZX60_63GLN_AllData.txt'
saveFile = 'ZX60_63GLN_Manufacturer.csv'


freqs = []
gain = []
inputVSWR = []
outputVSWR = []
directivity = []


with open(dataFile, 'r') as f:
	for line in f:
		holder = line.split()
		freqs.append(float(holder[0])*1E6)
		gain.append(float(holder[1]))
		directivity.append(float(holder[2]))
		inputVSWR.append(float(holder[3]))
		outputVSWR.append(float(holder[4]))

s21 = np.asarray(directivity) - np.asarray(gain)
s11 = convVSWR(np.asarray(inputVSWR))
s22 = convVSWR(np.asarray(outputVSWR))

newFreqs = np.linspace(18E8, 6E9, 1002)

newGain = np.interp(newFreqs, freqs, gain)
newS11 = (np.interp(newFreqs, freqs, s11))
newS22 = (np.interp(newFreqs, freqs, s22))
newS12 = np.interp(newFreqs, freqs, s21)

with open(saveFile, 'w') as f:
	f.write('Frequency,S11_Magnitude,S12_Magnitude,S21_Magnitude,S22_Magnitude,\n')
	
	for val in zip(newFreqs, newS11, newS12, newGain, newS22):
		f.write(str(val[0]) + ',' + str(val[1]) + ',' + str(val[2]) + ',' + str(val[3]) + ',' + str(val[4]) + ',\n')




