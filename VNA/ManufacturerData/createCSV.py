import numpy as np 
import pandas as pd  


def getData(dataFile):
	freqs = []
	data = []
	with open(dataFile, 'r') as f:
		for line in f:
			holder = line.split()
			freqs.append(float(holder[0])*1E6)
			data.append(float(holder[1]))

	return freqs, data


def convVSWR(arr):
	return 20*np.log10((arr - 1) / (arr + 1))

dataFiles = ['PE15A63000_Gain.txt', 'PE15A63000_VSWRInput.txt', 'PE15A63000_VSWROutput.txt']
saveFile = 'PE15A63000_Manufacturer.csv'
freqGain, dataGain = getData(dataFiles[0])
freqS11, dataS11 = getData(dataFiles[1])
freqS22, dataS22 = getData(dataFiles[2])

newFreqs = np.linspace(1E7, 1E9, 1002)

newDataGain = np.interp(newFreqs, freqGain, dataGain)
newDataS11 = convVSWR(np.interp(newFreqs, freqS11, dataS11))
newDataS22 = convVSWR(np.interp(newFreqs, freqS22, dataS22))


with open(saveFile, 'w') as f:
	f.write('Frequency,S11_Magnitude,S21_Magnitude,S22_Magnitude,\n')
	
	for val in zip(newFreqs, newDataS11, newDataGain, newDataS22):
		f.write(str(val[0]) + ',' + str(val[1]) + ',' + str(val[2]) + ',' + str(val[3]) + ',\n')




