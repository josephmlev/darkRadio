from cmath import rect
import glob
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import random 
import sys

def getManAntFac(fileName):
	freqs, amps = [], []
	with open(fileName, 'r') as f:
		for line in f:
			holder = line.split(',')
			freqs.append(float(holder[0]))
			amps.append(float(holder[1]))
	return freqs, amps 

def calcS11Complex(s11Mag, s11Phase):
	nprect = np.vectorize(rect)
	c = nprect(s11Mag, np.deg2rad(s11Phase))
	return c

def calcReflect(arr):
	arr = np.asarray(arr)
	return (arr - 1)/(arr + 1)


def getAllFiles(dirName):
	dataFiles = glob.glob(dirName + '/' + '*Pos*.csv' )
	return dataFiles 


def getSData(fileNames):
	freqs = []
	vswr = []
	reflec = []
	S11Mag = []
	S11Phase = []
	S11Complex = []
	for aFile in fileNames:
		with open(aFile, 'r') as f:
			df = pd.read_csv(f)
			freqs.append(np.asarray(df['Frequency'])/1E6)
			vswr.append(np.asarray(df['S11_VSWR']))
			reflec.append(calcReflect(vswr[-1]))
			S11Mag.append(10**(np.asarray(df['S11_Magnitude'])/20.))
			S11Phase.append(np.asarray(df['S11_Phase']))
			S11Complex.append(np.asarray(calcS11Complex(S11Mag[-1], S11Phase[-1])))
	
	return np.asarray(freqs), np.asarray(vswr), np.asarray(reflec), np.asarray(S11Mag), np.asarray(S11Phase), np.asarray(S11Complex) 

def calcQData(s11Complex, meanS11, freqs):
	num = np.mean(np.abs(s11Complex - meanS11)**2, axis = 0)
	den = (-1*np.abs(meanS11)**2 + 1)**2
	volume = 3.05 * 2.45 * 3.67
	wavelength = 3e2 / freqs[0]
	qVal = 8*(np.pi)**2*volume/wavelength**3*num/den
	return num, den, qVal

# Various font names
label_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'normal',
          'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 




dataDir = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data/'
dataDirRedone = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data2/'

fileNames = sorted(getAllFiles(dataDir))

fileNamesRedone = sorted(getAllFiles(dataDirRedone))



freqs, vswr, reflec, s11Mag, s11Phase, s11Complex = getSData(fileNames)
freqsRedone, vswrRedone, reflecRedone, s11MagRedone, s11PhaseRedone, s11ComplexRedone = getSData(fileNamesRedone)

meanS11 = np.mean(s11Complex, axis = 0)

meanS11Redone = np.mean(s11ComplexRedone, axis = 0)


num, den, qVal = calcQData(s11Complex, meanS11, freqs)
numRedone, denRedone, qValRedone = calcQData(s11ComplexRedone, meanS11Redone, freqsRedone)


medQVal = np.median(np.reshape(qVal[:-1], (-1, 25)), axis = 1)
medFreqs = np.median(np.reshape(freqs[0][:-1], (-1, 25)), axis = 1)


plt.plot(freqs[0], 10*np.log10(qVal), label = 'First Run')
plt.plot(freqsRedone[0], 10*np.log10(qValRedone), label = 'Second Run')
#plt.gca().set_yscale('log')

#plt.plot(medFreqs, 10*np.log10(medQVal), 'r--', label = 'Median Fit')

plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Q (dB)', **label_font)
plt.tight_layout()
plt.grid()
plt.legend(prop = legend_font)
#plt.savefig('CompositeQFirstAttempt_MedianFit.png', dpi = 100)
plt.show()

sys.exit(1)


plt.plot(freqs[0], 9.73/wavelength)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(r'Composite Antenna Factor (dBm$^{-1}$)', **label_font)
plt.grid()
#plt.savefig('CompositeAntennaFactor.png', dpi = 100)
plt.show()

plt.plot(freqs[0], 20*np.log10(9.73/wavelength/qVal**0.5))
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(r'Traditional Antenna Factor (dBm$^{-1}$)', **label_font)
plt.grid()
#plt.savefig('TraditionalAntennaFactor.png', dpi = 100)
plt.show()


sys.exit(1)
manAntFacFile = 'AB-900A_Fressapce_ComPower.txt'
manFreqs, manAFs = getManAntFac(manAntFacFile)

AFMeas = 20*np.log10(9.73/wavelength/qVal**0.5)
AFMed = np.median(np.reshape(AFMeas[:-1], (-1, 25)), axis = 1)
freqMed = np.median(np.reshape(freqs[0][:-1], (-1, 25)), axis = 1)


plt.plot(freqs[0], 20*np.log10(9.73/wavelength/qVal**0.5), label = 'In Room')
plt.plot(freqMed, AFMed, 'r--', label = 'Median Fit')

print(np.mean((9.73/wavelength/qVal**0.5)))
plt.plot(manFreqs, manAFs, label = 'Free space')
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(r'Traditional Antenna Factor (dBm$^{-1}$)', **label_font)
plt.grid()
plt.legend(prop = legend_font)
plt.savefig('TraditionalAntennaFactorAndManufacturer.png', dpi = 100)
plt.show()



#sys.exit(1)

#plt.plot(freqs[0], np.mean(vswr, axis = 0))
#plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
#plt.ylabel('VSWR (x:1)', **label_font)
#plt.legend(prop = legend_font)
#plt.show()


#for aVal in zip(freqs[:1], vswr[:1], labels[:1]):
#	plt.plot(aVal[0], aVal[1], label = aVal[2], alpha = 0.7)

plt.plot(freqs[0], np.mean(vswr, axis = 0))
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('VSWR (x:1)', **label_font)
plt.legend(prop = legend_font)
plt.savefig('MeanVSWR.png', dpi = 100)
plt.show()


#for aVal in zip(freqs, reflec, labels):
#	plt.plot(aVal[0], aVal[1], label = aVal[2], alpha = 0.7)

plt.plot(freqs[0], np.mean(reflec, axis = 0))
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(u'\u0393', **label_font)
plt.savefig('MeanReflectionCoefficient.png', dpi = 100)
plt.show()



#for val in zip(np.median(reflec, axis = 1), labels):
#	print('MEDIAN REFLECTION COEFFICIENT FOR ' + str(val[1]) + ': ' + str(round(val[0], 8)))

#print('MEAN REFLECTION COEFFICIENT: ' + str(round(np.mean([np.mean(reflec) for x in vswr]), 3)))
#print('MEDIAN REFLECTION COEFFICIENT: ' + str(round(np.median([np.median(reflec) for x in reflec]), 3)))

#plt.show()


meanData = np.mean(reflec, axis = 0)
medData = np.median(np.reshape(meanData[:-1], (-1, 25)), axis = 1)
medFreqs = np.median(np.reshape(freqs[0][:-1], (-1, 25)), axis = 1)

medDataInterp = np.interp(freqs[0], medFreqs, medData)
#plt.plot(freqs[0], [0] * len(meanData), label = 'Overall Mean')
#plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
#plt.ylabel('Difference (%)', **label_font)


totalReps = 1000
moves = []
stdTotal = []
maxTotal = []
meanTotal = []
for i in range(1, 30):
	meanArr = []
	maxArr = []
	stdArr = []

	for counter in range(totalReps):
		samp = random.sample(range(len(reflec)), i)
		subs = np.mean([reflec[x] for x in samp], axis = 0)
		deviation = np.abs((subs - medDataInterp))/medDataInterp*100
		meanArr.append(np.mean(deviation[1:-1]))
		maxArr.append(max(deviation[1:-1]))
		stdArr.append(np.std(deviation[1:-1]))
	moves.append(i)
	stdTotal.append(np.mean(stdArr))
	maxTotal.append(max(maxArr))
	meanTotal.append(np.mean(meanArr))
	print('FOR ' + str(i) + ' MOVES\n THE MEAN DEVIATION IS ' + str(round(np.mean(meanArr), 3)) + '%\n' \
		  + 'THE MAX DEVIATION IS: ' + str(round(max(maxArr), 3)) + '%\n' \
		  + 'THE STANDARD DEVIATION IS: ' + str(round(np.mean(stdArr), 3)) + '%')



plt.plot(moves, stdTotal, 'ro')
plt.xlabel('Number of Moves', labelpad = 15, **label_font)
plt.ylabel('Standard Deviation (%)', **label_font)
plt.savefig('StandardDeviation.png', dpi = 100)
plt.show()

plt.plot(moves, maxTotal, 'ro')
plt.xlabel('Number of Moves', labelpad = 15, **label_font)
plt.ylabel('Max Deviation (%)', **label_font)
plt.savefig('MaxDeviation.png', dpi = 100)
plt.show()

plt.plot(moves, meanTotal, 'ro')
plt.xlabel('Number of Moves', labelpad = 15, **label_font)
plt.ylabel('Mean Deviation (%)', **label_font)
plt.savefig('MeanDeviation.png', dpi = 100)
plt.show()

#samp2 = random.sample(range(len(reflec)), 1)
#subs2 = np.mean([reflec[x] for x in samp2], axis = 0)


#plt.plot(freqs[0], (subs1 - meanData)/meanData*100, label = 'Subsample 1')
#plt.plot(freqs[1], (subs2 - meanData)/meanData*100, label = 'Subsample 2')

#plt.legend(prop = legend_font)
#plt.show()
