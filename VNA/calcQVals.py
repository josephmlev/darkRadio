from cmath import rect
import glob
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import random 
import sys


# Calculate the complex S11 components into polar form
def calcS11Complex(s11Mag, s11Phase):
	nprect = np.vectorize(rect)
	c = nprect(s11Mag, np.deg2rad(s11Phase))
	return c

# Calculate reflection coefficient from VSWR
def calcReflect(arr):
	arr = np.asarray(arr)
	return (arr - 1)/(arr + 1)


# Get all .csv files in directory
def getAllFiles(dirName):
	dataFiles = glob.glob(dirName + '/' + '*Pos*.csv' )
	return dataFiles 


# Catch-all function that returns frequencies, vswr, reflection coefficients, S11 magnitude,
# S11 phase, and S11 values written in a + bi form
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

# Calculate q value from formula usin dimensions of the room
def calcQData(s11Complex, meanS11, freqs):
	num = np.mean(np.abs(s11Complex - meanS11)**2, axis = 0)
	#num = (np.abs(s11Complex - meanS11)**2)
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

LIGHTSPEED = 3.0e8


dataDir = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data/'
dataDirRedone = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data2/'

# Redone corresponds to second run of 10 positions
fileNames = sorted(getAllFiles(dataDir))
fileNamesRedone = sorted(getAllFiles(dataDirRedone))


freqs, vswr, reflec, s11Mag, s11Phase, s11Complex = getSData(fileNames)
freqsRedone, vswrRedone, reflecRedone, s11MagRedone, s11PhaseRedone, s11ComplexRedone = getSData(fileNamesRedone)



meanS11 = np.mean(s11Complex, axis = 0)

meanS11Redone = np.mean(s11ComplexRedone, axis = 0)
plt.plot(freqs[0], 20*np.log10(np.abs(meanS11Redone)))
plt.xlabel('Frequency (Hz)', labelpad = 15, **label_font)
plt.ylabel('S11 (log)', **label_font)
plt.show()
sys.exit(1)

meanImpedanceRedone = np.abs(50 * (1 + meanS11Redone) / (1 - meanS11Redone))

num, den, qVal = calcQData(s11Complex, meanS11, freqs)
numRedone, denRedone, qValRedone = calcQData(s11ComplexRedone, meanS11Redone, freqsRedone)



# Plot frequencies and q values 
#plt.plot(freqs[0], (qVal), label = 'First Run')
plt.plot(freqsRedone[0], 10*np.log10(qValRedone), label = 'Second Run')

#plt.gca().set_yscale('log')


plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Q (dB)', **label_font)
plt.tight_layout()
plt.grid()
#plt.legend(prop = legend_font)
plt.savefig('CompositeQ_6-27-23', dpi = 100)

#plt.savefig('CompositeQFirstAttempt_MedianFit.png', dpi = 100)
plt.show()

antFac = 9.73 * freqsRedone[0] * 1e6 / LIGHTSPEED/np.sqrt(meanImpedanceRedone/50.)


antFac2 = 9.73 * freqsRedone[0] * 1e6 / LIGHTSPEED*np.sqrt(1 - np.abs(meanS11Redone))


plt.plot(freqsRedone[0], 20*np.log10(antFac))
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(r'Composite Antenna Factor (dBm$^{-1}$)', **label_font)
plt.tight_layout()
plt.grid()
plt.legend(prop = legend_font)

plt.savefig('CompositeAF_6-27-23', dpi = 100)
plt.show()


lowFreq = (np.argmin(np.abs(freqsRedone[0] - 50)))
highFreq = (np.argmin(np.abs(freqsRedone[0] - 300)))
plt.plot(freqsRedone[0][lowFreq:highFreq], (20*np.log10(antFac2) - 5*np.log10(qValRedone))[lowFreq:highFreq], alpha = 0.7)

print(np.mean((20*np.log10(antFac) - 5*np.log10(qValRedone))[lowFreq:highFreq]))

plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(r'Effective Antenna Factor (dBm$^{-1}$)', **label_font)
plt.tight_layout()
plt.grid()
#plt.legend(prop = legend_font)
plt.savefig('EffectiveAF_7-12-23.pdf', dpi = 100)
plt.show()

df = {'Frequency (MHz)': freqsRedone[0], 'QVal (linear)': qValRedone, 'Composite AF (m^-1)': antFac, 'Effective AF (dBm^-1)': 20*np.log10(antFac) - 5*np.log10(qValRedone)}
allData_df = pd.DataFrame(data = df)
allData = allData_df.to_csv('AllData_6-27-23.csv')