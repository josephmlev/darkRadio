from cmath import rect
import glob
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import random 
import sys



def medFit(arr, smoothFactor):
	overshoot = len(arr) % smoothFactor

	if overshoot != 0 :
		print(np.reshape(arr[:-overshoot], (-1, smoothFactor)))
		return np.median(np.reshape(arr[:-overshoot], (-1, smoothFactor)), axis = 1)
	else:
		return np.median(np.reshape(arr[:-overshoot], (-1, smoothFactor)), axis = 1)
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
	dataFiles = glob.glob(dirName + '/' + '*.csv' )
	return dataFiles 


# Catch-all function that returns frequencies, vswr, reflection coefficients, S11 magnitude,
# S11 phase, and S11 values written in a + bi form
def getSDataOne(fileNames):
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

def getSDataTwo(fileNames):
	freqs = []
	s11 = []
	s11Phase = []
	s21 = []
	s21Phase = []
	s22 = []
	s22Phase = []
	for aFile in fileNames:
		with open(aFile, 'r') as f:
			df = pd.read_csv(f)
			freqs.append(np.asarray(df['Frequency'])/1E6)
			s11.append(10**(np.asarray(df['S11_Magnitude'])/20))
			s11Phase.append(np.asarray(df['S11_Phase']))
			s22.append(10**(np.asarray(df['S22_Magnitude'])/20))
			s22Phase.append(np.asarray(df['S22_Phase']))
			s21.append(10**(np.asarray(df['S21_Magnitude'])/20))
			s21Phase.append(np.asarray(df['S21_Phase']))

	return np.asarray(freqs), np.asarray(s11), np.asarray(s11Phase), np.asarray(s22), np.asarray(s22Phase), np.asarray(s21), np.asarray(s22Phase)


# Calculate q value from formula usin dimensions of the room
def calcQData_One(s11Complex, meanS11, freqs):
	num = np.mean(np.abs(s11Complex - meanS11)**2, axis = 0)
	#num = (np.abs(s11Complex - meanS11)**2)
	den = (-1*np.abs(meanS11)**2 + 1)**2
	volume = 3.05 * 2.45 * 3.67
	wavelength = 3e2 / freqs[0]
	qVal = 8*(np.pi)**2*volume/wavelength**3*num/den
	return num, den, qVal


def calcQData_Two(s11, s22, s21, freqs):
	num = np.mean(np.abs(s21)**2, axis = 0)
	#num = (np.abs(s11Complex - meanS11)**2)
	den = (1 - np.mean(np.abs(s11)**2, axis = 0))*(1 - np.mean(np.abs(s22)**2, axis = 0))
	volume = 3.05 * 2.45 * 3.67
	wavelength = 3e2 / freqs[0]
	qVal = 16*(np.pi)**2*volume/wavelength**3*num/den
	return num, den, qVal

# Various font names
label_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'normal',
          'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 




dataDirTwo = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/TwoDipoles/'
dataDirOne = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/SingleDipole/'

dataDirBicon = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data2/'
#dataDirRedone = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data2/'

fileNamesTwo = sorted(getAllFiles(dataDirTwo))
fileNamesOne = sorted(getAllFiles(dataDirOne))
fileNamesBicon = sorted(getAllFiles(dataDirBicon))

freqsOne, vswrOne, reflecOne, s11MagOne, s11PhaseOne, s11ComplexOne = getSDataOne(fileNamesOne)
freqsBicon, vswrBicon, reflecBicon, s11MagBicon, s11PhaseBicon, s11ComplexBicon = getSDataOne(fileNamesBicon)
freqsTwo, s11Two, s11PhaseTwo, s22Two, s22PhaseTwo, s21Two, s21PhaseTwo = getSDataTwo(fileNamesTwo)

s11ComplexTwo = []
s22ComplexTwo = []
s21ComplexTwo = []

s11ComplexTwo.append(np.asarray([calcS11Complex(val[0], val[1]) for val in zip(s11Two, s11PhaseTwo)]))
s22ComplexTwo.append(np.asarray([calcS11Complex(val[0], val[1]) for val in zip(s22Two, s22PhaseTwo)]))
s21ComplexTwo.append(np.asarray([calcS11Complex(val[0], val[1]) for val in zip(s21Two, s21PhaseTwo)]))

meanS11One = np.mean(s11ComplexOne, axis = 0)
meanS11Bicon = np.mean(s11ComplexBicon, axis = 0)


meanS11Two = np.mean(s11ComplexTwo, axis = 0)
meanS22Two = np.mean(s22ComplexTwo, axis = 0)
meanS21Two = np.mean(s21ComplexTwo, axis = 0)


numOne, denOne, qValOne = calcQData_One(s11ComplexOne, meanS11One, freqsOne)
numBicon, denBicon, qValBicon = calcQData_One(s11ComplexBicon, meanS11Bicon, freqsBicon)

numTwo, denTwo, qValTwo = calcQData_Two(meanS11Two, meanS22Two, meanS21Two, freqsTwo)

# Plot frequencies and q values 
#plt.plot(freqs[0], (qVal), label = 'First Run')

medLen = 25
medFreqs = medFit(freqsOne[0], medLen)
medQValOne = medFit(qValOne, medLen)
medQValTwo = medFit(qValTwo, medLen)
medQValBicon = medFit(qValBicon, medLen)


plt.plot(medFreqs, 10*np.log10(medQValOne), label = 'One Dipole', alpha = 0.7)
plt.plot(medFreqs, 10*np.log10(medQValTwo), label = 'Two Dipoles', alpha = 0.7)
plt.plot(medFreqs, 10*np.log10(medQValBicon), label = 'Bicon', alpha = 0.7)

#plt.gca().set_yscale('log')


plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Q (dB)', **label_font)
plt.tight_layout()
plt.grid()
plt.legend(prop = legend_font)

plt.xlim([50, 500])
plt.ylim([0, 30])
plt.savefig('CompositeQOverlaid_MedianFit.png', dpi = 100)
plt.show()