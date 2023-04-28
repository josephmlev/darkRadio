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

# Various font names
label_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'normal',
          'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 

dataDir = '/home/bgodfrey/DarkRadio/darkRadio/VNA/QTesting/Data/Calibration/'

#Look at run-to-run variation
fileNames = ['Position2_Run1_Cal2_4-11-23.csv', 'Position2_Run2_Cal2_4-11-23.csv', 'Position2_Run3_Cal2_4-11-23.csv']

labels = ['Run 1', 'Run 2', 'Run 3']



freqs = []
vswr = []
reflec = []
for aFile in fileNames:
	with open(dataDir + aFile, 'r') as f:
		df = pd.read_csv(f)
		freqs.append(np.asarray(df['Frequency'])/1E6)
		vswr.append(np.asarray(df['S11_VSWR']))

reflec = np.asarray([calcReflect(x) for x in vswr])

for val in zip(freqs, reflec, labels):
	plt.plot(val[0], val[1], label = val[2], alpha = 0.5)

delta = np.asarray([np.abs(reflec[0] - x)/x*100 for x in reflec])

plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel(u'\u0393', **label_font)
plt.legend(prop = legend_font)
plt.show()


bins = np.linspace(0, 50, 100)
for val in zip(freqs, delta, labels):
	#plt.plot(val[0], val[1], label = val[2], alpha = 0.3)
	plt.hist(np.clip(val[1], bins[0], bins[-1]), label = val[2] + u': \u03BC = ' + str(round(np.mean(val[1][1:-1]),3)) + u'%, \u03C3 = ' + str(round(np.std(val[1][1:-1]),3)) + '%', bins=bins, alpha = 0.5)
	#plt.hist(val[1], bins = 100, alpha = 0.5)

plt.gca().set_yscale('log')
plt.xlabel('Difference (%)', labelpad = 15, **label_font)
plt.ylabel('Counts', **label_font)
plt.legend(prop = legend_font)
plt.savefig('VariationPosition2Calibration2.png')
plt.show()



#Look at calibration-to-calibration variation
fileNames1 = ['Position2_Run1_Cal1_4-11-23.csv', 'Position2_Run2_Cal1_4-11-23.csv', 'Position2_Run3_Cal1_4-11-23.csv']
fileNames2 = ['Position2_Run1_Cal2_4-11-23.csv', 'Position2_Run2_Cal2_4-11-23.csv', 'Position2_Run3_Cal2_4-11-23.csv']

freqs1 = []
vswr1 = []
reflec1 = []
for aFile in fileNames1:
	with open(dataDir + aFile, 'r') as f:
		df = pd.read_csv(f)
		freqs.append(np.asarray(df['Frequency'])/1E6)
		vswr.append(np.asarray(df['S11_VSWR']))

reflec1 = np.asarray([calcReflect(x) for x in vswr])

freqs2 = []
vswr2 = []
reflec2 = []
for aFile in fileNames2:
	with open(dataDir + aFile, 'r') as f:
		df = pd.read_csv(f)
		freqs.append(np.asarray(df['Frequency'])/1E6)
		vswr.append(np.asarray(df['S11_VSWR']))

reflec2 = np.asarray([calcReflect(x) for x in vswr])


mean1 = np.mean(reflec1, axis = 0)
mean2 = np.mean(reflec2, axis = 0)

delta = np.asarray(np.abs(mean1 - mean2)/mean1*100)

bins = np.linspace(0, max(delta), 100)


plt.hist(np.clip(delta, bins[0], bins[-1]), label = u'\u03BC = ' + str(round(np.mean(delta[1:-1]),3)) + u'%\n\u03C3 = ' + str(round(np.std(delta[1:-1]),3)) + '%', bins=bins, alpha = 0.5)

plt.gca().set_yscale('log')
plt.xlabel('Difference (%)', labelpad = 15, **label_font)
plt.ylabel('Counts', **label_font)
plt.legend(prop = legend_font)
plt.savefig('VariationCalibration.png')
plt.show()


# Look at position variation
#Look at run-to-run variation
fileNames1 = ['Position1_Run1_Cal1_4-11-23.csv', 'Position1_Run2_Cal1_4-11-23.csv', 'Position1_Run3_Cal1_4-11-23.csv']
fileNames2 = ['Position2_Run1_Cal2_4-11-23.csv', 'Position2_Run2_Cal2_4-11-23.csv', 'Position2_Run3_Cal2_4-11-23.csv']


freqs1 = []
vswr1 = []
reflec1 = []
for aFile in fileNames1:
	with open(dataDir + aFile, 'r') as f:
		df = pd.read_csv(f)
		freqs.append(np.asarray(df['Frequency'])/1E6)
		vswr.append(np.asarray(df['S11_VSWR']))

reflec1 = np.asarray([calcReflect(x) for x in vswr])

freqs2 = []
vswr2 = []
reflec2 = []
for aFile in fileNames2:
	with open(dataDir + aFile, 'r') as f:
		df = pd.read_csv(f)
		freqs.append(np.asarray(df['Frequency'])/1E6)
		vswr.append(np.asarray(df['S11_VSWR']))

reflec2 = np.asarray([calcReflect(x) for x in vswr])


mean1 = np.mean(reflec1, axis = 0)
mean2 = np.mean(reflec2, axis = 0)

delta = np.asarray(np.abs(mean1 - mean2)/mean1*100)

bins = np.linspace(0, max(delta), 100)


plt.hist(np.clip(delta, bins[0], bins[-1]), label = u'\u03BC = ' + str(round(np.mean(delta[1:-1]),3)) + u'%\n\u03C3 = ' + str(round(np.std(delta[1:-1]),3)) + '%', bins=bins, alpha = 0.5)

plt.gca().set_yscale('log')
plt.xlabel('Difference (%)', labelpad = 15, **label_font)
plt.ylabel('Counts', **label_font)
plt.legend(prop = legend_font)
plt.savefig('VariationPosition.png')
plt.show()


