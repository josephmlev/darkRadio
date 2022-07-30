from scipy.signal import butter, filtfilt, find_peaks, freqz, sosfilt
import glob
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sys 

def getData(dataDir):
	fileList = sorted([aFile for aFile in glob.glob(dataDir + 'data*.h5')], key=lambda x: int(x[x.index('data_') + 5: x.index('.h5')]))
	print(fileList)

	offData = np.zeros(2**23)
	onData = np.zeros(2**23)

	totalSets = 0
	totalAverages = 0
	for aFile in fileList[:100]:
		print('ON FILE: ' + str(aFile))
		dataBin = h5py.File(aFile, 'r')
		fileAverages = int(dataBin.attrs['averages'])
		allKeys = [aKey for aKey in dataBin.keys()]

		for aKey in allKeys[:]:
			print('ON KEY ' + str(aKey))
			dataset = pd.read_hdf(aFile, key = aKey)

			offData = offData + np.asarray(dataset[dataset.keys()[0]])
			onData = onData + np.asarray(dataset[dataset.keys()[1]])
			totalSets += 1
			totalAverages += fileAverages
	offData = 10.*np.log10(2. * offData / totalSets  / (2**48 * 50. / 1000.))
	onData = 10.*np.log10(2. * onData / totalSets  / (2**48 * 50. / 1000.))


	return offData, onData, totalAverages

def plotTermSigData(fileNameTerm, avgDataBicon, freqs):
	offDataTerm = np.zeros(2**23)
	onDataTerm = np.zeros(2**23)

	totalSets = 0
	print('ON FILE: ' + str(fileNameTerm))
	dataBin = h5py.File(fileNameTerm, 'r')
	allKeys = [aKey for aKey in dataBin.keys()]

	for aKey in allKeys[:]:
		print('ON KEY ' + str(aKey))
		dataset = pd.read_hdf(fileNameTerm, key = aKey)

		offDataTerm = offDataTerm + np.asarray(dataset[dataset.keys()[0]])
		onDataTerm = onDataTerm + np.asarray(dataset[dataset.keys()[1]])
		totalSets += 1

	offDataTerm = 10.*np.log10(2. * offDataTerm / totalSets  / (2**48 * 50. / 1000.))
	onDataTerm = 10.*np.log10(2. * onDataTerm / totalSets  / (2**48 * 50. / 1000.))
	peakIndices = find_peaks(avgDataBicon[1:] - onDataTerm[1:], threshold = 1.75)[0]
	peakVals = np.asarray([(avgDataBicon[1:] - onDataTerm[1:])[x] for x in peakIndices]) 

	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Ratio (dB)', **label_font)
	#plt.plot(freqs[1:], onDataTerm[1:], alpha = 0.7, label = 'Bicon Sig Gen Off')
	plt.plot(avgDataBicon[1:] - onDataTerm[1:], alpha = 0.7, label = 'Bicon Sig Gen On')
	plt.plot(peakIndices, peakVals, 'r*')

	plt.title('Ratio Bicon, Injected 123.456MHz vs No Injection')
	plt.show()
	#plt.legend(prop = legend_font)
	return peakIndices


label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
		  'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 
plt.tick_params(axis='both', which='major', labelsize=11)
mpl.rcParams['agg.path.chunksize'] = 10000

dataDir = '../data/'
termData, antData, totalAverages = getData(dataDir)


freqs = np.asarray(range(2**23))*600/2**24


plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', labelpad = 15, **label_font)
plt.plot(freqs[1::], antData[1::], alpha = 0.9, label = 'Bicon')

plt.plot(freqs[1::], termData[1::], alpha = 0.9, label = 'Terminator')

totalTime = totalAverages * 2**24/(6*10**8)
aTitle = 'TOTAL DATA: '
if totalTime > 360000:
	aTitle += str(round(totalTime / (60*60*24), 3)) + ' DAYS'
elif totalTime > 6000:
	aTitle += str(round(totalTime / (60*60), 3)) + ' HOURS'
elif totalTime > 600:
	aTitle += str(round(totalTime / (60), 3)) + ' MINUTES'
else:
	aTitle += str(round(totalTime, 3)) + ' SECONDS'

plt.title(aTitle, **title_font)
plt.legend(prop = legend_font)
plt.show()