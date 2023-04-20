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

	for aFile in fileList[:]:
		print('ON FILE: ' + str(aFile))
		dataBin = h5py.File(aFile, 'r')
		allKeys = [aKey for aKey in dataBin.keys()]

		for aKey in allKeys[:]:
			print('ON KEY ' + str(aKey))
			dataset = pd.read_hdf(aFile, key = aKey)

			offData = offData + np.asarray(dataset[dataset.keys()[0]])
			onData = onData + np.asarray(dataset[dataset.keys()[1]])
			totalSets += 1

	offData = 10.*np.log10(2. * offData / totalSets  / (2**48 * 50. / 1000.))
	onData = 10.*np.log10(2. * onData / totalSets  / (2**48 * 50. / 1000.))


	return offData, onData

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




dataDirHot = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/HotAmp/'
#dataDirCold = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/ColdAmp/'


#print('GETTING HOT DATA')
offDataHot, onDataHot = getData(dataDirHot)




#print('GETTING COLD DATA')
#offDataCold, onDataCold = getData(dataDirCold)

freqs = np.asarray(range(2**23))*600/2**24

#saveFile = 'RoomTempAmp_2-11-22.txt'
#with open(saveFile, 'w') as f:
#	f.write('Frequency (MHz)\tTerminator (dBm)\tBicon(dBm)\n')
#	for val in zip(freqs[::1000], offDataHot[::1000], onDataHot[::1000]):
#		f.write(str(val[0]) + '\t\t' + str(val[1]) + '\t\t' + str(val[2]) + '\n')


#saveFile = 'ColdTempAmp_2-11-22.txt'
#with open(saveFile, 'w') as f:
#	f.write('Frequency (MHz)\tTerminator (dBm)\tBicon(dBm)\n')
#	for val in zip(freqs[::1000], offDataCold[::1000], onDataCold[::1000]):
#		f.write(str(val[0]) + '\t\t' + str(val[1]) + '\t\t' + str(val[2]) + '\n')

#sys.exit(1)

offData = offDataHot
onData = onDataHot	


#Terminated signal generator data: '/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/HotAmp/SigGenTerminated.h5'
fileTermSig = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/HotAmp/SigGenTerminated.h5'

peakIndices = plotTermSigData(fileTermSig, onData, freqs)

#for x in peakIndices:
#	if x > 100 and x < len(peakIndices) - 100:
#		onData[x] = np.median([onData[x-100:x+100]])
#	elif x < 100:
#		onData[x] = np.median([onData[0:200]])
#	else:
#		onData[x] = np.median([onData[-200:]])

plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', labelpad = 15, **label_font)
#plt.ylabel('Power Spectral Density (dBm/Hz)', **label_font)
#plt.title('Comparison Terminator')


gainFile = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/YFactor_ENR_Gains_1-22.txt'

gainFreqs = []
gainVals = []

with open(gainFile, 'r') as f:
	f.readline()
	for line in f:
		holder = line.split()
		gainFreqs.append(float(holder[0]))
		gainVals.append(float(holder[1]))

interpGain = np.interp(freqs[100::1000], gainFreqs, gainVals)

psdBicon =  onData[100::1000] #- interpGain - 10*np.log10(600*10**6/2**24)
psdTerm = offData[100::1000] #- interpGain - 10*np.log10(600*10**6/2**24)

plt.plot(freqs[100::1000], psdBicon, alpha = 0.9, label = 'Bicon')

plt.plot(freqs[100::1000], psdTerm, alpha = 0.9, label = 'Terminator')

#plt.plot([50, 300], [-173.9, -173.9], 'r--', alpha = 0.9, label = '295K Johnson')


plt.legend(prop = legend_font)
#plt.xlim([50, 300])
plt.ylim([-85.3, -77.7])
#plt.ylim([-175, -171])
#plt.savefig('/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/BiconVsTerminatorPSD_Every1000.png', bbox_inches='tight', dpi = 600)
plt.savefig('/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/BiconVsTerminator_Every1000.pdf', bbox_inches='tight')
plt.show()
sys.exit(1)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Temperature (K)', **label_font)
ampNoise = 10 ** (psdTerm/10.)*10**-3/(1.381*10**-23) - 295.5
plt.plot(freqs[100::1000], ampNoise, alpha = 0.9)
plt.xlim([50, 300])
plt.ylim([50, 200])
plt.savefig('/home/dark-radio/HASHPIPE/ROACH/PYTHON/DataRun_2-11-22/AmpTemperatureCalculation_Every1000.png', bbox_inches='tight', dpi = 600)
plt.show()

#plt.plot(freqs[100::1000], onData[100::1000]-offData[100::1000] - interpGain, alpha = 0.9, label = 'Terminator')



sys.exit(1)

plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.title('Comparison Bicon')

startIndex = int(50 / (600/2**24)) + 1
print('MEAN HOT TERMINATOR: ' + str(round(np.mean(offDataHot[startIndex:]), 3)) + ' dBm')
print('MEAN COLD TERMINATOR: ' + str(round(np.mean(offDataCold[startIndex:]), 3)) + ' dBm')

print('MEAN HOT BICON: ' + str(round(np.mean(onDataHot[startIndex:]), 3)) + ' dBm')
print('MEAN COLD BICON: ' + str(round(np.mean(onDataCold[startIndex:]), 3)) + ' dBm')

plt.plot(freqs[1:], onDataCold[1:], alpha = 0.7, label = 'Bicon Cold')
plt.plot(freqs[1:], onDataHot[1:], alpha = 0.7, label = 'Bicon Hot')
plt.legend(prop = legend_font)

plt.show()


plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.title('Comparison Bicon')

startIndex = int(50 / (600/2**24)) + 1
print('MEAN HOT TERMINATOR: ' + str(round(np.mean(offDataHot[startIndex:]), 3)) + ' dBm')
print('MEAN HOT BICON: ' + str(round(np.mean(onDataHot[startIndex:]), 3)) + ' dBm')

print('MEAN COLD TERMINATOR: ' + str(round(np.mean(offDataCold[startIndex:]), 3)) + ' dBm')
print('MEAN HOT BICON: ' + str(round(np.mean(onDataHot[startIndex:]), 3)) + ' dBm')

plt.plot(freqs[1:], offDataHot[1:], alpha = 0.7, label = 'Terminator Hot')
plt.plot(freqs[1:], onDataHot[1:], alpha = 0.7, label = 'Bicon Hot')
plt.legend(prop = legend_font)

plt.show()


plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.title('Comparison Bicon')

plt.plot(freqs[1:], offDataCold[1:], alpha = 0.7, label = 'Terminator Cold')
plt.plot(freqs[1:], onDataCold[1:], alpha = 0.7, label = 'Bicon Cold')
plt.legend(prop = legend_font)

plt.show()




#plt.show()


plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.plot(freqs[1:], offData[1:], alpha = 0.7, label = 'Terminator')
plt.plot(freqs[1:], onData[1:], alpha = 0.7, label = 'Bicon')
plt.legend(prop = legend_font)

plt.show()

dataDiff = onData - offData
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Ratio (dB)', **label_font)
plt.title('Ratio Method')
plt.plot(freqs[1:], dataDiff[1:])
plt.show()

dataDiffTony = (10**(onData/10.) - 10**(offData/10.))
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Difference (mW)', **label_font)
plt.title('Difference Method')
plt.plot(freqs[1:], dataDiffTony[1:])
plt.show()



#filteredDiff = butter_highpass_filter(dataDiff[1:], len(dataDiff[1:])/1000., len(dataDiff[1:]))

fc = 0.3

b, a = butter(5, fc, btype = 'high', analog = 'False')
filteredDiff = filtfilt(b, a, dataDiff[1:])

plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Filtered Ratio (dB)', **label_font)

plt.plot(freqs[1:], filteredDiff)
plt.show()

#print(np.std((filteredDiff)[1:]))

#writeFile = 'averagedData_24HrRun_2-8-22.txt'

#with open(writeFile, 'w') as f:
#	f.write('Frequency (MHz)\tTerm Power (dBm)\tRes Power (dBm)\n')
#	for val in zip(freqs, offData, onData):
#		f.write(str(val[0]) + '\t' + str(val[1]) + '\t' + str(val[2]) + '\n')




