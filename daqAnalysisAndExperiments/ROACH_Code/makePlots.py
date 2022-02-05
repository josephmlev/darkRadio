import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from scipy import stats 

def butter_highpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return b, a

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a
	
def butter_highpass_filter(data, cutoff, fs, order=5, high = True):
	if high:
		b, a = butter_highpass(cutoff, fs, order=order)
	else:
		b, a = butter_lowpass(cutoff, fs, order=order)
	y = filtfilt(b, a, data)
	return y

def gaussian(x, mean, amplitude, standard_deviation):
	return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))

def getClosest(freqList, knownFreqs, knownAmps):
	index = 0
	totalKnown = len(knownAmps) 
	nearestAmps = []
	for counter, freqVal in enumerate(freqList):
		if counter % 1000000 == 0:
			print('DONE WITH ' + str(counter))
		while freqVal > knownFreqs[index]:
			if index == totalKnown - 1:
				break
			index = index + 1
		if index == 0:
			#print(str(freqVal) + 'MHz IS LESS THAN ' + str(knownFreqs[0]) + 'MHz')
			nearestAmps.append(knownAmps[0])
		elif index == totalKnown - 1:
			#print(str(freqVal) + 'MHz IS GREATER THAN ' + str(knownFreqs[-1]) + 'MHz')
			nearestAmps.append(knownAmps[-1])
		else:
			#print(str(freqVal) + 'MHz IS BETWEEN ' + str(knownFreqs[index-1]) + ' AND ' + str(knownFreqs[index]) + ' MHz')
			x0 = knownFreqs[index]
			y0 = knownAmps[index]
			x1 = knownFreqs[index-1]
			y1 = knownAmps[index-1]
			slope = (y1 - y0) / (x1 - x0)
			nearest = slope*(freqVal - x0) + y0
			#print('GAIN RANGE: ' + str(round(knownAmps[index-1], 3)) + '-' + str(round(knownAmps[index], 3)) + 'dB')
			#print('INTERPOLATED VALUE: ' + str(round(nearest, 6)) + 'dB')
			nearestAmps.append(nearest)
	return nearestAmps

def getSystemGain(fileName):
	freqs = []
	gains = []
	with open(fileName, 'r') as f:
		f.readline()
		for line in f:
			holder = line.split()
			freqs.append(float(holder[0]))
			gains.append(float(holder[1]))

	return np.asarray(freqs), np.asarray(gains)

def simpleAnalysis(arr, highSigma = 5):
	arr = np.exp(arr/10.)
	stdVal = np.std(arr)
	meanVal = np.mean(arr)
	interestingArr = []
	for counter, val in enumerate(arr):
		if (val-meanVal)/stdVal > highSigma:
			interestingArr.append((counter, (val-meanVal)/stdVal))
		if counter%1000000 == 0:
			print('DONE WITH ' + str(counter) + ' VALUES')
	return interestingArr


def plotAcqs(fileNameBase, freqs, amps1, amps2):
	label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
			  'verticalalignment':'bottom'} 
	title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
	legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 

	plt.tick_params(axis='both', which='major', labelsize=11)
	mpl.rcParams['agg.path.chunksize'] = 10000
	fileName1 = fileName_Base + 'Sub1.bin'
	fileName2 = fileName_Base + 'Sub2.bin'
	#fileName1 = 'ArduinoInterferenceTest_11-14-21_On_Sub1.bin'
	#fileName2 = 'ArduinoInterferenceTest_11-14-21_Off_Sub2.bin'
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Power (dBm)', **label_font)
	plt.plot(freqs[1:], amps1[1:])
	plt.savefig(fileName1[0:fileName1.index('.bin')] + '.png')
	plt.show()
	plt.clf()
	
	#modifiedData = butter_highpass_filter(amps1[1:], len(amps1[1:])/10., len(amps1[1:]))
	#plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	#plt.ylabel('Amplitude (dBm)', **label_font)
	#plt.plot(freqs[1:], amps1[1:])
	#plt.savefig(fileName1[0:fileName1.index('.bin')] + '.png')
	#plt.show()
	#plt.clf()

	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Power (dBm)', **label_font)
	plt.gca().ticklabel_format(useOffset=False, style='plain')
	[i.set_linewidth(1.5) for i in plt.gca().spines.itervalues()]
	plt.plot(freqs[1:], amps2[1:])
	plt.savefig(fileName2[0:fileName2.index('.bin')] + '.png')
	plt.show()
	plt.clf()
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Power (dBm)', **label_font)
	plt.gca().ticklabel_format(useOffset=False, style='plain')
	[i.set_linewidth(1.5) for i in plt.gca().spines.itervalues()]
	#plt.plot(freqs[1:], amps1[1:], alpha = 0.7, label = u'50\u03A9 Terminator')
	#plt.plot(freqs[1:], amps2[1:], alpha = 0.7, label = 'Noise Source Off')
	plt.plot(freqs[1:], amps1[1:] + 10, alpha = 0.7, label = u'50\u03A9 Terminator 1/14/22')
	plt.plot(freqs[1:], amps2[1:], alpha = 0.7, label = u'50\u03A9 Terminator 12/20/21')

	#plt.ylim([-95, -85])
	plt.legend(prop = legend_font)
	#print('SAVING AS: ' + fileName1[0:fileName1.index('_Sub1.bin')] + '_Corrected_ Overlaid.png')
	#plt.savefig(fileName1[0:fileName1.index('_Sub1.bin')] + '_Corrected_Overlaid.png')

def plotAcqsGainCorrected(fileNameBase, freqs, amps1, amps2):
	label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
			  'verticalalignment':'bottom'} 
	title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
	legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 

	plt.tick_params(axis='both', which='major', labelsize=11)
	mpl.rcParams['agg.path.chunksize'] = 10000
	fileName1 = fileName_Base + 'Sub1.bin'
	fileName2 = fileName_Base + 'Sub2.bin'


	systemGainFile = 'YFactor_ENR_Gains_1-22.txt'
	sysFreqs, sysGains = getSystemGain(systemGainFile)
	
	SBBGainFile = 'SBB5089Z_Gains_Measured_12-21.txt'
	SBBFreqs = []
	SBBGains = []
	with open(SBBGainFile, 'r') as f:
		f.readline()
		for line in f:
			holder = line.split()
			SBBFreqs.append(float(holder[0]))
			SBBGains.append(float(holder[3]))

	#extraGainFile = 'NoiseMeasurements_PE15A1014ANDPE15A1012_10-26-21.txt'
	#extraFreqs = []
	#extraGains = []

	#sysGains = SBBGains
	#print(zip(extraFreqs, extraGains))		
	interpGains = getClosest(freqs, sysFreqs, sysGains)
	#interpExtra = getClosest(freqs, extraFreqs, extraGains)

	resBand = 10*np.log10(6.*10**8/2**24)		
	#sysGain = getClosest(freqs, sysFreqs, sysGains)

	amps1 = amps1 - interpGains  - resBand
	#amps1 = 10*np.log10(10**(amps1/10.) - 10**(-108/10.))

	amps2 = amps2 - interpGains - resBand
	#amps2 = 10*np.log10(10**(amps2/10.) - 10**(-108/10.))

	
	#amps1 = amps1 - sysGains - resBand
	#amps2 = amps2 - sysGains - resBand
	plt.ylim([-174, -171])
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('PSD (dBm/Hz)', **label_font)
	plt.plot(freqs[1:], amps1[1:] + 10)

	plt.savefig(fileName1[0:fileName1.index('.bin')] + '_YFACTOR_PSD_ZOOM.png')
	plt.show()
	plt.clf()

	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('PSD (dBm/Hz)', **label_font)
	plt.plot(freqs[1:], amps2[1:])
	#plt.ylim([-174, -171])
	#plt.savefig(fileName2[0:fileName2.index('.bin')] + '_PSD.png')
	plt.show()
	plt.clf()

	medianAmps1 = np.median(np.reshape(amps1 + 10, (-1, 2**10)), axis = 1)
	medianFreqs = np.median(np.reshape(freqs, (-1, 2**10)), axis = 1)

	medianAmps2 = np.median(np.reshape(amps2, (-1, 2**10)), axis = 1)


	roomTemp = -173.9
	ampTemp = (10**(medianAmps1/10.) - 10**(roomTemp/10.))*10**-3/(1.381*10**-23)

	ampTemp2 = (10**(medianAmps2/10.) - 10**(roomTemp/10.))*10**-3/(1.381*10**-23)



	#plt.ylim([-173, -171])
	#plt.savefig(fileName2[0:fileName2.index('.bin')] + '_YFACTOR_PSD_ZOOM.png')
	plt.show()
	plt.clf()

	plt.plot(medianFreqs, ampTemp, label = '1/14/22 Noise Source')
	plt.plot(medianFreqs, ampTemp, label = '12/20/21')

	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Amp Temperature (K)', **label_font)
	plt.legend(prop = legend_font)
	#plt.savefig(fileName2[0:fileName2.index('.bin')] + '_YFACTOR_TEMP_NOISE_COMPARISON.png')

	plt.show()


	#plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	#plt.ylabel('PSD (dBm/Hz)', **label_font)
	#plt.gca().ticklabel_format(useOffset=False, style='plain')
	#[i.set_linewidth(1.5) for i in plt.gca().spines.itervalues()]
	#plt.plot(freqs[1:], amps2[1:])
	#plt.savefig(fileName2[0:fileName2.index('.bin')] + '_PSD.png')
	#plt.clf()

	#plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	#plt.ylabel('PSD (dBm/Hz)', **label_font)
	#plt.gca().ticklabel_format(useOffset=False, style='plain')
	#[i.set_linewidth(1.5) for i in plt.gca().spines.itervalues()]
	#plt.plot(freqs[1:], amps1[1:], alpha = 0.7, label = u'50\u03A9 Terminator')
	#plt.plot(freqs[1:], amps2[1:], alpha = 0.7, label = 'Bicon')
	#plt.ylim([-170, -165])
	#plt.legend(prop = legend_font)
	#plt.savefig(fileName1[0:fileName1.index('_Sub1.bin')] + '_OverlaidPSD.png')

def plotRatio(fileNameBase, freqs, amps1, amps2, hist = False):
	label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
			  'verticalalignment':'bottom'} 
	title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
	legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 

	plt.tick_params(axis='both', which='major', labelsize=11)
	mpl.rcParams['agg.path.chunksize'] = 10000
	fileName1 = fileName_Base + 'Sub1.bin'
	fileName2 = fileName_Base + 'Sub2.bin'
	#fileName1 = 'ArduinoInterferenceTest_11-14-21_On_Sub1.bin'
	#fileName2 = 'ArduinoInterferenceTest_11-14-21_Off_Sub2.bin'
	ampsDiffLog = amps2 - amps1
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Ratio (dB)', **label_font)
	plt.gca().ticklabel_format(useOffset=False, style='plain')
	plt.ylim([-3, 3])
	plt.plot(freqs[1:], ampsDiffLog[1:])
	plt.savefig(fileName2[0:fileName2.index('_Sub2.bin')] + '_RATIO.png')
	plt.show()

	if hist:
		plt.clf()
		plt.tick_params(axis='both', which='major', labelsize=11)
		plt.gca().ticklabel_format(useOffset=False, style='plain')
		n, bins, _ = plt.hist(ampsDiff[1:-1], 100, range = [-1.5, 1.5], weights=np.ones(len(ampsDiff[1:-1])) / len(ampsDiff[1:-1]), linewidth = 1, histtype = 'stepfilled', color = 'blue', fill = True, zorder = 3)
		bin_centers = bins[:-1] + np.diff(bins) / 2
		popt, _ = curve_fit(gaussian, bin_centers, n)
		popt[2] = np.abs(popt[2])
		x_interval_for_fit = np.linspace(bins[0], bins[-1], 1000)
		plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), zorder = 2)
		plt.plot([popt[0]-6*popt[2], popt[0]-6*popt[2]], [10**-8, 10**-1], 'g-', linewidth = 3, zorder = 1)
		plt.plot([popt[0]+6*popt[2], popt[0]+6*popt[2]], [10**-8, 10**-1], 'g-', linewidth = 3, zorder = 1)
		plt.yscale('log', nonposy = 'clip')
		plt.ylim([10**-8, 10**-1])
		plt.xlim([-1.5, 1.5])
		plt.xlabel('Ratio (dB)', labelpad = 15, **label_font)
		plt.ylabel('Fraction', **label_font)
		plt.savefig(fileName2[0:fileName2.index('_Sub2.bin')] + '_HIST_RATIO_LOG.png', bbox_inches='tight', dpi = 600)

		numBigger = 0
		startFreq = 0
		endFreq = 1000

		startIndex = int(startFreq/freqStep)
		endIndex = int(endFreq/freqStep)
		badFreqs = []

		for counter, val in enumerate(ampsDiff[startIndex:endIndex]):
			if np.abs(val - popt[0])/popt[2] > 5: #and (freqs[counter+startIndex] > 200) or freqs[counter+startIndex] > 850):
				print('BAD VAL AT ' + str(freqs[counter+startIndex]) + ' MHz')
				numBigger = numBigger + 1
				badFreqs.append((freqs[counter+startIndex], (val - popt[0])/popt[2]))
		
		print('TOTAL HIGH SIGMA POINTS: ' + str(numBigger))
		badFileName = fileName_Base + 'badFreqs.txt'
		with open(badFileName, 'w') as f:
			for val in badFreqs:
				f.write(str(val[0]) + '\t: ' + str(val[1]) + '\n')


def plotDifferenceLog(fileNameBase, freqs, amps1, amps2):
	label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
			  'verticalalignment':'bottom'} 
	title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
	legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 
	plt.tick_params(axis='both', which='major', labelsize=11)
	mpl.rcParams['agg.path.chunksize'] = 10000
	fileName1 = fileName_Base + 'Sub1.bin'
	fileName2 = fileName_Base + 'Sub2.bin'
	#fileName1 = 'ArduinoInterferenceTest_11-14-21_On_Sub1.bin'
	#fileName2 = 'ArduinoInterferenceTest_11-14-21_Off_Sub2.bin'
	#ampsDiffLog = amps2 - amps
	
	ampsDiffLin_Pos_Amps = []
	ampsDiffLin_Pos_Freqs = []

	ampsDiffLin_Neg_Amps = []
	ampsDiffLin_Neg_Freqs = []

	ampsDiffLin = (10**(amps1/10.) - 10**(amps2/10.))
	for val in zip(freqs, ampsDiffLin):
		if val[1] > 0:
			ampsDiffLin_Pos_Amps.append(10*np.log10(val[1]))
			ampsDiffLin_Pos_Freqs.append(val[0])
		elif val[1] < 0:
			ampsDiffLin_Neg_Amps.append(10*np.log10(np.abs(val[1])))
			ampsDiffLin_Neg_Freqs.append(val[0])

	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Difference (dBm)', **label_font)
	plt.plot(ampsDiffLin_Pos_Freqs[1:], ampsDiffLin_Pos_Amps[1:])
	plt.savefig(fileName2[0:fileName2.index('_Sub2.bin')] + '_DIFFERENCE_POSITIVE_LOG.png')
	plt.clf()

	modifiedData = butter_highpass_filter(ampsDiffLin_Pos_Amps[1:], len(ampsDiffLin_Pos_Amps[1:])/10., len(ampsDiffLin_Pos_Amps[1:]))
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Filtered Difference (Filtered dBm)', **label_font)
	plt.plot(ampsDiffLin_Pos_Freqs[1:], modifiedData)
	plt.savefig(fileName2[0:fileName2.index('_Sub2.bin')] + '_DIFFERENCE_POSITIVE_LOG_FILTERED.png')
	plt.show()

	plt.clf()
	plt.tick_params(axis='both', which='major', labelsize=11)
	plt.gca().ticklabel_format(useOffset=False, style='plain')
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Difference (dBm)', **label_font)
	plt.plot(ampsDiffLin_Neg_Freqs[1:], ampsDiffLin_Neg_Amps[1:])
	plt.savefig(fileName2[0:fileName2.index('_Sub2.bin')] + '_DIFFERENCE_NEGATIVE_LOG.png')

def plotDifferenceLinear(fileNameBase, freqs, amps1, amps2):
	label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
			  'verticalalignment':'bottom'} 
	title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
	legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 
	plt.tick_params(axis='both', which='major', labelsize=11)
	mpl.rcParams['agg.path.chunksize'] = 10000
	fileName1 = fileName_Base + 'Sub1.bin'
	fileName2 = fileName_Base + 'Sub2.bin'
	#fileName1 = 'ArduinoInterferenceTest_11-14-21_On_Sub1.bin'
	#fileName2 = 'ArduinoInterferenceTest_11-14-21_Off_Sub2.bin'

	ampsDiffLin = (10**(amps1/10.) - 10**(amps2/10.))*10**12


	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Difference (fW)', **label_font)
	plt.ylim([-500, 500])
	plt.plot(freqs[1:], ampsDiffLin[1:])
	plt.savefig(fileName2[0:fileName2.index('_Sub2.bin')] + '_DIFFERENCE_LIN.png')


parser = argparse.ArgumentParser(description = 'Plot data from ROACH')

parser.add_argument('filename', metavar='base filename', type=str, nargs=1,
                    help='Base filename to process')
parser.add_argument('--all', action = 'store_true', help = 'make all plots, includes ratio, both acquisitions, and difference plots')
parser.add_argument('--acqs', action = 'store_true', help='only make plots of raw acquisitions')
parser.add_argument('--acqsPSD', action = 'store_true', help='only make plots of acquisition PSD')
parser.add_argument('--difflin', action = 'store_true', help='only make subtraction plots linear axes')
parser.add_argument('--difflog', action = 'store_true', help='only make subtraction plots logarithmic axes')
parser.add_argument('--ratio', action = 'store_true', help='only make ratio plots')
parser.add_argument('--ratiohist', action = 'store_true', help='make ratio histogram has to be explicitly specified')
parser.add_argument('--verbose', action = 'store_true', help='display information about plots')

args = parser.parse_args()
fileName_Base = args.filename[0]

fileName1 = fileName_Base + 'Sub2.bin'
fileName2 = fileName_Base + 'Sub1.bin'
#fileName2 = fileName_Base + 'Sub2.bin'
#fileName1 = 'ArduinoInterferenceTest_11-14-21_On_Sub1.bin'
#fileName2 = 'ArduinoInterferenceTest_11-14-21_Off_Sub2.bin'

print(fileName2)

allData = np.fromfile(fileName1, dtype = np.float32)
allData2 = np.fromfile(fileName2, dtype = np.float32)
amps1 = 10.*np.log10(2. * allData  / (2**48 * 50. / 1000.))
amps2 = 10.*np.log10(2. * allData2  / (2**48 * 50. / 1000.))

freqStep = (6.0*10**8) /2**24*10**-6 
freqs = [x * freqStep for x in range(2**23)]

if args.all:
	plotAcqs(fileName_Base, freqs, amps1, amps2)
	plt.clf()
	plotAcqsGainCorrected(fileName_Base, freqs, amps1, amps2)
	plt.clf()
	plotDifferenceLinear(fileName_Base, freqs, amps1, amps2)
	plt.clf()
	plotDifferenceLinear(fileName_Base, freqs, amps1, amps2)
	plt.clf()
	plotDifferenceLog(fileName_Base, freqs, amps1, amps2)
	plt.clf()
	plotRatio(fileName_Base, freqs, amps1, amps2)
	plt.clf()
if not(args.all) and args.acqs:
	plotAcqs(fileName_Base, freqs, amps1, amps2)
if not(args.all) and args.acqsPSD:
	plotAcqsGainCorrected(fileName_Base, freqs, amps1, amps2)
if not(args.all) and args.difflin:
	plotDifferenceLinear(fileName_Base, freqs, amps1, amps2)
if not(args.all) and args.difflog:
	plotDifferenceLog(fileName_Base, freqs, amps1, amps2)
if not(args.all) and args.ratio:
	if args.ratiohist:
		plotRatio(fileName_Base, freqs, amps1, amps2)
	else:
		plotRatio(fileName_Base, freqs, amps1, amps2, hist = True)

if args.verbose:
	amps1Sum = 10*np.log10(np.sum(10**(amps1/10.)))
	amps2Sum = 10*np.log10(np.sum(10**(amps2/10.)))
	ampsDiff = amps2 - amps1
	print('TOTAL POWER AMPS 1: ' + str(amps1Sum))
	print('TOTAL POWER AMPS 2: ' + str(amps2Sum))
	print('STD RATIO: ' + str(np.std(ampsDiff[1:])) + 'dB')
	print('MEAN RATIO: ' + str(np.mean(ampsDiff[1:])) + 'dB')







