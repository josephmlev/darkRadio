from scipy import signal
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.append('/home/dark-radio/darkRadio/drlib/')
import drlib as dr 

def applyFilter(cutoff, data, plotResponse = False):
	fs = len(data)
	#print(len(data[0]))
	fcNorm = 2./cutoff
	b, a = signal.butter(6,fcNorm, 'highpass', analog = False)

	if plotResponse:
		w, h = signal.freqz(b, a, worN = 2**18)
		plt.semilogx(1 / (w[1:] / (2*np.pi)), 20*np.log10(abs(h[1:])))
		plt.margins(0, 0.1)
		plt.xlim([10**7, 1])
		plt.gca().set_xticks([10**7, 10**6, 10**5, 10**4, 10**3, 10**2, 10**1, 10**0])
		plt.grid(which = 'both', axis = 'both')
		plt.axvline(cutoff, color = 'green')
		plt.xlabel('Number of Bins', labelpad = 15, **label_font)
		plt.ylabel('Amplitude (dB)', **label_font)
		plt.xticks(fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.show()
	return np.asarray(signal.filtfilt(b, a, data), dtype = 'float32')
	#return signal.filtfilt(b, a, data)

label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
		  'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 


dataDir = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/Analysis_9-26-22/'
antFile = 'run1_antData_avgAll11776Spectra_gitignore.npy'
termFile = 'run1_termData_avgAll11776Spectra_gitignore.npy'
freqFile = 'run1_freqData_avgAll11776Spectra_gitignore.npy'

antData = np.load(dataDir + antFile)
antDataLog = dr.fft2dBm(antData[1:])

termData = np.load(dataDir + termFile)
termDataLog = dr.fft2dBm(termData[1:])

freqData = np.load(dataDir + freqFile)
plt.plot(freqData, antDataLog, label = 'Bicon')
plt.plot(freqData, termDataLog, label = 'Terminator')
plt.legend(prop = legend_font)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.show()

antDataFiltered = applyFilter(100, antDataLog)
termDataFiltered = applyFilter(100, termDataLog)
plt.plot(freqData, antDataFiltered, alpha = 0.7, label = 'Bicon')
plt.plot(freqData, termDataFiltered, alpha = 0.7, label = 'Terminator')

plt.legend(prop = legend_font, loc = 'upper right')
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Filtered Power (dBm)', **label_font)
plt.ylim([-0.2, 0.2])
plt.show()


minVal = min((10**(antDataLog/10.) - 10**(termDataLog/10.)))


diffDataLog = 10*np.log10((10**(antDataLog/10.) - 10**(termDataLog/10.)) + np.abs(minVal)*1.05)
plt.plot(freqData, diffDataLog)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power Difference (dBm)', **label_font)
plt.savefig('PowerDifferenceOffsetAdded.png', bbox_inches = 'tight', dpi = 100)
#plt.ylim([-0.2, 0.2])
plt.show()

diffDataFiltered = applyFilter(100, diffDataLog)
plt.plot(freqData, diffDataFiltered)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Filtered Power (dBm)', **label_font)
plt.savefig('PowerDifferenceOffsetAdded_Filtered.png', bbox_inches = 'tight', dpi = 100)
plt.show()

q3, q1 = np.percentile(diffDataFiltered, [75 ,25])
iqr = q3 - q1
# Freedman-Diaconis rule
binWidth = 2*iqr / ((2**24)**(1/3.))

bins = np.linspace(-0.03, 0.03, int(0.03/binWidth))
#bins = np.linspace(-0.3, 0.3, int(0.6/binWidth))
print('NUM BINS: ' + str(len(bins)))

n, bins, patches = plt.hist(np.clip(antDataFiltered, bins[0], bins[-1]), bins=bins)
#n, bins, patches = plt.hist(np.clip(diffDataFiltered, bins[0], bins[-1]), bins=bins)

print('NUMBER IN FIRST BIN: ' + str(n[0]))
print('NUMBER IN LAST BIN: ' + str(n[-1]))
plt.xlabel('Filtered Power (dB)', labelpad = 15, **label_font)
plt.ylabel('Counts', **label_font)
plt.gca().set_yscale('log')
plt.savefig('PowerDifferenceOffsetAdded_Histogram.png', bbox_inches = 'tight', dpi = 100)

plt.show()
sys.exit(1)

medLen = 2**12

medTerm = np.median(termData.reshape((-1, medLen)), axis = 1)
medTermLog = dr.fft2dBm(medTerm)
medFreq = np.median(np.concatenate(([0], freqData)).reshape((-1, medLen)), axis = 1)


#spl = UnivariateSpline(freqData, termDataLog)
#xs = np.linspace(freqData[0], freqData[-1], 2**16)



plt.plot(freqData, termDataLog, label = 'Terminator')
plt.plot(medFreq, medTermLog, label = 'Terminator Median')
#plt.plot(xs, spl(xs), 'g', lw=3, label = 'Spline Fit')
plt.legend(prop = legend_font)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.show()
freqStep = freqData[1] - freqData[0]
startIndex = 10 / freqStep

antDataLogReshape = np.concatenate(([0], antDataLog)).reshape((-1, medLen))

antDataCorrection = medTermLog - np.median(antDataLogReshape, axis = 1)

newAntDataLog = np.asarray([val[0] + val[1] for val in zip(antDataLogReshape, antDataCorrection)]).reshape((-1))

plt.plot(freqData, termDataLog, label = 'Terminator')
plt.plot(freqData, newAntDataLog[1:], label = 'Antenna Median')
plt.legend(prop = legend_font)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (dBm)', **label_font)
plt.show()

startIndex = 0 #int(50 / freqStep)
endIndex = len(newAntDataLog) #int(299/freqStep)
#newAntDataLog = 10*np.log10(np.abs(10**(newAntDataLog/10.) - 10**(np.concatenate(([0], termDataLog))/10))) 
#newAntDataLogFiltered = applyFilter(100, newAntDataLog)
newAntDataLog = newAntDataLog - np.concatenate(([0], termDataLog))
newAntDataLogFiltered = np.asarray([applyFilter(100, x) for x in newAntDataLog.reshape((-1, medLen))]).reshape((-1))
plt.plot(freqData[startIndex:endIndex], newAntDataLogFiltered[startIndex+1:endIndex])
plt.legend(prop = legend_font)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Filtered Power (dBm)', **label_font)
plt.ylim([-0.2, 0.2])
#plt.savefig(dataDir + 'AntDataReshaped_Filtered.png', bbox_inches = 'tight', dpi = 100)
plt.show()


q3, q1 = np.percentile(newAntDataLogFiltered[startIndex:endIndex], [75 ,25])
iqr = q3 - q1
# Freedman-Diaconis rule
binWidth = 2*iqr / ((2**24)**(1/3.))

#bins = np.linspace(-0.1, 0.1, 10000)
bins = np.linspace(-0.015, 0.015, int(0.03/binWidth))
#bins = np.linspace(-5*np.std(newAntDataLogFiltered), 5*np.std(newAntDataLogFiltered), int(10*np.std(newAntDataLogFiltered)/binWidth))

print('NUM BINS: ' + str(len(bins)))
n, bins, patches = plt.hist(np.clip(newAntDataLogFiltered[startIndex:endIndex], bins[0], bins[-1]), bins=bins)
plt.xlim([-0.02, 0.02])
#n, bins, patches = plt.hist(newAntDataLogFiltered[startIndex:endIndex], bins = bins)

print(n)
print('FIRST BIN: ' + str(n[0]))
print('LAST BIN: ' + str(n[-1]))
plt.xlabel('Filtered Power (dBm)', labelpad = 15, **label_font)
plt.ylabel('Counts', **label_font)
plt.gca().set_yscale('log')
plt.savefig(dataDir + 'AntennaMedianNormalized_Histogram.png', bbox_inches = 'tight', dpi = 100)
plt.show()
