from scipy import signal
from scipy.stats import norm
import matplotlib.pyplot as plt 
import multiprocessing as mp
import numpy as np

label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
		  'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 


def applyFilter(cutoff, data, plotResponse = False):
	fs = len(data)
	#print(len(data[0]))
	fcNorm = 2./cutoff
	b, a = signal.butter(6,fcNorm, 'highpass', analog = False)

	if plotResponse:
		w, h = signal.freqz(b, a, worN = 2**18)
		plt.semilogx(1 / (w[1:] / (2*np.pi)), 20*np.log10(abs(h[1:])))
		plt.margins(0, 0.1)
		plt.xlim([10**5, 1])
		plt.gca().set_xticks([10**5, 10**4, 10**3, 10**2, 10**1, 10**0])
		plt.grid(which = 'both', axis = 'both')
		plt.axvline(cutoff, color = 'green')
		plt.xlabel('Number of Bins', labelpad = 15, **label_font)
		plt.ylabel('Amplitude (dB)', **label_font)
		plt.xticks(fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.savefig('FilteredResponse.png', bbox_inches = 'tight', dpi = 100)

		plt.show()
	return np.asarray(signal.filtfilt(b, a, data), dtype = 'float32')

def compGaussCDF(zscore):
	return norm.cdf(zscore)

# Calculate the integral of the data between succeeding step points.
def calcSigHeights(steps, mean, std):
	oldCDF = 0 
	sigPoints = []
	for val in steps:
		newCDF = compGaussCDF((val - mean)/std)
		integralVal = newCDF - oldCDF
		sigPoints.append(integralVal)
		oldCDF = newCDF
	sigPoints[0] = 0
	return np.asarray(sigPoints)

def genGaussData(noiseLength, logUnits = False):
	mean = 0
	std = 1

	# Generate Gaussian random noise
	noiseSamples = np.random.normal(loc = mean, scale = std, size = noiseLength)


	# Do FFT and normalize
	dataFFT = np.abs(np.fft.fft(noiseSamples, n = noiseLength, norm = None)[:int(noiseLength/2)+1])**2
	dataFFT = 1./(noiseLength**2)*np.asarray([x if (counter == 0 or counter == len(dataFFT) -1) else 2*x for counter, x in enumerate(dataFFT)])
	
	# Convert to dBm if needed
	if logUnits:
		dataFFT = 10*np.log10(dataFFT*1000)
	
	return dataFFT


def getSNR(fftLength, totalAvg):
	np.random.seed()
	#print('ON RUN ' + str(index))
	ampSig = 0.0002
	fs = 2**10*100
	# Set the mean to 0 for right now
	maxZ = 3 
	minZ = -3
	normMean = 0
	stepSize = fs / fftLength
	normSTD = 100
	sigSteps = np.arange(minZ * normSTD, maxZ * normSTD + 0.99*stepSize, stepSize)
	sigPoints = calcSigHeights(sigSteps, normMean, normSTD)
	#print(sigPoints)
	maxIndex = np.argmax(sigPoints)
	paddedSignal = ampSig * np.pad(sigPoints, (int(fftLength/4 - maxIndex), int(fftLength/4 - (len(sigPoints) - maxIndex - 1))), 'constant', constant_values=(0, 0))		
	#freqs = np.arange(0, fs/2 + stepSize, stepSize)
	#plt.plot(freqs, paddedSignal)
	#plt.show()
	noiseData = np.zeros(int(fftLength/2) + 1)
	for x in range(totalAvg):
		noiseData += genGaussData(fftLength)
	noiseData = noiseData/totalAvg + paddedSignal

	convSignal = np.convolve(noiseData, sigPoints[::-1], mode = 'same')
	filteredConv = applyFilter(10*2**(int(np.log2(fftLength)-10)), convSignal[int(fftLength/20):-int(fftLength/20)])
	#plt.plot(filteredConv)
	#plt.plot([np.argmax(filteredConv)], [max(filteredConv)], 'r*')
	guessSigma = np.std(filteredConv[int(len(filteredConv)/4):int(len(filteredConv)/2.3)])
	guessMean = np.mean(np.std(filteredConv[int(len(filteredConv)/4):int(len(filteredConv)/2.3)]))
	snrVal = ((max(filteredConv) - guessMean)/guessSigma)
	#plt.show()
	return snrVal

noiseLength = 2**10
# Initially assume that our resolution is 1ppm 
centerFreq = 100 # In MHz
stepSize = centerFreq/10**6 

freqs = np.linspace(centerFreq - stepSize*noiseLength/2, centerFreq + stepSize*noiseLength/2 + stepSize, int(noiseLength/2) + 1)
print(freqs[0:10])
print(freqs[-10:])
noiseData = genGaussData(noiseLength)
#print(np.std(noiseData)**2)

# Notice that the mean value of the FFT is equal to the variance (which is one) with
# some prefactors
print(np.mean(noiseData))
print(2/noiseLength)

print(np.std(noiseData))

plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.plot(freqs, noiseData)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power ', **label_font)
plt.title('Linear Power Spectrum of Band-Limited GWN', **title_font)
plt.show()


# Take some averages
totalAvg = 1000
noiseData = np.zeros(int(noiseLength/2) + 1)
for x in range(totalAvg):
	noiseData += genGaussData(noiseLength)

# Plot the spectrum
plt.plot(freqs[1:-1], ((noiseData/totalAvg))[1:-1])
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power (arb)', **label_font)
plt.title('Averaged Linear Power Spectrum of BLGWN', **title_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.show()

# Now let's add a signal in the frequency domain. We can do this because the DFT
# is power preserving. Assume that the signal is Gaussian, and normalized so that the total
# energy of the signal is a * standard deviation of the noise
# For right now, set the standard deviation to be equal to the 3x the 
# resolution of our FFT (1ppm) or 300 Hz. 
# We are also only going to compute to +/-3 standard deviations.

# Set the width to be wider to show the process

normSTD = stepSize * 3

# Set the mean to 0 for right now
normMean = 0

# Min and max z values
minZ = -3
maxZ = 3

# Total energy of our signal in units of standard deviation of the white noise
energyZ = 5

# Sample frequencies at multiples of our minimum sampling resolution.
# Right now, this is centered around 0, but can be shifted as needed.
steps = np.arange(minZ * normSTD, maxZ * normSTD + stepSize, stepSize)
print(steps)



oldCDF = 0
sigPoints = calcSigHeights(steps, normMean, normSTD)
plt.plot(steps*10**6, sigPoints, 'bo')
plt.xlabel('Frequency (Hz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.title('Sample Gaussian Signal', **title_font)
plt.show()

# See what happens if the signal is narrowed
normSTD = stepSize / 6
steps = np.arange(minZ * normSTD, maxZ * normSTD + 1.01*stepSize, stepSize)
sigPoints = calcSigHeights(steps, normMean, normSTD)
plt.plot(steps*10**6, sigPoints, 'bo')
plt.xlabel('Frequency (Hz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.title('Sample Gaussian Signal Narrow', **title_font)
plt.show()

# Overlap two signals to see the process
normSTDNarrow = stepSize / 6
normSTDWide = stepSize * 3
stepsNarrow = np.arange(minZ * normSTDNarrow, maxZ * normSTDNarrow + 0.99*stepSize, stepSize)
sigPointsNarrow = calcSigHeights(stepsNarrow, normMean, normSTDNarrow)
stepsWide = np.arange(minZ * normSTDWide, maxZ * normSTDWide + 0.99*stepSize, stepSize)
print(stepsWide)
sigPointsWide = calcSigHeights(stepsWide, normMean, normSTDWide)

plt.plot(stepsNarrow*10**6, sigPointsNarrow, 'bo', label = 'Narrow')
plt.plot(stepsWide*10**6, sigPointsWide, 'ro', label = 'Wide')

plt.xlabel('Frequency (Hz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.title('Sample Gaussian Signals Narrow/Wide', **title_font)
plt.legend(prop = legend_font)
plt.show()

# Now let's align the maximum bin with our frequency axis
maxIndex = np.argmax(sigPointsWide)
offset = freqs[int(len(freqs)/2)]*10**6 - stepsWide[maxIndex] 
plt.plot((stepsWide + offset)/10**6, sigPointsWide, 'ro')
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Sample Gaussian Signal With Frequency Offset', **title_font)
plt.show()


# Increase the length of the signal to be the same as the noise signal
noiseLength = 2**10

# Set the total energy of the signal
ampSig = 5*(2/noiseLength)

paddedSignal = ampSig  * np.pad(sigPointsWide, (int(noiseLength/4 - maxIndex), int(noiseLength/4 - (len(sigPointsWide) - maxIndex - 1))), 'constant', constant_values=(0, 0))
noiseData = genGaussData(noiseLength)
centerFreq = 100 # In MHz
stepSize = centerFreq/10**6 
freqs = np.linspace(centerFreq - stepSize*noiseLength/2, centerFreq + stepSize*noiseLength/2 + stepSize, int(noiseLength/2) + 1)
plt.plot(freqs, paddedSignal + noiseData)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('0-Padded Sample Gaussian Signal With Frequency Offset', **title_font)
plt.show()



# Now we actually have to do an FFT. I'm going to do this 'correctly' so we don't have
# to arbitrarily add in multiplicative factors. And I'm going to assume perfect mixing.
# Assume we want a 1 ppm signal at 100 MHz and are doing a length 2^10 FFT. We need
# a resolution of 100 Hz. So, set our sampling rate at (2**10*100) is about 100 kHz.
# At the end, we'll offset everything.

fftLength = 2**10
fs = 2**10*100
stepSize = fs / fftLength

freqs = np.asarray(range(int(fftLength/2)+1)) * stepSize
offset = 1E8 - np.median(freqs)
freqs += offset

noiseData = genGaussData(fftLength)

plt.plot(freqs/10**6, noiseData)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Power Spectrum of BLGWN', **title_font)
plt.show()


# Add in a signal now
normSTD = 50 # In Hz

# Set the mean to 0 for right now
normMean = 0

# Min and max z values
minZ = -3
maxZ = 3

# Total energy of our signal (arb)
ampSig = 0.0001


sigSteps = np.arange(minZ * normSTD, maxZ * normSTD + 0.99*stepSize, stepSize)
sigPoints = calcSigHeights(sigSteps, normMean, normSTD)
maxIndex = np.argmax(sigPoints)
paddedSignal = ampSig * np.pad(sigPoints, (int(fftLength/4 - maxIndex), int(fftLength/4 - (len(sigPoints) - maxIndex - 1))), 'constant', constant_values=(0, 0))


totalData = noiseData + paddedSignal
plt.plot(freqs/10**6, totalData)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Gaussian White Noise With Signal', **title_font)
plt.show()

# Cannot see the signal very well, so do some averaging
noiseData = np.zeros(int(fftLength/2) + 1)
totalAvg = 2**10*2**6
for x in range(totalAvg):
	noiseData += genGaussData(fftLength)

noiseData = noiseData/totalAvg + paddedSignal

plt.plot(freqs/10**6, noiseData)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Gaussian White Noise With Signal ' + str(totalAvg) + ' Averages', **title_font)
plt.show()


convSignal = np.convolve(noiseData, sigPoints[::-1], mode = 'same')
plt.plot(freqs/10**6, convSignal)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Matched Filtered Gaussian White Noise With Signal ' + str(totalAvg) + ' Averages', **title_font)
plt.show()

filteredConv = applyFilter(10, convSignal[10:-10])
plt.plot(freqs[10:-10]/10**6, filteredConv)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Filtered Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Filtered Matched Filtered Gaussian White Noise With Signal ' + str(totalAvg) + ' Averages', **title_font)
guessSigma = np.std(filteredConv[int(len(filteredConv)/4):int(len(filteredConv)/2.3)])
guessMean = np.mean(np.std(filteredConv[int(len(filteredConv)/4):int(len(filteredConv)/2.3)]))
plt.axvline(x = freqs[int(len(filteredConv)/2.3)]/10**6, color = 'r')
plt.axvline(x = freqs[int(len(filteredConv)/4)]/10**6, color = 'r')
print('SNR: ' + str(round((max(filteredConv) - guessMean)/guessSigma, 4)))
plt.show()


# Same thing but with a much longer FFT
fftLength = 2**10
numAvg = 2**16

stepSize = fs / fftLength
freqs = np.asarray(range(int(fftLength/2)+1)) * stepSize
noiseData = genGaussData(fftLength)
offset = 1E8 - np.median(freqs)
freqs += offset

sigSteps = np.arange(minZ * normSTD, maxZ * normSTD + 0.99*stepSize, stepSize)
sigPoints = calcSigHeights(sigSteps, normMean, normSTD)
maxIndex = np.argmax(sigPoints)
paddedSignal = ampSig * np.pad(sigPoints, (int(fftLength/4 - maxIndex), int(fftLength/4 - (len(sigPoints) - maxIndex - 1))), 'constant', constant_values=(0, 0))

totalData = noiseData + paddedSignal
plt.plot(freqs/10**6, totalData)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Gaussian White Noise With Signal ' + str(fftLength) + '-Point FFT', **title_font)
plt.show()

'''
totalData = np.zeros(int(fftLength/2) + 1)
for fftStep in range(numAvg):
	totalData += genGaussData(fftLength)


totalData = totalData/numAvg + paddedSignal

convSignal = np.convolve(totalData, sigPoints[::-1], mode = 'same')
# Convolve with the time reversed copy of the padded signal
plt.plot(freqs/10**6, convSignal)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Convolved Power', **label_font)
plt.title('Long FFT After Applying Matched Filter', **title_font)
plt.show()


filteredConv = applyFilter(10*2**6, convSignal[10*2**6:-10*2**6])
plt.plot(freqs[10*2**6:-10*2**6]/10**6, filteredConv)
plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
plt.ylabel('Filtered Power', **label_font)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.title('Filtered Matched Filtered Gaussian White Noise With Signal ' + str(numAvg) + ' Averages', **title_font)
guessSigma = np.std(filteredConv[int(len(filteredConv)/4):int(len(filteredConv)/2.3)])
guessMean = np.mean(np.std(filteredConv[int(len(filteredConv)/4):int(len(filteredConv)/2.3)]))
print('SNR: ' + str(round((max(filteredConv) - guessMean)/guessSigma, 4)))
plt.axvline(x = freqs[int(len(filteredConv)/2.3)]/10**6, color = 'r')
plt.axvline(x = freqs[int(len(filteredConv)/4)]/10**6, color = 'r')
plt.show()
'''


totalIts = 500
fftLength = 2**10
totalAvg = 2**10*2**6

#getSNR(2**10, 2**16)
pool = mp.Pool(processes=mp.cpu_count())
returnedObs = [pool.apply_async(getSNR, args=(fftLength, totalAvg)) for x in range(totalIts)]
pool.close()
pool.join()
snrVals = [x.get() for x in returnedObs]
print(snrVals)

print('MEAN SNR: ' + str(np.mean(snrVals)))
print('MEDIAN SNR: ' + str(np.median(snrVals)))
print('MAX SNR: ' + str(max(snrVals)))
print('MIN SNR: ' + str(min(snrVals)))
print('STD OF SNR: ' + str(np.std(snrVals)))


totalIts = 500
fftLength = 2**16
totalAvg = 2**10

pool = mp.Pool(processes=mp.cpu_count())

returnedObs = [pool.apply_async(getSNR, args=(fftLength, totalAvg)) for x in range(totalIts)]
pool.close()
pool.join()
snrVals = [x.get() for x in returnedObs]
print(snrVals)

print('MEAN SNR: ' + str(np.mean(snrVals)))
print('MEDIAN SNR: ' + str(np.median(snrVals)))
print('MAX SNR: ' + str(max(snrVals)))
print('MIN SNR: ' + str(min(snrVals)))
print('STD OF SNR: ' + str(np.std(snrVals)))