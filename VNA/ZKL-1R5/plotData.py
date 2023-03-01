import matplotlib.pyplot as plt 
import numpy as np 


label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
          'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'24', 'color':'red', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'16', 'style':'normal'} 


def getData(filename):
	freqs = []
	data = []
	with open(filename, 'r') as f:
		for line in f:
			holder = line.split('\t')
			freqs.append(float(holder[0]))
			data.append(float(holder[1]))
	return freqs, data


def getClosest(val, arrX, arrY):
	return np.interp(val, arrX, arrY)


gainFile = 'ZKL-1R5_Gain.txt'
vswrFile = 'ZKL-1R5_VSWR.txt'
directivityFile = 'ZKL-1R5_Directivity.txt' 

freqsGain, ampsGain = getData(gainFile)
freqsVSWR, ampsVSWR = getData(vswrFile)
freqsDirectivity, ampsDirectivity = getData(directivityFile)
plotData = True

if plotData:
	plt.plot(np.asarray(freqsGain), ampsGain, 'b-', linewidth = 3)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Gain (dB)', **label_font)
	plt.tight_layout()
	plt.grid()
	plt.show()


	plt.plot(np.asarray(freqsDirectivity), ampsDirectivity, 'b-', linewidth = 3)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('Directivity (dB)', **label_font)
	plt.tight_layout()
	plt.grid()
	plt.show()

	plt.plot(np.asarray(freqsVSWR), ampsVSWR, 'b-', linewidth = 3)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font)
	plt.ylabel('VSWR (x::1)', **label_font)
	plt.tight_layout()
	plt.grid()
	plt.show()


findFreqs = [123, 145]

gainInterp = getClosest(findFreqs, freqsGain, ampsGain)
vswrInterp = getClosest(findFreqs, freqsVSWR, ampsVSWR)
directivityInterp = getClosest(findFreqs, freqsDirectivity, ampsDirectivity)

print('THE INTERPOLATED GAINS FOR FREQUENCIES ' + str(findFreqs) + ' ARE: ' + str(gainInterp) + ' dB')
print('THE INTERPOLATED VSWR FOR FREQUENCIES ' + str(findFreqs) + ' ARE: ' + str(vswrInterp))
print('THE INTERPOLATED DIRECTIVITIES FOR FREQUENCIES ' + str(findFreqs) + ' ARE: ' + str(directivityInterp) + ' dB')













