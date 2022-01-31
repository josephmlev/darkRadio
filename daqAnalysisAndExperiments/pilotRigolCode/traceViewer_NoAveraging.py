import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from scipy.stats import rayleigh
import os
import sys
import plottingClass


def errVar(mu4, std, n):
	return np.sqrt(1./n*(mu4 - (n - 3.) / (n - 1.) * std**4))

def firstDifference(arr):
	return [x[0] - x[1] for x in zip(arr[1:], arr[:-1])]

fileNameList = []
for file in os.listdir('./'):
	if file.endswith('.CSV'):
		fileNameList.append(file)

dataObjects =[]
stdVals = []
timeVals = []
centerVals = []
totalSTD = []

for aFile in fileNameList:
	holder = plottingClass.plottingClass(color = 'blue')
	holder.setData(fileName = aFile)
	holder.removeFirstPoint()
	holder.setData(xData = 'mult 1E-6', yData = 'mult 1E9')
	totalTime =  (float(holder.getDescription().split()[0][0:-1]))
	if not(holder.centerFrequency in centerVals):
		centerVals.append(holder.centerFrequency)
		dataObjects.append([])
		#stdVals.append([])
		#timeVals.append([])

	index = centerVals.index(holder.centerFrequency)
	dataObjects[index].append((holder, totalTime))



dataObjects = [x for _,x in sorted(zip(centerVals,dataObjects))]
dataObjects[0].sort(key = lambda x: x[1])
centerVals.sort()


holderTraceVals = [0]*len(dataObjects[0][0][0].y)
correctVals = []

for cfIndex in range(0, len(dataObjects)):
	for counter, val in enumerate(dataObjects[cfIndex]):
		for traceVal in zip(val[0].y, holderTraceVals):
			correctVals.append((counter + 1) * traceVal[0] - counter * traceVal[1])
		#print 'BEFORE: ' + str(holderTraceVals[0]) + ' CURRENT: ' + str(val[0].y[0]) + ' FUTURE: ' + str(correctVals[0])
		holderTraceVals = val[0].y
		val[0].y = correctVals
		#val[0].y = firstDifference(firstDifference(correctVals))
		#val[0].x = val[0].x[1:-1]
		correctVals = []
	holderTraceVals = [0]*len(dataObjects[0][cfIndex][0].y)

std = []
timeVals = []
errorVals = []

for val in dataObjects[0]:
	
	stdVal = np.std(val[0].y)
	mu4 = 3*stdVal**4
	errorVar = errVar(mu4, stdVal, 9999)
	errorVals.append(1/(2*stdVal)*errorVar)
	std.append(stdVal)

	timeVals.append(val[1])
	plt.plot(val[0].x, val[0].y, 'b-', linewidth = 2)
	plt.xlabel('Time (s)', fontsize=24, labelpad = 15)
	plt.ylabel('Amplitude (nV)', fontsize = 24)
	#plt.ylim([28, 31.3])
	plt.title(str(round(val[1], 2)) + 's')
	plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
	plt.gca().tick_params(labelsize=16)
	plt.show()
	plt.clf()
plt.rc('font', family='serif', weight = 'bold')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

plt.errorbar(timeVals, std, yerr = errorVals, fmt = 'o', color = 'blue', linewidth = 2)
plt.xlabel('Time (s)', fontsize=24, labelpad = 15)
plt.ylabel('Standard Deviation (pV)', fontsize = 24, labelpad = 15)

plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().tick_params(labelsize=16)
plt.show()




	