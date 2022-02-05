import numpy as np
import matplotlib.pyplot as plt 
import sys 

def calcPDF(x):
	return 1/(np.pi * np.sqrt(1. - x**2))

def calcInt(binEdges):
	return [100/np.pi*(np.arcsin(val[0]) - np.arcsin(val[1])) for val in zip(binEdges[1:], binEdges[:-1])]

inputFile = sys.argv[1]
allData = []

with open(inputFile, 'r') as f:
	for counter, line in enumerate(f):
		try:
			allData.append(np.float32(line))
		except Exception as e:
			print 'FUCK UP!!!'
			print counter
			print line
		if counter%1000000 == 0:
			print('DONE WITH ' + str(counter) + ' LINES')
np.asarray(allData).tofile(inputFile[0:inputFile.index('.')] + '.bin')

