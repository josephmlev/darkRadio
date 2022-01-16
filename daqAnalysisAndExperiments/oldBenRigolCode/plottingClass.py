import matplotlib.pyplot as plt
import os
import csv 
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class plottingClass:
	def __init__(self, label = 'NOTHING', xData = [], yData = [], linewidth = 2.0, lines = 1, points = 0, pointSize = 20, color = 'black'):
		self.x = []
		self.y = []
		for val in zip(xData, yData):
			self.x.append(val[0])
			self.y.append(val[1])
	
		self.label = label
		self.linewidth = linewidth
		self.lines = lines
		self.points = points
		self.color = color
		self.pointSize = pointSize

		self.centerFrequency = 0
		self.data = 0
		self.span = 0
		self.resolutionBandwidth = 0
		self.videoBandwidth = 0
		self.referenceLevel = 0
		self.referenceUnits = 0
		self.sweepTime = 0
		self.sweepUnits = 0
		self.numPoints = 0
		self.description = 0

	def plot(self):
		if self.lines == 1:
			plt.plot(self.x, self.y, label=self.description, color = self.color, linewidth = self.linewidth)
		if self.points == 1:
			plt.scatter(self.x, self.y, label = self.description, s = self.pointSize, color = self.color)
	
	def getDescription(self):
		return self.description

	def removeFirstPoint(self):
		self.x = self.x[1:]
		self.y = self.y[1:]
	def setLegend(self):
		self.label = self.description
		plt.legend()

	def getXData(self, **kwargs):
		if 'all' in kwargs:
			return self.x
		elif 'min' in kwargs:
			return min(self.x)
		elif 'max' in kwargs:
			return max(self.x)

	def getYData(self, **kwargs):
		if 'all' in kwargs:
			return self.y
		elif 'min' in kwargs:
			return min(self.y)
		elif 'max' in kwargs:
			return max(self.y)

	def setAxes(self, **kwargs):
		if 'dataX' in kwargs:
			holderMinX = np.array(kwargs['dataX']).min()
			holderMaxX = np.array(kwargs['dataX']).max()
			minX = holderMinX - (0.1*(holderMaxX - holderMinX))
			maxX = holderMaxX + (0.1*(holderMaxX - holderMinX))
			plt.xlim(xmin = minX, xmax = maxX)
		if 'dataY' in kwargs:
			holderMinY = np.array(kwargs['dataY']).min()
			holderMaxY = np.array(kwargs['dataY']).max()
			minY = holderMinY - (0.1*(holderMaxY - holderMinY))
			maxY = holderMaxY + (0.1*(holderMaxY - holderMinY))
			plt.ylim(ymin = minY, ymax = maxY)
		if 'range' in kwargs:
				if kwargs['range'] == 'normal':
					minX = min(self.x) - (0.1*(max(self.x) - min(self.x)))
					maxX = max(self.x) + (0.1*(max(self.x) - min(self.x)))
					minY = min(self.y[1:]) - 0.01*(max(self.y[1:]) - min(self.y[1:]))
					maxY = max(self.y) + 0.05*(max(self.y) - min(self.y))
					plt.xlim(xmin = minX, xmax = maxX)
					plt.ylim(ymin = minY, ymax = maxY)
		
		if 'everything' in kwargs:
				holder = kwargs['everything']
				minX = min(holder[0]) - (0.1*(max(holder[1]) - min(holder[0])))
				maxX = max(holder[1]) + (0.1*(max(holder[1]) - min(holder[0])))
				minY = min(holder[2]) - 0.01*(max(holder[3]) - min(holder[2]))
				maxY = max(holder[3]) + 0.2*(max(holder[3]) - min(holder[2]))
				plt.xlim(xmin = minX, xmax = maxX)
				plt.ylim(ymin = minY, ymax = maxY)

	def plotText(self):
		minY, maxY = plt.gca().get_ylim()
		minX, maxX = plt.gca().get_xlim()
		data_minX = min(self.x)
		data_maxX = max(self.x)
		data_minY = min(self.y)
		data_maxY = max(self.y)
		holderString = "Min: " + str(data_minX) + " MHz"
		plt.text(minX + (maxX - minX)*0.85, minY + (maxY - minY)*0.9, holderString, fontsize = 14)
		holderString = "Max: " + str(data_maxX) + " MHz"
		plt.text(minX + (maxX - minX)*0.85, minY + (maxY - minY)*0.85, holderString, fontsize = 14)
		holderString = "RBW: " + str(int((self.resolutionBandwidth)/1.E2)/10.0) + " kHz"
		plt.text(minX + (maxX - minX)*0.85, minY + (maxY - minY)*0.8, holderString, fontsize = 14)
		holderString = "VBW: " + str(int((self.videoBandwidth)/1.E2)/10.0) + " kHz"
		plt.text(minX + (maxX - minX)*0.85, minY + (maxY - minY)*0.75, holderString, fontsize = 14)

	def changeData(self, **kwargs):
		if 'xData' in kwargs:
			holder = kwargs['xData']
			location = float(holder[0])
			multVal = float(holder[1])
			self.x[location] = self.x[location]*multVal
		elif 'yData' in kwargs:
			holder = kwargs['yData']
			location = holder[0]
			multVal = float(holder[1])
			self.y[location] = self.y[location]*multVal

	def setData(self, **kwargs):
		#Can take either a file name or set the data
		if 'xData' in kwargs:
			holder = kwargs['xData'].split()
			if len(holder) == 2:
				if 'mult' in holder:
					if is_number(holder[(holder.index('mult') + 1)%2]):
						multVal = float(holder[(holder.index('mult') + 1)%2])
						for counter, val in enumerate(self.x):
							self.x[counter] = val * multVal
					else:
						print is_number('10**-6')
						print '\'' + str(holder[(holder.index('mult') + 1)%2]) + '\' IS NOT A VALID MULTIPLICATION'
				if 'add' in holder:
					if is_number(holder[(holder.index('add') + 1)%2]):
						addVal = float(holder[(holder.index('add') + 1)%2])
						for counter, val in enumerate(self.x):
							self.x[counter] = val + addVal
					else:
						print 'NOT A VALID ADDITION'
		if 'yData' in kwargs:
			holder = kwargs['yData'].split()
			if len(holder) == 2:
				if 'mult' in holder:
					if is_number(holder[(holder.index('mult') + 1)%2]):
						multVal = float(holder[(holder.index('mult') + 1)%2])
						for counter, val in enumerate(self.y):
							self.y[counter] = val * multVal
					else:
						print '\'' + str(holder[(holder.index('mult') + 1)%2]) + '\' IS NOT A VALID MULTIPLICATION'
				if 'add' in holder:
					if is_number(holder[(holder.index('add') + 1)%2]):
						addVal = float(holder[(holder.index('add') + 1)%2])
						for counter, val in enumerate(self.y):
							self.y[counter] = val + addVal
					else:
						print 'NOT A VALID ADDITION'
		if 'fileName' in kwargs:
			print kwargs['fileName']
			if os.path.isfile(kwargs['fileName']):
				if kwargs['fileName'].endswith('.CSV'):
					with open(kwargs['fileName'], 'rb') as CSVFile:
					# Get more data from the CSV file
						reader = csv.reader(x.replace('\0', '') for x in CSVFile)
						dataPoint = True
						#lineReader = csv.reader(CSVFile, delimiter = ' ', quotechar = '|')
						for row in reader:
							#print row
							if len(row) > 1:
								if row[0].find('/') > -1:
									self.date = row[0].strip()
								elif row[0].find('Center Frequency') > -1:
									self.centerFrequency = int(row[1].strip())
									#print 'TEST: ' + str(self.centerFrequency)
								elif row[0].find('Span') > -1:
									self.span = int(row[1].strip())
								elif row[0].find('Resolution Bandwidth') > -1:
									self.resolutionBandwidth = int(row[1].strip())
								elif row[0].find('Video Bandwidth') > -1:
									if row[1].strip() == 'NA':
										self.videoBandwidth = -1
									else:
										self.videoBandwidth = int(row[1].strip())
								elif row[0].find('Reference Level') > -1:
									self.referenceLevel = float(row[1].strip())
									self.referenceUnits = row[2].strip()
								elif row[0].find('Acquisition Time') > -1:
									self.sweepTime = float(row[1].strip())
									self.sweepUnits = row[2].strip()
								elif row[0].find('Num Points') > -1:
									self.numPoints = int(row[1].strip())
								elif row[0].find('Description') > -1:
									self.description = row[1]
									print row[1]
							for val in row:
								if not(is_number(val)):
									dataPoint = False

							if len(row) > 0 and dataPoint == True:
								self.x.append(float(row[0]))
								self.y.append(float(row[1]))
								
							dataPoint = True

				elif kwargs['fileName'].endswith('.txt'):
					aFile = open(kwargs['fileName'],'r')
					lines = aFile.readlines()[1:]
					for val in lines:
						(self.x).append(float((val.split())[0]))
						(self.y).append(float((val.split())[1]))
					aFile.close()
