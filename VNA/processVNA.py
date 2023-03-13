import argparse
import glob
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import sys


# Various font names
label_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'normal',
          'verticalalignment':'bottom'} 
title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 


# Calculate VSWR from S11/S22 Magnitude
def calcVSWR(arr):
	return (1 + np.asarray(10**(arr/20.))) / (1 - np.asarray(10**(arr/20.)))

def getManufacturerData(ampName, colName):
	if not(os.path.exists('./ManufacturerData/')):
		print('COULD NOT FIND ManufacturerData IN CURRENT DIRECTORY')
		return np.empty(1), np.emtpy(1)
	
	if ampName.index('-') < ampName.index('_'):
		findName = ampName[:ampName.index('-')]
	else:
		findName = ampName[:ampName.index('_')]

	manFiles = []
	
	for aFilename in glob.glob('./ManufacturerData/' + '*' + str(findName) + '*'):
		manFiles.append(aFilename)	

	if len(manFiles) == 0:
		print('NO MANUFACTURER DATA FOUND')
		return np.empty(1), np.empty(1)

	
	allData = pd.read_csv(manFiles[0])

	freqs = np.asarray(allData['Frequency'])/1E6 
	if colName in allData:
		manData = np.asarray(allData[colName])
		if 'S11' in colName or 'S22' in colName:
			manData = calcVSWR(manData)
		return freqs, manData 

	print('COULD NOT FIND COLUMN IN MANUFACTURER DATA')
	return freqs, np.empty(1)


def isNumber(x):
	try:
		float(x)
		return True 
	except:
		return False 

def correctRangeSetting(aStr):
	print('A STRING ' + str(aStr))
	if not(aStr[0] == '('):
		return False
	if not(aStr[-1] == ')'):
		return False 
	if not(',' in aStr):
		return False 

	holder = aStr[1:-1].split(',')
	if len(holder) == 2 and isNumber(holder[0]) and isNumber(holder[1]):
		return True 

	return False

# Various command-line options
parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs = 1, help = 'Data file directory', type = str)
parser.add_argument('filename', nargs = '+', help = "Strings to find in file names", type = str)
#parser.add_argument('-d_d', '--dir', help= "Name of save directory", type = str, required = False, default = './' )
#parser.add_argument('-v', '--verbose', help= "Print information about conversion process", required = False, action='store_true')
parser.add_argument('-k', '--keys', help = 'Print all the column names in .csv file', required = False, action = 'store_true')
parser.add_argument('-cols','--columns', nargs='*', help='Set columnns to plot', required=False)
parser.add_argument('-ov', '--overlay', help = 'Overlay plots. Default is to separate', required = False, default = False, action = 'store_true')
parser.add_argument('-s', '--save', help = 'Save data to .png', required = False, action = 'store_true')
parser.add_argument('-sd', '--savedir', help = 'Give save directory default is ./Plots/. Will create if does not already exist', required = False, default = './Plots/')
parser.add_argument('-man', '--man', help = 'Get associated manufacturer data', default = False, action = 'store_true')
parser.add_argument('-xra', '--xrange', help = 'Set x-range requires(xmin, xmax) setting', required = False)
parser.add_argument('-yra', '--yrange', help = 'Set x-range requires (ymin, ymax) setting', required = False)

args = parser.parse_args()


# Checks to see if data directory / files exist
if not(os.path.exists(args.directory[0])):
	print('DATA FILE DIRECTORY NOT FOUND TRYING CURRENT DIRECTORY')
elif not(os.path.exists('./' + args.directory[0])):
	sys.exit(1)
elif os.path.exists('./' + args.directory[0]):
	print('DATA FILES FOUND IN CURRENT DIRECTORY')
	args.directory[0] = './' + args.directory[0]

if args.save:
	if not(os.path.exists(args.savedir)):
		 os.mkdir(args.savedir, mode = 0o777)
		 print('MADE NEW SAVE DIRECTORY AT ' + str(args.savedir))


xLimFlag = False
xLim = (0, 0)
yLimFlag = False
yLim = (0, 0)
if args.xrange:
	if(correctRangeSetting(args.xrange)):
		holder = (args.xrange)[1:-1].split(',')
		print(holder)
		xLim = (float(holder[0]), float(holder[1]))
		if xLim[0] > xLim[1]:
			print('LOWER X-LIMIT HIGHER THAN UPPER X-LIMIT EXITING')
			sys.exit(1)
		print('SET X-RANGE TO ' + str(xLim))
		xLimFlag = True
	else:
		print('INCORRECTLY SET X-RANGE EXITING...')
		sys.exit(1)

if args.yrange:
	if(correctRangeSetting(args.yrange)):
		holder = (args.yrange)[1:-1].split(',')
		yLim = (float(holder[0]), float(holder[1]))
		if yLim[0] > yLim[1]:
			print('LOWER Y-LIMIT HIGHER THAN UPPER Y-LIMIT EXITING')
			sys.exit(1)
		print('SET Y-RANGE TO ' + str(yLim))
		yLimFlag = True
	else:
		print('INCORRECTLY SET Y-RANGE EXITING...')
		sys.exit(1)


# Get all the files that match the string
processFiles = []


for val in args.filename:
	for aFilename in glob.glob(args.directory[0] + '*' + str(val) + '*'):
		processFiles.append(aFilename)	

print('FILES CONTAINING ' + str(args.filename[0]) + ' IN ' + str(args.directory[0]) + ": " + str(processFiles))
if len(processFiles) == 0:
	print('NO FILES FOUND')
	sys.exit(1)


if len(args.columns) == 0:
	print('NO DATA TO PLOT')
	sys.exit(1)

# If you just want to get the keys print all the keys in each of the found data files
if args.keys:
	for aFile in processFiles:
		data = pd.read_csv(aFile)
		# Removed 'UNNAMED COLUMN'
		data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
		print('KEYS IN ' + str(aFile))
		for aKey in data.keys():
			print('\t ' + str(aKey))
	sys.exit(1)


# Plot the data at each of the keys specified at the command line

allMan = []

allAmpNames = []
allCols = []


for aFile in processFiles:
		#ampName = aFile[aFile.rindex('/') + 1:aFile.rindex('.')]
		ampName = aFile[aFile.rindex('/')+1:aFile.rindex('.')]
		
		if ampName[:ampName.index('_')] not in allAmpNames:
			allAmpNames.append(ampName[:ampName.index('_')])

		if not(args.overlay):
			allMan = []

		if xLimFlag:
			plt.xlim(xLim)
		if yLimFlag:
			plt.ylim(yLim)	
		plt.xticks(fontsize=16)
		plt.yticks(fontsize=16)
		data = pd.read_csv(aFile)
		# Removed 'UNNAMED COLUMN'
		data = data.loc[:, ~data.columns.str.contains('^Unnamed')]		
		for aKey in args.columns:
			if aKey in data:
				goodMan = True
				plotData = np.asarray(data[aKey])
				if args.man:
					manFreqs, manAmps = getManufacturerData(ampName, aKey)
					if len(allMan) > 0:
						if len(manAmps) > 1:
							if np.sum([np.sum(np.abs([x - manAmps[:10]])) for x in allMan]) == 0:
								goodMan = False
						else:
							goodMan = False
					if len(manAmps) <= 1:
						goodMan = False

					if goodMan:
						allMan.append(manAmps[:10])


				if args.man and goodMan:
					if len(args.columns) <= 1:
						plt.plot(manFreqs, manAmps, label = 'Manufacturer ' + str(ampName[:ampName.index('-')]))
					else:
						plotTitle = str(ampName[:ampName.index('_')]) 
						if 'S21' in aKey:
							plotTitle += ' GAIN'
						elif 'S11' in aKey:
							plotTitle += ' INPUT VSWR'
						elif 'S22'  in aKey:
							plotTitle += ' OUTPUT VSWR'
						elif 'S12' in aKey:
							plotTitle += ' DIRECTIVITY'
						plt.plot(manFreqs, manAmps, label = 'Manufacturer ' + plotTitle)

				freqFlag = True
				if 'Frequency' in data.keys():
					freqs = np.asarray(data['Frequency'])/1E6
				else:
					freqs = np.asarray(range(len(plotData)))
					freqFlag = False
				plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font) if freqFlag else plt.xlabel('Index', labelpad = 15, **label_font)
				if 'S21_Magnitude' in aKey:
					if len(args.columns) <=  1:
						plt.plot(freqs[1:], plotData[1:], label = ampName)
					else:
						plt.plot(freqs[1:], plotData[1:], label = ampName + ' GAIN')

					if 'GAIN' not in allCols:
						allCols.append('GAIN')
					plt.ylabel('GAIN (dB)', **label_font)
					plt.title('GAIN ' + str(ampName), **title_font)
				elif 'S22_Magnitude' in aKey or 'S11_Magnitude' in aKey:
					plotData = calcVSWR(plotData)
					plt.ylabel('VSWR (x:1)', **label_font)
					if 'S11_Magnitude' in aKey:
						if len(args.columns) <= 1:
							plt.plot(freqs[1:], plotData[1:], label = ampName)
						else:
							plt.plot(freqs[1:], plotData[1:], label = ampName + ' INPUT VSWR')	
						plt.title('INPUT VSWR ' + str(ampName), **title_font)
						if not('INPUT VSWR' in allCols):
							allCols.append('INPUT VSWR')
					if 'S22_Magnitude' in aKey:
						if len(args.columns) <= 1:
							plt.plot(freqs[1:], plotData[1:], label = ampName)
						else:
							plt.plot(freqs[1:], plotData[1:], label = ampName + ' OUTPUT VSWR')
						if not('OUTPUT VSWR' in allCols):
							allCols.append('OUTPUT VSWR')
						plt.title('OUTPUT VSWR ' + str(ampName), **title_font)
				elif 'S12_Magnitude' in aKey:
					if 'S21_Magnitude' in data:
						if not('DIRECTIVITY' in allCols):
							allCols.append('DIRECTIVITY')
						
						if len(args.columns) <= 1:
							plt.plot(freqs[1:], (np.asarray(data['S21_Magnitude']) + np.asarray(data[aKey]))[1:], label = ampName)
						else:
							plt.plot(freqs[1:], (np.asarray(data['S21_Magnitude']) + np.asarray(data[aKey]))[1:], label = ampName + ' DIRECTIVITY')
						plt.title('DIRECTIVITY ' + str(ampName), **title_font)
						plt.ylabel('DIRECTIVITY (dB)', **label_font)
					else:
						if not('S12' in allCols):
							allCols.append('S12')
						if len(args.columns) <= 1:
							plt.plot(freqs[1:], (np.asarray(data[aKey]))[1:], label = ampName)
						else:
							plt.plot(freqs[1:], (np.asarray(data[aKey]))[1:], label = ampName + ' S12')
						plt.title('S12 ' + str(ampName), **title_font)
						plt.ylabel('S12 (dB)', **label_font)

				if args.man and goodMan:
					plt.legend(prop = legend_font)
				if not(args.overlay):
					if args.save:
						plt.tight_layout()
						plt.savefig(args.savedir + ampName + '.png')
						plt.gca().clear()
					else:
						plt.tight_layout()
						plt.show()
			else:
				print('COULD NOT FIND ' + str(aKey)  + ' COLUMN IN ' + str(aFile))
	
if args.overlay:
	plt.tight_layout()
	plt.legend(prop = legend_font)
	if args.save:
		plt.savefig(args.savedir + ampName[:ampName.rindex('_')] + '_Overlaid.png')
	else:
		print(allAmpNames)
		title = ''
		for val in allCols:
			title += val + ', '
		title = title[:-2] + ' '
		for val in allAmpNames:
			title += val + ', '
		title = title[:-2]
		plt.title(title, **title_font)
		plt.show()



