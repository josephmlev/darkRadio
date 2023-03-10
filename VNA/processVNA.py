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

#def getManufacturerData(fileString)

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


for aFile in processFiles:
		#ampName = aFile[aFile.rindex('/') + 1:aFile.rindex('.')]
		ampName = aFile[aFile.rindex('/')+1:aFile.rindex('.')]
		
		data = pd.read_csv(aFile)
		
		# Removed 'UNNAMED COLUMN'
		data = data.loc[:, ~data.columns.str.contains('^Unnamed')]		
		for aKey in args.columns:
			if aKey in data:
				plotData = data[aKey]
				if args.man:
					getManufacturerData(args.filename)

				freqFlag = True
				if 'Frequency' in data.keys():
					freqs = np.asarray(data['Frequency'])/1E6
				else:
					freqs = np.asarray(range(len(plotData)))
					freqFlag = False
				plt.xlabel('Frequency (MHz)', labelpad = 15, **label_font) if freqFlag else plt.xlabel('Index', labelpad = 15, **label_font)
				if 'S21_Magnitude' in aKey:
					plt.plot(freqs, plotData, label = ampName)
					plt.ylabel('GAIN (dB)', **label_font)
					plt.title('GAIN ' + str(args.filename), **title_font)
				elif 'S22_Magnitude' or 'S11_Magnitude' in aKey:
					plotData = calcVSWR(plotData)
					plt.plot(freqs[1:], plotData[1:], label = ampName)
					plt.ylabel('VSWR (x:1)', **label_font)
					if 'S11_Magnitude' in aKey:
						plt.title('INPUT VSWR ' + str(args.filename), **title_font)
					if 'S22_Magnitude' in aKey:
						plt.title('OUTPUT VSWR ' + str(args.filename), **title_font)
				
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
		plt.show()



