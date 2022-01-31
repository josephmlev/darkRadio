import numpy as np 
import time
import csv
import os
import sys
import visa
import ConfigParser
from stat import S_IREAD, S_IWUSR

# Description: Creates the configuration parser
# Inputs: The name of the configuration file
# Returns: A config parser object
def configure(fileName):
	configParser = ConfigParser.RawConfigParser()   
 	configParser.read(fileName)
 	return configParser

# Description: Read the current state of the run
# Inputs: The name of the state file
# Returns: A tuple consisting of the current file number, the amount of time data
#          has been taken for, and the current center frequency
def getCurrentState(fileName):
	# Set the state file to read only - this makes it less likely to accidentally delete it
	os.chmod(fileName, S_IREAD)
	with open(fileName) as f:
		# Remove any newline characters
		currentFileNum = int(f.readline().rstrip())
		currentTime = float(f.readline().rstrip())
		centerFreq = int(float(f.readline().rstrip()))
	return (currentFileNum, currentTime, centerFreq)

# Description: If you interrupt the run (say to change the batteries) this updates the
#			   current state file
# Inputs: The name of the state file, the current file number, the total time that data
#		  has been taken at the current frequency, and the current center frequency
# Returns: None
def saveCurrentState(fileName, currentFileNum, currentTime, centerFreq):
	#os.chmod(fileName, 0o777)
	# Allow the state file to be written
	os.chmod(fileName, S_IWUSR)
	with open(fileName, 'wt') as f:
		f.write(str(currentFileNum) + '\n')
		f.write(str(currentTime) + '\n')
		f.write(str(centerFreq) + '\n')
	# Set the state file to read only
	os.chmod(fileName, S_IREAD)

# Description: Start a scan
# Inputs: The span of the next scan, the center frequency, and the number of averages
#		  to be done on the Rigol
# Returns: None
def startScan(span, centerFrequency, numAverages):
	
	# Set the span (page 2-141)
	print 'THE SPAN IS: ' + str(span) 
	print 'THE CENTER FREQUENCY IS: ' + str(centerFrequency)
	INST.write(':SENS:FREQ:SPAN ' + str(span))

	# Set the center frequency (2-139)
	INST.write(':SENS:FREQ:CENT ' + str(centerFrequency))

	# Set the number of averages (2-118)
	INST.write(':SENS:AVER:COUN ' + str(numAverages))

	# Set the power unit to DBM (2-215) - note that linear units do not work
	INST.write(':UNIT:POW DBM')

# Description: Take the trace data in dbm and convert it to linear units
# Inputs: An array of powers in dBm
# Returns: An array of voltages in Volts
def convert(holderVals):
	holderVals = holderVals.split(',')
	return [np.sqrt(10**((float(x) - 30.)/10.)*50) for x in holderVals]
	#return [float(x) for x in holderVals]

# Description: Save the trace data as a CSV file - the format is equivalent to the output
#			   from the Agilent spectrum analyzer
# Inputs: A file name and a description for the data
# Returns: None
def saveFile(fileName, description, data):
	# Get the current date
	DATE = (time.strftime("%m/%d/%Y") + "  " + time.strftime("%H:%M:%S"))
	
	TITLE = 'TITLE:			'
	MODEL = 'MODEL:			' + ",RSA5065,,"
	print fileName
	f = open(fileName, 'wt')
	try:
		writer = csv.writer(f)
		writer.writerow((DATE, '', ''))
		writer.writerow(('Title:', ' ', ''))
		writer.writerow(('Model:', 'RSA5065', ''))
		writer.writerow(('Serial Number:', 'RSA5B192000021', ''))
		writer.writerow(('Center Frequency:', str(int(centerFrequency)), 'Hz'))
		writer.writerow(('Span:', str(int(span)), 'Hz'))
		writer.writerow(('Resolution Bandwidth', str(int(RATIO*400000)), 'Hz'))
		writer.writerow(('Video Bandwidth', str('NA'), 'Hz'))
		writer.writerow(('Reference Level: ', str(RLEV[0:-3]), '-dBm'))
		writer.writerow(('Acquisition Time: ', ACQ_TIME, 'Sec'))
		writer.writerow(('Num Points: ', str(SWEEP_POINTS), ''))
		writer.writerow(('Description: ', str(description)))
		writer.writerow(('', '', ''))
		writer.writerow(('', 'Trace 1', ''))
		writer.writerow(('Hz' , 'Volts'))
		# Write the frequency data
		for counter, item in enumerate(data):
			writer.writerow((str((centerFrequency - (span / 2.)) + counter*(span / (len(data) - 1.))), str(item)))
	finally:
		f.close()

	# Set the save files as read only to prevent accidental deletion - also prevents you 
	# from overwriting traces
	os.chmod(fileName, S_IREAD)







configVals = configure('CONFIG.txt')

# Set the acqusition time
ACQ_TIME = float(configVals.get('Configuration file', 'ACQ TIME'))*0.001

# Set the IP address
ADDRESS = configVals.get('Configuration file', 'ADDRESS')

# Set the maximum frequency
END_FREQUENCY = float(configVals.get('Configuration file', 'END FREQUENCY'))

# Set the preamplifier gain
GAIN = float(configVals.get('Configuration file', 'GAIN'))

# Set the number of averages done on the Rigol
NUM_AVERAGES = int(configVals.get('Configuration file', 'NUM AVERAGES'))

# Set the prefix on the saved files
PREFIX = configVals.get('Configuration file', 'PREFIX')

# Set the resolution over span ratio
RATIO = float(configVals.get('Configuration file', 'RATIO'))

# Set the resolution
RESOLUTION = float(configVals.get('Configuration file', 'RESOLUTION'))

# Set the relative level for the display
RLEV = configVals.get('Configuration file', 'RLEV')

# Set the name of status file
SAVE_FILE = configVals.get('Configuration file', 'SAVEFILE')

# Set the total amount of time to stay at each window (in minutes)
TOTAL_TIME = float(configVals.get('Configuration file', 'TOTAL TIME'))*60


# Read the contents of the status file
START_FILE, currentTime, centerFrequency = getCurrentState('status.txt')

# Set the span based off the center frequency and required resolution
# This is the solution to (center frequency - 0.5*span) * resolution / span = ratio
span = centerFrequency / (RATIO / RESOLUTION + 0.5) 

# Open the instrument (note that this is not static and needs to be reset every time you
# restart the script
#RM = visa.ResourceManager('@py')
RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)

# Get the number of sweep points (2-161)
SWEEP_POINTS  = int(INST.query(':SWE:POIN?'))


# Set the Rigol into real time spectrum analyzer mode (2-89)
#print INST.query('INST:NSEL?')
INST.timeout = 10000
try:
	INST.write(':INST:SEL RTSA')
except Exception as e: 
	print e

#INST.write_ascii_values('INST:NSEL ', 2)
# Rigol suggests sleeping for 8 seconds after issuing the previous command
#time.sleep(8)

# Set the detector into positive peak detect mode (2-134)
INST.write('SENS:DET:FUNC AVER')

# Set the external gain (2-130)
INST.write(':CORR:SA:GAIN ' + str(GAIN))

# Set the reference level for the y scale (2-65)
INST.write(':DISP:WIND:TRAC:Y:SCAL:SPAC LOG')

# Set the y-scale unit to dbm (2-215)
INST.write(':UNIT:POW DBM')

# Set into continuous mode (2-87)
INST.write(':INIT:CONT ON')

# Set the attenuation of the RF front-end attenuator to 0 dB (2-152)
INST.write(':SENS:POW:RF:ATT 0')

# Set the y-scale reference level (2-65)
INST.write(':DISP:WIND:TRAC:Y:SCAL:RLEV ' + str(RLEV))

# Set the window to be rectangular (2-123)
INST.write(':BAND:SHAP RECT')

# Set the acquisition time in seconds (2-116)
INST.write(':ACQ:TIME ' + str(ACQ_TIME))

# Set the analyzer to be in the normal measurement state (2-44)
INST.write(':CONF:NORM')

# Set the trace to average (2-198)
INST.write(':TRAC1:TYPE AVER')

# Print various information about the spectrum analyzer settings

#print INST.query("*IDN?")
#print 'UNIT: ' + str(INST.query('UNIT:POW?'))
#print 'SINGLE MODE: ' + str(INST.query(':INIT:CONT?'))
#print 'SWEEP POINTS: ' + str(INST.query(':SENSE:SWEEP:POINTS?'))
#print 'RF ATTENUATION: ' + str(INST.query(':SENS:POW:RF:ATT?'))
#print 'EXTERNAL GAIN: ' + str(INST.query(':SENSe:CORR:SA:RF:GAIN?'))
#print 'INPUT IMPEDANCE: ' + str(INST.query(':SENS:CORR:IMP?')) + str(' OHMS')
#print 'SWEEP POINTS: ' + str(INST.query(':SWE:POIN?'))
#print 'ACQUISITION TIME: ' + str(INST.query(':ACQ:TIME?'))

#print 'CENTER FREQUENCY: ' + str(centerFrequency)
#print 'CURRENT TIME: ' + str(currentTime)
#print 'TOTAL TIME: ' + str(TOTAL_TIME)
#print 'SPAN: ' + str(span)
#print 'RESOLUTION: ' + str(RESOLUTION)
#print 'NUM AVERAGES: ' + str(NUM_AVERAGES)

# Start the scan
startScan(span, centerFrequency, NUM_AVERAGES)


# Array to store the trace
vals = [0]*SWEEP_POINTS

# Current trace number
currentTrace = START_FILE
try:
	while centerFrequency < END_FREQUENCY:
		while currentTime < TOTAL_TIME:
			
			# Check if the current average count is greater than the requested number
			# of averages (2-119)
			if float(INST.query(':TRAC:AVER:COUN:CURR?')) > NUM_AVERAGES:
				# Get the trace data (2-192) and convert it to voltage units
				holderVals = convert(INST.query('TRAC:DATA? TRACE1'))
				
				# Alternative way of storing the data as a moving average
				#for counter, i in enumerate(holderVals):
				#	vals[counter] = vals[counter] + (i - vals[counter])/(currentTrace+1)
				
				# Update the current time
				currentTime = currentTime + ACQ_TIME*NUM_AVERAGES
				print 'DONE WITH ' + str((currentTime)) + 'S OF DATA' 
				description = str(ACQ_TIME*NUM_AVERAGES) + 'S OF DATA '  + str(GAIN) + ' dB gain'
				saveFile(PREFIX + '_' + str(currentTrace) + '.CSV', description, holderVals)
				
				# Update the trace number
				currentTrace = currentTrace + 1 

				# Begin a new scan
				if currentTime < TOTAL_TIME:
					startScan(span, centerFrequency, NUM_AVERAGES)
		
		# Update the center frequency, span, and reset the total integration time
		centerFrequency = centerFrequency + span
		span = centerFrequency / (RATIO / RESOLUTION + 0.5)
		currentTime = 0
		if centerFrequency < END_FREQUENCY:
			startScan(span, centerFrequency, NUM_AVERAGES)

# If there is a keyboard interrupt, save the current state of the data. Note, that this does
# not save trace data
except (KeyboardInterrupt, ValueError):
	print 'YOU INTERRUPTED THE PROGRAM'
	saveCurrentState(SAVE_FILE, currentTrace, currentTime, centerFrequency)
