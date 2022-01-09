# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:06:17 2021

@author: phys-simulation
"""

import numpy as np
import time
import csv
import os
import configparser
import subprocess
import pyvisa as visa
import configparser
import serial as pyser
import sys, smtplib
from stat import S_IREAD, S_IWUSR
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart





# Monkey patch the del function in visa.Resource and visa.ResourceManager to stop them from issuing warnings...
old_del_Resource = visa.Resource.__del__
old_del_ResourceManager = visa.ResourceManager.__del__

def new_del_Resource(self):
    try:
        old_del_Resource(self)
    except Exception:
        pass

def new_del_ResourceManager(self):
    try:
        old_del_ResourceManager(self)
    except Exception:
        pass

def Send_Warning(message_subject, message_text, to_list):
    msg = MIMEMultipart()
    msg['From']='darkradioerrors@gmail.com'
    msg['Subject']=message_subject
    msg.attach(MIMEText(message_text,'plain'))
    server=smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('darkradioerrors@gmail.com','sourcescanvertical')
    for to_addr in to_list:
        msg['To']=to_addr
        text=msg.as_string()
        server.sendmail('darkradioerrors@gmail.com', to_addr, text)
        server.quit()
    return

# Description: Creates the configuration parser
# Inputs: The name of the configuration file
# Returns: A config parser object
def configure(fileName):
    configParser = configparser.RawConfigParser()  
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
        goUp = int(float(f.readline().rstrip()))
    return (currentFileNum, currentTime, centerFreq, goUp)

# Description: If you interrupt the run (say to change the batteries) this updates the
#   current state file
# Inputs: The name of the state file, the current file number, the total time that data
#  has been taken at the current frequency, and the current center frequency
# Returns: None
def saveCurrentState(fileName, currentFileNum, currentTime, centerFreq, goUp):
#os.chmod(fileName, 0o777)
# Allow the state file to be written
os.chmod(fileName, S_IWUSR)
with open(fileName, 'wt') as f:
f.write(str(currentFileNum) + '\n')
f.write(str(currentTime) + '\n')
f.write(str(centerFreq) + '\n')
f.write(str(goUp) + '\n')
# Set the state file to read only
os.chmod(fileName, S_IREAD)

def writeCenterFrequencies(fileName, centFreqArr):
with open(fileName, 'wt') as f:
os.chmod(fileName, S_IWUSR)
for val in centFreqArr:
f.write(str(val) + '\n')
os.chmod(fileName, S_IREAD)

# Description: Start a scan
# Inputs: The span of the next scan, the center frequency, and the number of averages
#  to be done on the Rigol
# Returns: None
def startScan(span, centerFrequency, numAverages):
    
    # Set the span (page 2-141)
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
    #return [np.sqrt(10**((float(x) - 30.)/10.)*50) for x in holderVals]
    return [float(x) for x in holderVals]

# Description: Reads a value from the Arduino-controlled thermistor
# Inputs: None
# Returns: Temperature reading
def getTemp():
port = '/dev/ttyACM0'
ser = pyser.Serial(port, 9600, timeout=1)
startChar = '<'
endChar = '>'
abortChar = '^'
listenForStart = True
listenForData = True
currentTime = time.time()
MAX_TIME = 3000

while listenForStart == True:
holder = ser.readline()
readChar = (holder.decode("utf-8").strip())
if time.time() - currentTime > MAX_TIME:
ser.write(abortChar)
elif readChar == startChar:
ser.write(endChar.encode())
listenForStart = False
else:
print('DID NOT HEAR START...')
#time.sleep(1)


while listenForData == True:
holder = ser.readline()
ser.write(ser.write(endChar.encode()))
listenForData = False
return (holder.decode("utf-8").strip())


# Description: Save the trace data as a CSV file - the format is equivalent to the output
#   from the Agilent spectrum analyzer
# Inputs: A file name and a description for the data
# Returns: None
def saveFile(fileName, description, data, resolution):
# Get the current date
DATE = (time.strftime("%m/%d/%Y") + "  " + time.strftime("%H:%M:%S"))

TITLE = 'TITLE: '
MODEL = 'MODEL: ' + ",RSA5065,,"
print(fileName)
f = open(fileName, 'wt')
try:
writer = csv.writer(f)
writer.writerow((DATE, '', ''))
writer.writerow(('Title:', ' ', ''))
writer.writerow(('Model:', 'RSA5065', ''))
writer.writerow(('Serial Number:', 'RSA5B192000021', ''))
writer.writerow(('Center Frequency:', str(int(centerFrequency)), 'Hz'))
writer.writerow(('Span:', str(int(span)), 'Hz'))
writer.writerow(('Resolution Bandwidth', str(resolution), 'Hz'))
writer.writerow(('Video Bandwidth', str('NA'), 'Hz'))
writer.writerow(('Reference Level: ', str(RLEV[0:-3]), '-dBm'))
writer.writerow(('Acquisition Time: ', ACQ_TIME, 'Sec'))
writer.writerow(('Num Points: ', str(SWEEP_POINTS), ''))
writer.writerow(('Description: ', str(description)))
writer.writerow(('', '', ''))
writer.writerow(('', 'Trace 1', ''))
writer.writerow(('Hz' , 'dBm'))
# Write the frequency data
for counter, item in enumerate(data):
writer.writerow((str((centerFrequency - (span / 2.)) + counter*(span / (len(data) - 1.))), str(item)))
finally:
f.close()

# Set the save files as read only to prevent accidental deletion - also prevents you
# from overwriting traces
os.chmod(fileName, S_IREAD)

with open(SAVE_TEMP,'a+') as f:
f.write(str(DATE) + '\t' + str(fileName) + '\t' + str(getTemp()) + '\n')






#shutdownList =['dapolin@ucdavis.edu','bpgodfrey@ucdavis.edu','tyson@physics.ucdavis.edu','tyson.physics@gmail.com','hillbrand@ucdavis.edu','jmlev@ucdavis.edu','sklomp@ucdavis.edu']
shutdownList = ['bpgodfrey@ucdavis.edu']
warningList = ['bpgodfrey@ucdavis.edu']

configVals = configure('CONFIG.txt')

# Set the acqusition time
ACQ_TIME = float(configVals.get('Configuration file', 'ACQ TIME'))*0.001

# Set the IP address
ADDRESS = configVals.get('Configuration file', 'ADDRESS')

# Set the maximum frequency
END_FREQUENCY = float(configVals.get('Configuration file', 'END FREQUENCY'))*10**6

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


SAVE_CENT_FILE = 'centFreq.txt'

SAVE_DIR = 'Data'

SAVE_TEMP = 'tempData.txt'

# Set the total amount of time to stay at each window (in minutes)
TOTAL_TIME = float(configVals.get('Configuration file', 'TOTAL TIME'))*60


# Read the contents of the status file
START_FILE, currentTime, counter, goUp = getCurrentState('status.txt')


centFreqArr = []
if os.path.isfile(SAVE_CENT_FILE):
with open(SAVE_CENT_FILE) as f:
for line in f:
centFreqArr.append(float(line))
centerFrequency = centFreqArr[counter]
firstGo = False
print(centFreqArr)
else:
subprocess.call(['rm', '-f', './' + SAVE_DIR + '/' + '*.CSV'])
firstGo = True
centerFrequency = 50000000
counter = 0



# Set the span based off the center frequency and required resolution
# This is the solution to (center frequency - 0.5*span) * resolution / span = ratio
span = centerFrequency / (RATIO / RESOLUTION + 0.5)

# Open the instrument (note that this is not static and needs to be reset every time you
# restart the script
RM = visa.ResourceManager('@py')
INST = RM.open_resource(ADDRESS)

# Get the number of sweep points (2-161)
SWEEP_POINTS  = int(INST.query(':SWE:POIN?'))

# Set the Rigol into real time spectrum analyzer mode (2-89)
INST.write(':INST:NSEL RTSA')

# Rigol suggests sleeping for 8 seconds after issuing the previous command
time.sleep(8)

# Set the detector into sample detect mode (2-134)
INST.write('SENS:DET:FUNC SAMP')

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

# Set the y-scale per division level to 5 dBm (2-64)
INST.write(':DISP:WIND:TRAC:Y:PDIV ' + str(10))

# Set the window to be rectangular (2-123)
INST.write(':BAND:SHAP KAIS ')

# Set the acquisition time in seconds (2-116)
INST.write(':ACQ:TIME ' + str(ACQ_TIME))

# Set the analyzer to be in the normal measurement state (2-44)
INST.write(':CONF:NORM')

# Set the trace to average (2-198)
INST.write(':TRAC:MODE AVER')
INST.write(':TRAC:TYPE AVER')

# Print various information about the spectrum analyzer settings

#print(INST.query("*IDN?"))
#print('UNIT: ' + str(INST.query('UNIT:POW?')))
#print('SINGLE MODE: ' + str(INST.query(':INIT:CONT?')))
#print('SWEEP POINTS: ' + str(INST.query(':SENSE:SWEEP:POINTS?')))
#print('RF ATTENUATION: ' + str(INST.query(':SENS:POW:RF:ATT?')))
#print('EXTERNAL GAIN: ' + str(INST.query(':SENSe:CORR:SA:RF:GAIN?')))
#print('INPUT IMPEDANCE: ' + str(INST.query(':SENS:CORR:IMP?')) + str(' OHMS'))
#print('SWEEP POINTS: ' + str(INST.query(':SWE:POIN?')))
#print('ACQUISITION TIME: ' + str(INST.query(':ACQ:TIME?')))

#print('CENTER FREQUENCY: ' + str(centerFrequency))
#print('CURRENT TIME: ' + str(currentTime))
#print('TOTAL TIME: ' + str(TOTAL_TIME))
#print('SPAN: ' + str(span))
#print('RESOLUTION: ' + str(RESOLUTION))
#print('NUM AVERAGES: ' + str(NUM_AVERAGES))



# Array to store the trace
vals = [0]*SWEEP_POINTS


# Current trace number
currentTrace = START_FILE
try:
longSleep = True
# Start the scan
startScan(span, centerFrequency, NUM_AVERAGES)
if firstGo:
centFreqArr.append(centerFrequency)
#currentTime = currentTime + ACQ_TIME*NUM_AVERAGES
while currentTime < TOTAL_TIME:

# Check if the current average count is greater than the requested number
# of averages (2-119)
try:
if float(INST.query(':TRAC:AVER:COUN:CURR?')) > NUM_AVERAGES:
# Get the trace data (2-192) and convert it to voltage units
holderVals = convert(INST.query('TRAC:DATA? TRACE1'))
#holderVals = INST.query('TRAC:DATA? TRACE1')
# Alternative way of storing the data as a moving average
#for counter, i in enumerate(holderVals):
# vals[counter] = vals[counter] + (i - vals[counter])/(currentTrace+1)
#print 'DONE WITH ' + str((currentTime)) + 'S OF DATA'
print('CENTER FREQUENCY: ' + str(centerFrequency))
print('TOTAL DATA: ' + str(currentTime))

description = str(ACQ_TIME*NUM_AVERAGES) + 'S OF DATA '  + str(GAIN) + ' dB gain'
resBand = float(INST.query(':SENS:BAND:RES?'))
saveFile('./' + SAVE_DIR + '/' + PREFIX + '_' + str(currentTrace) + '.CSV', description, holderVals, resBand)

saveCurrentState(SAVE_FILE, currentTrace, currentTime, counter, float(bool(goUp)))
# Update the trace number
currentTrace = currentTrace + 1
#print('COUNTER: ' + str(counter))
#print('LENGTH OF CENT ARR: ' + str(len(centFreqArr)))
#print(centFreqArr)
if firstGo:
if END_FREQUENCY <= centerFrequency + span/2.:
firstGo = False
currentTime = currentTime + ACQ_TIME*NUM_AVERAGES
goUp = False
counter = len(centFreqArr) - 1
writeCenterFrequencies(SAVE_CENT_FILE, centFreqArr)
else:
centerFrequency = centerFrequency + span - (centerFrequency - span/2.)*RESOLUTION
centFreqArr.append(centerFrequency)

span = centerFrequency / (RATIO / RESOLUTION + 0.5)
longSleep = True

# Begin a new scan
startScan(span, centerFrequency, NUM_AVERAGES)
else:
if goUp:
if counter < len(centFreqArr) - 1:
counter = counter + 1
centerFrequency = centFreqArr[counter]
else:
goUp = False
currentTime = currentTime + ACQ_TIME*NUM_AVERAGES
centerFrequency = centFreqArr[-1]
counter = len(centFreqArr) - 1

else:
if counter > 0:
counter = counter - 1
centerFrequency = centFreqArr[counter]
else:
goUp = True
currentTime = currentTime + ACQ_TIME*NUM_AVERAGES
centerFrequency = centFreqArr[0]
counter = 0

span = centerFrequency / (RATIO / RESOLUTION + 0.5)
longSleep = True

# Begin a new scan
startScan(span, centerFrequency, NUM_AVERAGES)
else:
# Sleep while waiting for data to complete
if longSleep:
print('SLEEPING FOR: ' + str(NUM_AVERAGES*ACQ_TIME) + ' S')
time.sleep(NUM_AVERAGES*ACQ_TIME)
longSleep = False
else:
# Don't keep pinging the Rigol if it's not done taking data
time.sleep(0.2)

except visa.VisaIOError as e:
print('THERE WAS A TIMEOUT ERROR')
Send_Warning("WARNING: Dark Radio Timeout", "The dark radio script has experienced a Visa timout with the follow error: " + str(e) + '\n Some info about the warning: \n' + str(e.args) + '\n' + str(RM.last_status) + '\n' +  str(RM.visalib.last_status) , warningList)


# Update the center frequency, span, and reset the total integration time
# Overlap of 2 bins between previous scan and current scan

print('END FREQUENCY: ' + str(END_FREQUENCY))
print('CURRENT END: ' + str(centerFrequency + span/2.))

print('SPAN: ' + str(span))
centerFrequency = centerFrequency + span - (centerFrequency - span/2.)*RESOLUTION
span = centerFrequency / (RATIO / RESOLUTION + 0.5)
currentTime = currentTime + NUM_AVERAGES*ACQ_TIME


# If there is a keyboard interrupt, save the current state of the data. Note, that this does
# not save trace data
except (KeyboardInterrupt, ValueError):
print('YOU INTERRUPTED THE PROGRAM')
saveCurrentState(SAVE_FILE, currentTrace, currentTime, counter, float(bool(goUp)))
except Exception as e:
print('THE ERROR IS: ' + str(e))
print('ERROR ISSUING E-MAIL ALERT')
Send_Warning("WARNING: Dark Radio Shutdown", "The dark radio script has shut down unexpectedly with the follow error: " + str(e), shutdownList)


print('PROGRAM FINISHED SUCCESSFULLY')
saveCurrentState(SAVE_FILE, currentTrace, currentTime, counter, float(bool(goUp)))
