import codecs
import numpy as np
import pandas as pd
import serial, time
import sys
import glob


# Returns reading in Kelvin
KELVIN_COMMAND = 'KRDG?'

# Returns reading in Fahrenheit
FAHRENHEIT_COMMAND = 'FRDG?'

# Returns reading in Celsius
CELSIUS_COMMAND = 'CRDG?'


# Returns the identity of the Lakeshore
IDENTITY_COMMAND = '*IDN?'

# Filename of data table 
DATA_FILE = '100KThermistorTable.csv'

# Magic from online to find serial ports that is agnostic to OS type
# Taken from here: https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


# This is all basically copied from here: 
# https://stackoverflow.com/questions/35732628/character-conversion-errors-when-using-pyserial
# Unless you want to end up in debugging hell, I'd recommend not changing this. This has been
# verified through a good amount of trial and error.

def sendCommand(ser, command):
	#ser.flushOutput()
	ser.write((command + chr(10)).encode('utf-8'))
	time.sleep(0.1)
	out = []
	while ser.inWaiting() > 0:
		holder = ser.read(1)
		# Only keep 7 bits around
		out.append(chr(ord(holder) & 0x7F))
	x = ''
	if len(out) > 0:
		for c in out:
			x += c 
	return x
	

# Method to get temperature
# Takes in serial object and a character flag depending on whether you want Kelvin (K), Celsius (C), or Fahrenheit (F)
def getTemp(ser, charFlag):
	command = FAHRENHEIT_COMMAND if charFlag == 'F' else KELVIN_COMMAND
	temp = sendCommand(ser, command)
	if charFlag == 'K':
		return float(temp) 
	elif charFlag == 'C':
		return round(float(temp) - 273.15, 5)
	elif charFlag == 'F':
		# Converts from Kelvin to Fahrenheit, but there isn't enough precision on the Lakeshore to get
		# you the MSB
		#return round((float(temp) - 273.15)*9/5 + 32, 5)
		return round(float(temp), 5)
	else:
		print('NO IDEA WHAT UNITS YOU WANT - RETURNING KELVIN')
		return float(temp)

# Configure the device to be used as a 1000Ω Platinum thermistor
# with a custom voltage curve

def configureDevice(ser):
	# Set analog output parameter values (5-13)
	# 0: Voltage mode
	# 4: 0-475K
	sendCommand(ser, 'ANALOG 0,4')
	print('ANALOG: ' + str(sendCommand(ser, 'ANALOG?')))
	print('SETTING INPUT TYPE')
	
	# Set input type parameter (4-19)
	# 0: Silicon diode
	# 1: GaAlAs diode
	# 2: 100Ω Platinum/250
	# 3: 100Ω Platinum/500
	# 4: 1000Ω Platinum
	# 5: NTC RTD
	# Note: Setting this parameter is required for what units
	# 		you then feed into the user configuration table. It
	# 		also sets the current used for making a measurement
	#		(see table 1-2 on page 1-7). As of 3/17/23, I cannot 
	#		get a user configuration curve to work on input type 
	#		parameter 5 but have been sucessful with 4
	print(sendCommand(ser, 'INTYPE 4;INTYPE?'))
	
	# Set the user input curve - must be 21 (4-19)
	print('SETTING INPUT CURVE')
	print(sendCommand(ser, 'INCRV 21;INCRV?'))


# Load data from a .csv file
def loadData(ser, filename, keys):
	data = pd.read_csv(filename)
	parsedData = []
	for val in keys:
		if val in data.keys():
			parsedData.append(np.asarray(data[val]))
		else:
			print('COULD NOT FIND ' + str(val) + ' IN ' + str(filename))

	return tuple(parsedData)

# Load curve onto Lakeshore
def loadCurve(ser, res, temps):
	# Set curve points for the user curve (4-17)
	# Required arguments:
	#	Curve (21)
	#	Curve point index (between 1 and 200)
	# 	Resistor value (make sure to have the right units depending on input type) up to 6 digits
	#	Temperature value (in Kelvin) up to 6 digits
	for counter, val in enumerate(zip(temps, res)):
		writeStr = 'CRVPT 21,' + str(counter + 1) + ',' + str(round(val[1], 5)) + ',' + str(round(val[0], 3))
		print(str(counter+1) + ': ' + str(writeStr))
		sendCommand(ser, writeStr)
		print('SET CURVE POINT:')
		# Query the set curve point to make sure that it was appropriately set (4-17)
		print(str(sendCommand(ser, 'CRVPT? 21,' + str(counter+1))))

	# Issuing a reset command is required to final curve point entry (4-17)
	print(sendCommand(ser, '*RST'))
	time.sleep(1)


# NOTE: If you are using a gender-changer, you'll need a null modem as well to get 
# 		the correct pinout
if __name__ == '__main__':
	# Get all serial ports - choosing the first one
	allSerialPorts = serial_ports()

	# Setup for serial connection
	# Baud rate: 9600
	# Date length: 7
	# Stop bits: 1
	# Parity: odd
	ser = serial.Serial(
		  allSerialPorts[0],
		  baudrate = 9600, 
		  bytesize = 7, 
		  timeout = 3, 
		  stopbits = serial.STOPBITS_ONE, 
		  parity = serial.PARITY_ODD, 
		)

	# Check to see that the LakeShore is configured properly
	ident = (str(sendCommand(ser, IDENTITY_COMMAND)))
	
	if not(ident):
		print('COULD NOT TALK WITH DEVICE')
		sys.exit(1)

	print('LAKESHORE IDENTITY: ' + str(ident))

	# Reset Lakeshore   
	print(sendCommand(ser, '*RST'))
	
	ser.flushInput()
	ser.flushOutput()
	
	# Modify this if you want to change the device / curve type
	configureDevice(ser)

	

	# Set the curve header for the user input curve (4-16/4-17)
	# Format is user curve (21), name of curve, serial number, 
	# curve data format, temperature limit (ignored by device),
	# and temperature coefficient (1 is negative, 2 is positive).
	# I think the temperature coefficient is actually determined
	# by the device itself.
	# Curve data format specifier:
	#	2: V/K (required by Si, GaAlAs)
	#	3: Ω/K (required by 250 Pt, 500 Pt, 1000 Pt)
	#	4: log10 Ω / K (required by ntcrtd)
	#	From Table 3-1 on page 3-4
	print(sendCommand(ser, 'CRVHDR 21,DINGO,001,3,325,1'))
	print(sendCommand(ser, 'CRVHDR? 21'))

	
	# Get Lakeshore units (4-21)
	#	0: Kelvin
	#	1: Celsius
	#	2: Sensor units
	#	3: Fahrenheit
	#print('LAKESHORE UNITS: ' + str(sendCommand(ser, 'DISPFLD?')))

	# Get temperatures and resistances from data file
	temps, res = loadData(ser, DATA_FILE, ['Temp', 'Center'])

	

	# Convert temperature to Kelvin
	temps = np.asarray(temps) + 273.15	
	
	# Resistance values (kludge for using a 1kΩ resistor)
	res = 1/((1/(np.asarray(res)*1000)) + 1/1000.)
	
	for counter in range(len(res)):
		res[counter] = (res[counter] + (90 - counter)*3)
	#print(res)

	loadCurve(ser, res, temps)
	
	
	# After load curve, make sure to reset your input type and curve 
	configureDevice(ser)
	
	#print(sendCommand(ser, 'INTYPE 4;INTYPE?'))
	# Set the user input curve - must be 21 (4-19)
	#print('SETTING INPUT CURVE')
	#print(sendCommand(ser, 'INCRV 21;INCRV?'))


	# Read temperatures
	time.sleep(1)

	for i in range(10):
		print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(ser, 'F')) + u'\N{DEGREE SIGN}F')
		time.sleep(1)


	for i in range(10):
		print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(ser, 'C')) + u'\N{DEGREE SIGN}C')
		time.sleep(1)

	for i in range(10):
		print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(ser, 'K')) + 'K')
		time.sleep(1)
