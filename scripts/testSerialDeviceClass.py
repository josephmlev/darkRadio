import serialdevice as sd 
import time 



# A test of the serial device class 

# Returns reading in Kelvin
KELVIN_COMMAND = 'KRDG?'

# Returns reading in Fahrenheit
FAHRENHEIT_COMMAND = 'FRDG?'

# Returns reading in Celsius
CELSIUS_COMMAND = 'CRDG?'


# Method to get temperature
# Takes in serial object and a character flag depending on whether you want Kelvin (K), Celsius (C), or Fahrenheit (F)
def getTemp(ser, charFlag):
	command = FAHRENHEIT_COMMAND if charFlag == 'F' else KELVIN_COMMAND
	temp = ser.sendCommand(command, sleepTime = 0.1)
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


# Get list of available devices
sd.listDevices()


multimeter = sd.SerialDevice(loc = '/dev/ttyUSB0')
lakeshore = sd.SerialDevice(loc = '/dev/ttyUSB1', parity = 'odd')

print('MULTIMETER INFO: ' + str(multimeter.sendCommand('*IDN?')))
print('LAKESHORE INFO: ' + str(lakeshore.sendCommand('*IDN?')))

for i in range(3):
	print('THE MEASURED RESISTANCE IS: ' + str(round(float(multimeter.sendCommand('MEAS:FRES? 10000, 0.01')),3)) + u'\u03A9')
	time.sleep(0.5)




# Reset the Lakeshore back to how it was configured at power up
lakeshore.sendCommand('*RST')

# Setting into voltage mode - setting temperature range from 0 to 475K
lakeshore.sendCommand('ANALOG 0,4')
# Setting the input type to Pt1000
print(lakeshore.sendCommand('INTYPE 4;INTYPE?'))
 #Set the user input curve - must be 21 (4-19)
 # Set to the user curve
print('SETTING INPUT CURVE')
print(lakeshore.sendCommand('INCRV 21;INCRV?'))

for i in range(3):
	print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(lakeshore, 'F')) + u'\N{DEGREE SIGN}F')
	time.sleep(1)





