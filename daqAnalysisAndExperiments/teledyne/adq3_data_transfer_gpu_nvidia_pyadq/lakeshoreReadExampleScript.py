import codecs
import serial, time
import sys
import glob


# Returns reading in Kelvin
KELVIN_COMMAND = 'KRDG?'

# Returns reading in Fahrenheit
FAHRENHEIT_COMMAND = 'FRDG?'

# Returns the identity of the Lakeshore
IDENTITY_COMMAND = '*IDN?'

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

def sendCommand(ser, command):
	ser.flushInput()
	ser.flushOutput()
	ser.write((command + chr(13) + chr(10)).encode('utf-8'))
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

	print('LAKESHORE IDENTITY: ' + str(sendCommand(ser, IDENTITY_COMMAND)))
	for i in range(10):
		print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(ser, 'F')) + u'\N{DEGREE SIGN}F')
		time.sleep(1)


	for i in range(10):
		print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(ser, 'C')) + u'\N{DEGREE SIGN}C')
		time.sleep(1)

	for i in range(10):
		print(str(i+1) + ') THE TEMPERATURE IS: ' + str(getTemp(ser, 'K')) + 'K')
		time.sleep(1)

