import getpass
import glob
import numpy as np
import serial
import serial.tools.list_ports
import sys
import time

def listDevices():
	""" Lists serial port names
    """
	# Get the current username
	username = getpass.getuser()

	# Get a list of available serial ports
	ports = serial.tools.list_ports.comports()

	# Print the list of ports and their descriptions
	for port, desc, hwid in sorted(ports):
	    print("Port found: {}: {} [{}]".format(port, desc, hwid))


	# Search for a specific keyword in the port description
	keyword = "valon"
	matching_ports = [(p.device, p.description) for p in ports if keyword.lower() in p.description.lower()]

	# Print the list of matching ports
	if matching_ports:
	    print("Matching ports found:")
	    for port, desc in matching_ports:
	        print("{}: {}".format(port, desc))
	    print()


def serial_ports():
    """ Set a serial port for a device

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

class SerialDevice:
	''' Generic serial device class that is trying to standardize interface with various serial instruments
	Attributes:
		baudrate: The baud rate of the device. Defaults to 9600.
		databits: The number of data bits. Defaults to 7.
		timeout:  Timeout of a command. Defaults to 3s.
		stopbits: The number of stop bits: Defaults to 1.
		parity:	  The parity: Defaults to even.
		term:	  Termination character. Defaults to \"\r\n\"
	'''
	def __init__(self,
		baudrate = 9600, 
		databits = 7, 
		timeout  = 3, 
		stopbits = 1,
		parity   = 'even', 
		term     = chr(13) + chr(10), 
		loc      = ''
		):
		self.baudrate = baudrate 
		self.databits = databits
		self.timeout = timeout
		self.term = term
		if stopbits == 1:
			self.stopbits = serial.STOPBITS_ONE
		elif stopbits == 1.5:
			self.stopbits = serial.STOBITS_ONE_POINT_FIVE
		elif stopbits == 2:
			self.stopbits = serial.STOPBITS_TWO 
		else:
			print('BAD STOPBITS SETTING TO 1')
			self.stopbits = serial.STOPBITS_ONE

		if parity.lower() == 'even':
			self.parity = serial.PARITY_EVEN
		elif parity.lower() == 'odd':
			self.parity = serial.PARITY_ODD
		else:
			self.parity = serial.PARITY_NONE


		if not(len(loc)):
			serialList = serial_ports()
			if len(serialList) == 0:
				print('COULD NOT FIND SERIAL PORT')
				sys.exit(1)
			else:
				self.deviceLoc = serialList[0]
		else:
			self.deviceLoc = loc 

		self.device = serial.Serial(
		  self.deviceLoc,
		  baudrate = self.baudrate, 
		  bytesize = self.databits, 
		  timeout = self.timeout, 
		  stopbits = self.stopbits, 
		  parity = self.parity, 
		)

	# Send a command to the device. Sleeptime is very long and can be shortened, but be careful
	# because it can corrupt the output from the device
	def sendCommand(self, command, sleepTime = 1):
		""" Send a command over the serial device
		:parameters:
			1. command:   str
				- A string representing the command to send to the device
			2. sleepTime: float
				- The amount of time to sleep between sending the command and checking for a response
        :raises:
        	Nothing
        :returns:
            A string either corresponding to the received command or a success/failure message indicating
            whether a command was sucessfull sent or not
    """
		(self.device).write((command + self.term).encode('utf-8'))
		time.sleep(sleepTime)
		out = []
		while (self.device).inWaiting() > 0:
			# Read byte-by-byte
			holder = (self.device).read(1)
			# Ignore whatever bits aren't specified as databits
			out.append(chr(ord(holder) & (2**(self.databits) - 1)))
		# Silly way to conver the list to a string
		x = ''
		if len(out) > 0:
			for c in out:
				x += c 
		
		if len(x) > 0:
			return x
		elif '?' in command:
			return 'ERROR: QUERY RETURNED NOTHING'
		else:
			return 'SENT COMMAND'

