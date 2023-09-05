import serial 
import serial.tools.list_ports
import getpass
import os
import pyvisa
import time
import datetime

# Get the current username
username = getpass.getuser()

# Get a list of available serial ports
ports = serial.tools.list_ports.comports()

# Search for a specific keyword in the port description
keyword = "arduino"
matching_ports = [(p.device, p.description) for p in ports if keyword.lower() in p.description.lower()]

# Print the list of matching ports
if matching_ports:
    print('Ports with "arduino" found:')
    for port, desc in matching_ports:
        print(f'{port} : {desc}')


if len(matching_ports) > 1:
    print("Multple arduinos. Manually enter name")
    portName = input('>> ')
if len(matching_ports) == 0:
    print("No Arduinos found")
    exit()
    
print('Enter a port name. Ex /dev/ttyACM1')
print(f'Or press enter to use {desc} on {port}')
usrInput = input('>> ') 
if usrInput == "":
    portName = port
else:
    portName = useInput
	

# Change the ownership of the serial port device file to the current user
os.system(f'sudo chown {username}:{username} {portName}')
ser = serial.Serial(portName, 115200)

def is_number(x):
	try:
		float(x)
		return True
	except Exception as e:
		return False

def getTemp():
    """ Gets temp data from Arduino. Average 5 acqusitions together.
    Args:
    none
    Return:
    Average Temperature -- Average Termperature 
    """
    totalTries = 0
    maxTries = 20
    totalAvg = 5
    tempCount = 0
    totalTemp = 0
    while tempCount < totalAvg:
        ser.write(b'0b2')
        possibleTemp = ser.readline()
        totalTries = totalTries + 1
        if(is_number(possibleTemp)):
            totalTemp = totalTemp + float(possibleTemp)
            tempCount = tempCount + 1
        if totalTries > maxTries:
            print('TOO MANY READS TO GET TEMPERATURE')
            break
    if tempCount == 0:
        return 0
    else:
        return totalTemp / tempCount + 273.15
    return()

while True:
    print(datetime.datetime.now(), getTemp())
    time.sleep(5)
