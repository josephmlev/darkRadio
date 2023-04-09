import serial 
import serial.tools.list_ports
import getpass
import os
import pyvisa

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

while True:
    dataIn = input('enter to toggle')
    ser.write(b'0b0')
    print(0)
    dataIn = input('enter to toggle')
    ser.write(b'0b1')
    print(1)
