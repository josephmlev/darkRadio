import serial 
import serial.tools.list_ports
import getpass
import os
import pyvisa
import time

# Get the current username
username = getpass.getuser()

# Get a list of available serial ports
ports = serial.tools.list_ports.comports()

# Search for a specific keyword in the port description
keyword = "ACM" #arduino uno doesnt have arduino in the name :( this is a hack
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
ser = serial.Serial(portName, 9600)


while True: #waits for an input command then sends it via serial connection to the arduino, also waits for serial data sent back from arduino
    command = input("Arduino command (desired position # / step / position / quit): ")
    ser.write(command.encode('utf-8'))
    time.sleep(1)
    data = ser.readline().decode('utf-8').strip()
    #if data:
    print(f"Arduino says: {data}")
    print('help')

    
    if command == 'quit':
        break
'''
while True:
        response        = ser.readline()
        responseClean  = response.decode('utf-8')
        responseStrip  = responseClean.strip()
        if not responseStrip:
            break
        print(f'Command Received: {responseStrip}')
        if 'error' in responseStrip:
            print('******** Error recieved from valon ********')
            raise ValueError'''