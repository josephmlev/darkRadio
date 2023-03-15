#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu March 9 2023

@author: Joseph Levine
"""


import serial 
import serial.tools.list_ports
import getpass
import os
import pyvisa







# Get the current username
username = getpass.getuser()

# Get a list of available serial ports
ports = serial.tools.list_ports.comports()

# Print the list of ports and their descriptions
for port, desc, hwid in sorted(ports):
    print("Port found: {}: {} [{}]".format(port, desc, hwid))


# Search for a specific keyword in the port description
keyword = "USB-Serial"

matching_ports = [(p.device, p.description) for p in ports if keyword.lower() in p.description.lower()]

# Print the list of matching ports
if matching_ports:
    print("Matching ports found:")
    for port, desc in matching_ports:
        print("{}: {}".format(port, desc))
    print()
else:
    print("No matching port found.")


######
# HARD CODE PORT NAME HERE
######
# Change the ownership of the serial port device file to the current user
os.system('sudo chown {0}:{0} /dev/ttyUSB1'.format(username))

######
# AND HERE
######
# Configure the serial connection
ser = serial.Serial('/dev/ttyUSB1', 9600, timeout =1)

def writeCmd(text):
    # Calculate the frequency tuning word for the desired frequency
    
    # Construct the command string to set the frequency
    command = str(text) + '\r'
    print(command)
    
    # Send the command to the device
    ser.write(command.encode('utf-8'))
    
    # Wait for the device to respond
    while True:
        response        = ser.readline()
        print(response)
        responseClean  = response.decode()
        responseStrip  = responseClean.strip()
        if not responseStrip:
            break
        print(f'Command Recieved: {responseStrip}')
        if 'error' in responseStrip:
            print('******** Error recieved from valon ********')
            raise ValueError
        


#ser.write(b'F 100000000 \r')
#print(ser.readline())
while True:
    cmd = input("Enter your command ")
    writeCmd(cmd)




