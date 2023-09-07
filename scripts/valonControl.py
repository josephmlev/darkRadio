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
import time

# This is a very basic command line script that allows you to talk to a Valon synth
# It can handle multple Valons connected, but can only talk to one. If 
# you want to talk to multple at the same time, you can run multple 
# instances of this script

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
else:
    print("No matching port found.")

########
# A bunch of garbage logic to connect if multple valons are found
########
# If you have multple valons, name their serial numbers here
valonSN_lookup = {'5019'    : '12203141', #nickname:SN
                '5009'      : '12202784'
                }

if len(matching_ports)>1:
    print("Multple Valons found. Please select from the following list:")
    for valName in valonSN_lookup:
        print(valName)
    chosenValonName = input()
    if chosenValonName not in valonSN_lookup:
        print('Error! Chosen name not in provided valon SN lookup table')
        raise(ValueError)
    else:
        print(f'connecting to {chosenValonName}')
    
    # Search for a specific keyword in the port description
    keyword         = valonSN_lookup[chosenValonName]
    matching_ports  = [(p.device, p.description) for p in ports if keyword in p.description.lower()]

    for port, desc, hwid in sorted(ports):
        if keyword in hwid:
            port = port
            break
        else:
            continue

# Change the ownership of the serial port device file to the current user
os.system(('sudo chown {0}:{0} '+port).format(username))

# Configure the serial connection
ser = serial.Serial(port, 9600, timeout =.1)

def writeCmd(text):
    # Construct the command string to set the frequency
    command = str(text)
    
    # Send the command to the device
    # There is a slick way to do this, but I did it very explicit 
    ser.write(bytearray(command, encoding='utf-8') + b"\r")
    s = command.encode()
    
    # Wait for the device to respond
    while True:
        response        = ser.readline()
        responseClean  = response.decode('utf-8')
        responseStrip  = responseClean.strip()
        if not responseStrip:
            break
        print(f'Command Received: {responseStrip}')
        if 'error' in responseStrip:
            print('******** Error recieved from valon ********')
            raise ValueError
        
while True:
    cmd = input("Enter your command or h for examples ")
    print(cmd)
    if cmd == 'h':
        print('################Help mode################')
        print('see pg. 23: https://www.valontechnology.com/5009users/Valon_5009_opman.pdf')
        print('example: set frequency to 50MHz, source 2.')
        print('s2;f50m')
        print()
    writeCmd(cmd)





