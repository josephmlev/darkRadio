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

# Change the ownership of the serial port device file to the current user
os.system('sudo chown {0}:{0} /dev/ttyUSB0'.format(username))

# Configure the serial connection
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout =1)

def writeCmd(text):
    # Calculate the frequency tuning word for the desired frequency
    
    # Construct the command string to set the frequency
    command = str(text) + '\r'
    
    # Send the command to the device
    ser.write(command.encode('utf-8'))
    
    # Wait for the device to respond
    while True:
        response        = ser.readline()
        responseClean  = response.decode()
        responseStrip  = responseClean.strip()
        if not responseStrip:
            break
        print(f'Command Recieved: {responseStrip}')
        if 'error' in responseStrip:
            print('******** Error recieved from valon ********')
            raise ValueError
        
    
    # Check for errors in the response
    #if response != b'OK\n':
    #    raise Exception('Error setting frequency: ' + response.decode('utf-8'))

#ser.write(b'F 100000000 \r')
#print(ser.readline())
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







import sys
import serial
import serial.tools.list_ports

class VSerialPort( serial.Serial ):

    portLines = []
    portLineCount = 0
    portLineIndex = 0
    
    def __init__( self ):
        # Call the base constructor
        serial.Serial.__init__( self )

        portList = []
        for port, desc, hwid in serial.tools.list_ports.comports():
            print( 'Port: ', port, ' Desc: ', desc, ' HwId: ', hwid )
            if ( hwid.find( "FTDI" ) != -1 ):       # on Windows PC
                portList.append( port )
            elif ( desc.find( "Future" ) != -1 ):   # on Raspberry Pi
                portList.append( port )
            else:
                print('asdf')
                portList.append( port )

        if ( len( portList ) == 0 ):
            print( "No FTDI com ports are available" )
            return
        
        self.baudrate = 9600
        self.timeout = 1.0
        self.port = portList[ 0 ]
        self.open()

        self.write( '\r' )
        self.readAll()
        if ( len( self.portLines ) == 0 ):
            self.baudrate = 115200
            self.write( '\r' )
            self.readAll()
            if ( len( self.portLines ) == 0 ):
                print( "Can't communicate with 5009" )
                # exit()

        print( "Using " + self.port )

        # ----- End of Constructor -----
                
    def writeline( self, text ):
        print( text )
        if ( not self.isOpen() ):
            return
        self.write( text + '\r' )

        
    def readAll( self ):
        # Prepare the array to hold the incoming lines of text
        del self.portLines[:]   # clear input array   
        self.portLineCount = self.portLineIndex = 0

        if ( not self.isOpen() ):
            return

        text = self.readline()
        while ( 1 ):
            if ( text == "" ):
                return 
            sys.stdout.write( text )

            self.portLines.append( text )
            self.portLineCount += 1

            # Stop reading when we get a prompt 
            if ( len( text ) == 5 ):
                if text[ 4 ] == '>':
                    sys.stdout.flush()
                    return

            text = self.readline()

    # Read from the array of previously-received lines of text
    def lineGet( self ):
        i = self.portLineIndex
        self.portLineIndex += 1
        if ( self.portLineIndex > self.portLineCount ):
            return ''
        return self.portLines[ i ]

#valon = VSerialPort()
#valon.writeline('f1000000000')



'''
rm = pyvisa.ResourceManager()
rm.list_resources()
rm.open_resource('USB0')'''

'''
source = b'1'
freq = b'1.7G'
ser.write(b"S " + (source) + b"; f " + (freq))'''


'''
#Finding Valon resource name
unplug = input('make sure valon is unplugged, press enter to confirm')
l1 = list(rm.list_resources())
plugin = input('plug in valon, press enter to confirm')
l2 = list(rm.list_resources())

difference = list(set(l2) - set(l1))
valonName = difference[0]
print(valonName)
inst = rm.open_resource(valonName)
#----------------
#Valon Control
while True:
    source = input("Choose source (1 or 2): ")
    freq = input("Input desired frequence, add H,M,G for units, ex: 1.7G = 1.7 GHz: ")
    att = input("Input desire attenuation, The attenuator has a range of 0dB to 31.5dB in 0.5dB steps: ")
    print(inst.query("S " + source + "; f " + freq))
    print(inst.query("S " + source + "; att " + att))
'''