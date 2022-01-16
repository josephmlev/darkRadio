#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:37:21 2019
communicates with serial port and asks for the most recent bytes. Arduino
updates the temperature as specified in "temp_sense_arduino.ino"
@author: jmlevine
"""
import serial as pyser
import io
import time

def get_temp():
    #Establish a connection with arduino. Port will probably need to be changed.
    #You can find a list of available ports with terminal command:5
    #python -m serial.tools.list_ports
    #/dev/ttyACM0        
    #/dev/ttyS0 
    port = '/dev/ttyACM0'
    ser = pyser.Serial(port, 9600, timeout=2)
    
    #now that connection is made, ser.readline() 
    ser.readline()
    
    #convert from utf-8 to string
#    decoded_bytes = (T[0:len(T)-2].decode("utf-8"))
    
    #print(T)
    #return(decoded_bytes)
        
    #print(unpack(T, b''))
    #print(T)
    
    #may want to close after every loop to improve stability
    ser.close()




#from time import sleep
#while True:
#   get_temp()
#    sleep(1)

s = pyser.Serial(port = '/dev/ttyACM0',  baudrate=9600, bytesize=pyser.EIGHTBITS, parity=pyser.PARITY_NONE, stopbits=pyser.STOPBITS_ONE, timeout=1)
#s = pyser.Serial('/dev/ttyACM0', 9600, timeout = 2)
#s.stopbits = 1

#s_io = io.TextIOWrapper(io.BufferedRWPair(s, s, 1), newline = '\r\n', line_buffering = True)
time.sleep(2)
print(s.readline())
#s.close()
