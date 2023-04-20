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

