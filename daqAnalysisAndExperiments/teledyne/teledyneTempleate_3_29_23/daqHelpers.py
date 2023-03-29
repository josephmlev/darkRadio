import serial 
import serial.tools.list_ports
import getpass
import os


def valonCom(command,
    verboseMode = 1):
    # Get the current username
    username = getpass.getuser()

    # Get a list of available serial ports
    ports = serial.tools.list_ports.comports()

    # Print the list of ports and their descriptions
    # Unused while only one valon. May be useful with a second one.
    #for port, desc, hwid in sorted(ports):
    #    print("Port found: {}: {} [{}]".format(port, desc, hwid))

    # Search for a specific keyword in the port description
    keyword = "valon"
    matching_ports = [(p.device, p.description) for p in ports if keyword.lower() in p.description.lower()]

    # Print the list of matching ports
    if matching_ports:
        for port, desc in matching_ports:
            if verboseMode == 1:
                print("Matching ports found:" +"{}: {}".format(port, desc))
    else:
        print("No matching port found.")
        raise Exception("Valon Not Found")

    # Change the ownership of the serial port device file to the current user
    os.system('sudo chown {0}:{0} /dev/ttyUSB0'.format(username))

    # Configure the serial connection
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout =.1 )

    # Construct the command string to set the frequency
    commandStr = str(command) + '\r'
    
    # Send the command to the device
    ser.write(commandStr.encode('utf-8'))
    
    responseList = []
    # Wait for the device to respond
    while True:
        response        = ser.readline()
        responseClean  = response.decode()
        responseStrip  = responseClean.strip()
        if not responseStrip:
            break
        if verboseMode: 
            print(f'Command Recieved: {responseStrip}')
        if 'error' in responseStrip:
            print('******** Error recieved from valon ********')
            raise Exception('Valon Error')
        responseList.append(responseStrip)
    if responseList == []:
        print("******** No response from Valon! ********")
        raise Exception('Valon Response Error')
    print()
    return responseList


class arduino():

    def __init__(self):


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

        'buggy logic if multple arduinos'
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
            portName = usrInput
            
        # Change the ownership of the serial port device file to the current user
        os.system(f'sudo chown {username}:{username} {portName}')
        self.ser = serial.Serial(portName, 115200)
    
    def switch(self,
            switchState
        ):
        '''
        switchState: bool
        '''
        if switchState == 0:
            self.ser.write(b'0b0')
            print('Switched 0')
        elif switchState == 1:
            self.ser.write(b'0b1')
            print('Switched 1')
        else:
            print('Invalid argument given to switch method of arduino class')


