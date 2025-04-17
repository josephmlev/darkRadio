import serial 
import serial.tools.list_ports
import getpass
import os
import time

######### NOTE 4/17/25: This uses arduino code reverbChamberTesting/norrisArduino/arduinoStepperControl_4_10_25/

class arduino():
    def __init__(self):
        # Get the current username
        username = getpass.getuser()

        # Get a list of available serial ports
        ports = serial.tools.list_ports.comports()

        # Search for a specific keyword in the port description
        keyword = "ACM" #arduino uno doesn't have arduino in the serial name. This is a hack
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
    
    def writeCmd(self,
            cmd
        ):
        if isinstance(cmd, str):
            cmd = cmd.encode('utf-8')
        self.ser.write(cmd)
        # Initialize timer and wait for response.
        start_time = time.time()
        response = None
        
        while True:
            # Check if any data is available in the serial buffer.
            if self.ser.in_waiting > 0:
                # Read a line from Arduino (assumes the Arduino sends newline-terminated messages)
                response = self.ser.readline().strip()
                break
            
            # Check if more than 2 minutes (120 seconds) has elapsed.
            if time.time() - start_time > 120:
                print("TIMEOUT")
                return
            
            # Sleep briefly to prevent high CPU usage.
            time.sleep(0.1)
        
        # Print the received response. Decode it if it's in bytes.
        try:
            print(response.decode('utf-8'))
        except AttributeError:
            # In case response is already a string
            print(response)

        # decode response, and return the int (if it is there) or None if not
        decoded_response = response.decode('utf-8') if isinstance(response, bytes) else response
        if '=' in decoded_response:
            # Split on '=' and take the portion after it, then strip spaces.
            try:
                value = int(decoded_response.split('=')[-1].strip())
                return value
            except ValueError:
                return None
        else:
            return None

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

    def is_number(self,
        x
    ):
        try:
            float(x)
            return True
        except Exception as e:
            return False

    def getTemp(self
    ):
        """ Gets temp data from Arduino. Average 5 acqusitions together.
        Args:
        none
        Return:
        Average Temperature -- Average Termperature in Kelvin
        """
        totalTries = 0
        maxTries = 20
        totalAvg = 5
        tempCount = 0
        totalTemp = 0
        while tempCount < totalAvg:
            self.ser.write(b'0b2')
            possibleTemp = self.ser.readline()
            totalTries = totalTries + 1
            if(self.is_number(possibleTemp)):
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