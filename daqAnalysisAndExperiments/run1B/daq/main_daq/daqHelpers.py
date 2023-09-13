import serial 
import serial.tools.list_ports
import getpass
import os
import settings as s
import h5py
import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import time
import numpy as np

# Check if the data and plottingSpec subdirectories exist
# Make them otherwise 
if not os.path.isdir(os.path.join(s.SAVE_DIRECTORY, "data")):
    os.mkdir(os.path.join(s.SAVE_DIRECTORY, "data"))
if not os.path.isdir(os.path.join(s.SAVE_DIRECTORY, "plottingSpec")):
    os.mkdir(os.path.join(s.SAVE_DIRECTORY, "plottingSpec"))

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
    os.system(('sudo chown {0}:{0}'+ port).format(username))

    # Configure the serial connection
    ser = serial.Serial(port, 9600, timeout =.1 )

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

if s.SWITCH or s.TEMPERATURE:
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



def writeH5(specDict,
    runInfoDict,
    acqNum,
    switchPos,
    dataDir
):
    '''
    Given a spectrum and some info, packs an H5 file every numSpecPerFile
    spectra.
    
    Inputs:
    specDict         : dictionary of np arrays
        Dict containing arrays of spectra
    runInfoDict      : dictionary
        Dict containing info about the run
    acqNum          : int
        Number of acqusition since start of run
    numSpecPerFile  : int
        How many spectra to write per file. Should be set
        to keep files around 1GB (about 16 for two 2^24 FFTs)
    dataDir         : str
        directory to save data. Note this needs to be created ahead of time
        and there should be a subdirectory called data. 
    '''

    #number games to figure out what file to write
    mod         = acqNum%s.NUM_SPEC_PER_FILE
    fileNum     = int((acqNum-mod)/s.NUM_SPEC_PER_FILE)
    print(f"On acqNum: {acqNum}")

    if s.SAVE_H5:
        #create file object. Creates h5 file if needed, else appends to existing file ('a' flag) 
        fileName    = dataDir+'data/'+str(fileNum)+'.hdf5'
        f       = h5py.File(fileName, 'a') 
        
        #create new group object for each acqusition
        acqGrp  = f.create_group(str(acqNum))

        #pack spectra as dataseta
        for specName in specDict:
            acqGrp.create_dataset(specName, data=specDict[specName], dtype = 'f')
        
        #pack run into as attributes
        for infoName in runInfoDict:
            acqGrp.attrs[infoName] = runInfoDict[infoName]
        acqGrp.attrs['File Number'] = fileNum
        
        #if on a new file, make previous file read only. Note last file wont be read only
        if mod == 0 and fileNum != 0 and s.READ_ONLY_H5:
            os.chmod(dataDir+'data/'+str(fileNum-1)+'.hdf5', S_IREAD|S_IRGRP|S_IROTH)
    
        #write to database text file in dataDir
        if not os.path.exists(dataDir+'database.txt'):
            infoStr = ''
            for infoKey in runInfoDict:
                infoStr += infoKey + ','
            infoStr += 'File Number\n' 

            file = open(dataDir + 'database.txt', 'w')
            file.writelines(infoStr)
            file.close()

        lineToWrite = ''
        for infoKey in runInfoDict:
            infoData = str(runInfoDict[infoKey])
            lineToWrite += infoData + ', '
        lineToWrite += str(fileNum) +'\n'

        file = open(dataDir + 'database.txt', 'a')
        file.writelines(lineToWrite)
        file.close()
def avgAndConvertFFT(fftSum, numFft, s):
    #move from GPU to RAM
    fftSumCpu           = fftSum.cpu()
    avgSpec_fftsq       = np.zeros(np.shape(fftSumCpu))
    avgSpec_W           = np.zeros(np.shape(fftSumCpu))
    ###################AVERAGE AND CONVERT#################### 
    #average FFT^2 spectra
    avgSpec_fftsq[:,0]   = np.asarray(fftSumCpu[:,0]/numFft[0])
    avgSpec_fftsq[:,1]   = np.asarray(fftSumCpu[:,1]/numFft[1])
    
    #convert FFT^2 to power spectra (Watts)
    #P_W = V^2/R
    #V   = 2 * 1/len_FFT * FFT * 2^(ADC code to V) NOTE: 2 is because Rfft
    #P_W = 2 * FFT^2 * (adcCode2V)/R  
    adcCode2V           = 0.5/2**(8*s.BYTES_PER_SAMPLES) #max V/max code 
    avgSpec_W[:,0]      = 2 * avgSpec_fftsq[:,0] * (adcCode2V**2) /(50*s.CH0_RECORD_LEN**2)
    avgSpec_W[:,1]      = 2 * avgSpec_fftsq[:,1] * (adcCode2V**2) /(50*s.CH1_RECORD_LEN**2)

    specDict        = {'chASpec_W'  : avgSpec_W[:,0],
                    'chBSpec_W'     : avgSpec_W[:,1]}
    return specDict

def procFftAndSave(fftSum, 
                    numFft,
                    date_time_done,
                    switchPos,
                    temperature,
                    s):
    ti = time.time()

    # Get FFT^2 sum(s) from GPU buffer into RAM. 
    # Divide by number of FFTs to compute average
    # Return dictionary of spectra  
    specDict = avgAndConvertFFT(fftSum,
                                                numFft,
                                                s)

    #auto extract acquisition number
    if not os.path.exists(s.SAVE_DIRECTORY+'database.txt'):
        acqNum = 0
    else:
        acqNum = int(open(s.SAVE_DIRECTORY + 'database.txt', 'r'
                        ).readlines()[-1].split(',')[0].strip()) + 1 

    runInfoDict     = {'ACQ NUM'            : acqNum, #number of spectra since start of run. Must be first
                        'DATETIME'          : date_time_done,
                        'SWITCH_POS'        : switchPos,  
                        'ANT POS IDX'       : s.ANT_POS_IDX,
                        'TEMP'              : round(temperature,3),
                        'LEN FFT LOG2'      : int(np.log2(s.CH0_RECORD_LEN)),
                        'SAMPLE RATE MHZ'   : s.SAMPLE_RATE/1e6,
                        'NOF_BUFFERS'       : s.NOF_BUFFERS_TO_RECEIVE,
                        }
    if s.SAVE_AMP_CHAIN == 1: #option to make text file smaller without amp chain info
        runInfoDict.update(s.SETUP_DICT)

    if s.SAVE_W_SPEC == 1:
        # Check if the file already exists
        if os.path.exists(s.PATH_TO_SAVE_SINGLE_SPEC + '.npy'):
            # Prompt the user if they want to overwrite
            overwrite = input(f"{s.PATH_TO_SAVE_SINGLE_SPEC} already exists. Do you want to overwrite it? (yes/no) ").strip().lower()
            
            if overwrite == 'yes':
                np.save(s.PATH_TO_SAVE_SINGLE_SPEC, specDict)
            elif overwrite == 'no':
                # Ask for a new path and name
                new_path = input("Please provide a new path and name for the file: ")
                np.save(new_path, specDict)
            else:
                print("Invalid input. File was not saved.")
        else:
            # If the file doesn't already exist, save it as usual
            np.save(s.PATH_TO_SAVE_SINGLE_SPEC, specDict)

    writeH5(specDict,
        runInfoDict,
        acqNum,
        switchPos,
        s.SAVE_DIRECTORY)
    np.save(s.SAVE_DIRECTORY + 'plottingSpec/chA_W_switch' + str(switchPos), specDict['chASpec_W'])
    np.save(s.SAVE_DIRECTORY + 'plottingSpec/chB_W_switch' + str(switchPos), specDict['chBSpec_W'])

    # compute average spec for plotting
    if acqNum == 0 or acqNum == 1: #should not == 1, this is a hack that should be resolved. 3/23/23
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch' + str(switchPos) + '.npy', specDict['chASpec_W'])
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch' + str(switchPos) + '.npy', specDict['chBSpec_W'])
    else:
        pastSpecA = np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch' + str(switchPos) + '.npy')
        pastSpecB = np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch' + str(switchPos) + '.npy')
        avgSpecA  = (((acqNum - 1) * pastSpecA) + specDict['chASpec_W'])/acqNum
        avgSpecB  = (((acqNum - 1) * pastSpecB) + specDict['chBSpec_W'])/acqNum
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch' + str(switchPos) + '.npy', avgSpecA)
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch' + str(switchPos) + '.npy', avgSpecB)

    print(f"Processed previous acquisition in {time.time() -ti}")
