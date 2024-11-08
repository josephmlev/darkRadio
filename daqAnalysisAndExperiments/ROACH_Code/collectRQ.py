from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool
import argparse
import bisect
import configparser
import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt 
import h5py as h5
import os
import pandas as pd
import sys 
"""
Convert the float parameters in the config file to floats
Parameters:
   aString: The string in the config file
Returns:
    holder (1D list): List of floats
"""
def getFloatBounds(aString):
    holder = aString[aString.index('[')+1:aString.index(']')]
    holder = holder.split(',')
    holder = [float(x) for x in holder]
    return holder

"""
Convert the date parameters in the config file to floats
Parameters:
   aString: The date string in the config file
Returns:
    holder (1D list): List of datetime objects written as month/day/year hour:minute:second.microseconds
"""
def getDateBounds(aString):
    holder = aString[aString.index('[')+1:aString.index(']')]
    holder = holder.split(',')
    holder = [datetime.strptime(x.strip(), '%m/%d/%Y %H:%M:%S.%f')  for x in holder]
    return holder

"""
Take in a config file and return a dictionary of bounds
Parameters:
   configFile: The config file
   configName: The name of the configuration setup
Returns:
    configDict (dictionary): Dictionary of bounds
"""
def getAllBounds(configFile, configName):
    configSetup = configparser.ConfigParser()
    configSetup.read(configFile)
    configDict = {}
    tempBounds = getFloatBounds(configSetup[configName]['Temperature'])
    freqBounds = getFloatBounds(configSetup[configName]['Frequency'])
    dateBounds = getDateBounds(configSetup[configName]['Date'])
    antWestBounds =  getFloatBounds(configSetup[configName]['AntennaWest'])
    antVertBounds = getFloatBounds(configSetup[configName]['AntennaVert'])
    antSouthBounds = getFloatBounds(configSetup[configName]['AntennaSouth'])
    antThetaBounds = getFloatBounds(configSetup[configName]['AntennaTheta'])
    antPhiBounds = getFloatBounds(configSetup[configName]['AntennaPhi'])
    measChoice = configSetup['TEST']['Measurement']
    configDict['Temp'] = tempBounds 
    configDict['Freq'] = freqBounds
    configDict['Date'] = dateBounds
    configDict['Ant'] = (antWestBounds, antVertBounds, antSouthBounds, antThetaBounds, antPhiBounds)
    configDict['Choice'] = measChoice
    return configDict

"""
Return the indices in the "database" that fall within the specified
bound. This function assumes that the database is ordered by the 
parameter being searched for.
Parameters:
   val: The value in the config file
   parsedList: List of the parameter being searched for (e.g. date/temperature/antenna position)
Returns:
    holderIndices: List of all the indices in the sorted list that fall in the bounds
"""
def parseOnce(val, parsedList):
    if len(parsedList) == 0:
        return []
    
    holderIndices = []
    if val[0] > val[1]:
        print('CONFUSED ORDERING')
        return []
    if val[0] == -1:
        startIndex = 0
    elif val[0] < parsedList[0]:
        startIndex = 0
    elif val[0] > parsedList[-1]:
        print('EMPTY LIST')
        return []
    else:
        startIndex = bisect.bisect_left(parsedList, val[0])
    if val[1] == -1:
        endIndex = len(parsedList)
    elif val[1] > parsedList[-1]:
        endIndex = len(parsedList)
    else:
        endIndex = bisect.bisect_right(parsedList, val[1])
    
    [holderIndices.append(x) for x in range(startIndex, endIndex)]    
    return holderIndices

"""
Obtain all the datasets that fall within the bounds given in the config
file.
Parameters:
   configFile: The name of the configuration file
   configName: The name of the configuration setup in the config file 
Returns:
    parsedList: List of all the parsed data files defined as a list of tuples
                of the form ((file number, save number), date, temperature, antenna position).)
                    - Date is a datetime object
                    - Antenna position is a tuple of the form (west, vertical, south, theta, phi)

"""
def getParsedList(configFile, configName):
    dbFile = '/group/tysongrp/SearchableDatabase.txt'
    allData = []
    configDict = getAllBounds(configFile, configName)
    with open(dbFile, 'r') as f:
        f.readline()
        for line in f:
            holder = line.split()
            holder = [x.replace(',', '') if counter > 0 else x for counter, x in enumerate(holder)]
            try:
                dateVal = datetime.strptime(holder[1] + ' ' + holder[2], '%Y-%m-%d %H:%M:%S.%f')
            except:
                 dateVal = datetime.strptime(holder[1] + ' ' + holder[2], '%Y-%m-%d %H:%M:%S')

            tempVal = float(holder[3])
            antPos = (float(holder[4][1:]), float(holder[5]), float(holder[6]), float(holder[7]), float(holder[8][:-1]))
            fileNum = float(holder[0][1:holder[0].index(',')])
            runNum = float(holder[0][holder[0].index(',')+1:-2])
            allData.append(((fileNum, runNum), dateVal, tempVal, antPos))
    
    holderIndices = []
    allData = sorted(allData, key = lambda x: x[1])
    parsedList = allData
    for val in np.reshape(configDict['Date'], (-1, 2)):
        [holderIndices.append(x) for x in parseOnce(val, [x[1] for x in parsedList])]
    
    holderIndices = np.asarray([*set(holderIndices)])
    parsedList = [parsedList[x] for x in holderIndices]
    

# configDict['Ant'] = (antWestBounds, antVertBounds, antSouthBounds, antThetaBounds, antPhiBounds)
# allData.append(((fileNum, runNum), dateVal, tempVal, antPos, antPos))

    for antSortVal in range(len(configDict['Ant'])):
        parsedList = sorted(parsedList, key = lambda x: x[3][antSortVal])
        holderIndices = []
        for val in np.reshape(configDict['Ant'][antSortVal], (-1, 2)):
            [holderIndices.append(x) for x in parseOnce(val, [x[3][0] for x in parsedList])]

        holderIndices = np.asarray([*set(holderIndices)])
        parsedList = [parsedList[x] for x in holderIndices]

    parsedList = sorted(parsedList, key = lambda x: x[2])
    holderIndices = []
    for val in np.reshape(configDict['Temp'], (-1, 2)):
        [holderIndices.append(x) for x in parseOnce(val, [x[2] for x in parsedList])]
    
    holderIndices = np.asarray([*set(holderIndices)])
    parsedList = [parsedList[x] for x in holderIndices]  
    
    parsedList = sorted(parsedList, key=lambda x: (x[0][0], x[0][1]))
    return parsedList, configDict
    #[print(x) for x in parsedList]



#num_arrays = 100
num_processes = mp.cpu_count()
num_simulations = 1000
sentinel = None

"""
Read the associated files in the parsed file list and store the data in
a queue, which is written to a file. All this is done using multiprocessing
speed up the read times. 
Parameters:
   readQueue: The queue that contains all the files to be written to a file. Stored in the following
              structure: (file number, list of data tuples)
                - tuple is of the form (measurement number, tuple of parameters)
                    o Tuple of parameters is of the form 
                      (save number, date, temperature, antenna position, frequency, measurement choice)
                        x Date is a datetime object
                        x Antenna position is a tuple of the form (west, vertical, south, theta, phi)
                        x Measurement choice is a string that is either 'Antenna', 'Terminator', or 'Both'

   output: The name of the output queue. Stored in the following structure: (data, tuple of parameters)
                - data is an numpy array of either antenna or terminator data or two arrays
                  if the measurement choice is both
                - Tuple paramters is of the form (measurement number, tuple of parameters) exactly
                  like the read queue
    
    dataDir: The directory where the data files are stored
Returns:
    Nothing. Puts data in a queue

"""
def collectData(readQueue, output, dataDir):
    freqStep = 6.0*10**8/2**23
    antData = []
    termData = []
    for parsedVal in iter(readQueue.get, sentinel):
        aFile = 'data_' + str(parsedVal[0]) + '.h5'
        #print(parsedVal)
        for writeData in parsedVal[1]:
            #print('TEST: ' + str(writeData[0]))
            dataset = pd.read_hdf(dataDir + aFile, key = 'measdata_'+ str(writeData[0])) 
            startFreq = writeData[1][3][0]
            endFreq = writeData[1][3][1]
            startIndex = int(startFreq*10**6 / freqStep)
            endIndex = int(endFreq*10**6/freqStep)
            if writeData[1][4] == 'Antenna' or writeData[1][4] == 'Both':
                antData = np.asarray(dataset[dataset.keys()[1]])[startIndex:endIndex] #the other one
            if writeData[1][4] == 'Terminator' or writeData[1][4] == 'Both':
                termData =  np.asarray(dataset[dataset.keys()[0]])[startIndex:endIndex] #term or bicon 
            if len(antData) == 0:
                output.put((termData, writeData))
            elif len(termData) == 0:
                output.put((antData, writeData))
            else:
                output.put((antData, termData, writeData))
    
            antData = []
            termData = []
       
"""
Write the data stored in the output queue to an .h5 file. Right now, only a single .h5 file
is saved, which can get large very quickly if too small of a subset of data are taken
Parameters:
   output: The name of the output queue. Stored in the following structure: (data, tuple of parameters)
                - data is an numpy array of either antenna or terminator data or two arrays
                  if the measurement choice is both
                - Tuple paramters is of the form (measurement number, tuple of parameters) exactly
                  like the read queue
    Returns:
    Nothing. Saves data to an .h5 in the same directory as where the script is called (change this)

"""
def writeFiles(output):
    #hdf = pt.openFile('simulation.h5', mode='w')
    aFile = h5.File('/group/tysongrp/RQTest.h5', 'w')
    first = True
    biconCounter = 0
    termCounter = 0
    while True:
        data = output.get()
        if first:
            freqStep = 6.0*10**8/2**23
            freqs = np.asarray(range(len(data[0])))*freqStep
            aFile.create_dataset('Freqs', data = freqs)
            first = False    
        if data:
            if data[-1][1][4] == 'Both':
                dataBicon = aFile.create_dataset('bicon_' + str(biconCounter), data=data[0])
                dataBicon.attrs['Date'] = str(data[-1][1][0])
                dataBicon.attrs['Run'] = str(data[-1][0])
                dataBicon.attrs['Antenna West'] = str(data[-1][1][2][0])
                dataBicon.attrs['Antenna Vertical'] = str(data[-1][1][2][1])
                dataBicon.attrs['Antenna South'] = str(data[-1][1][2][2])
                dataBicon.attrs['Antenna Theta'] = str(data[-1][1][2][3])
                dataBicon.attrs['Antenna Phi'] = str(data[-1][1][2][4])
                dataBicon.attrs['Temperature'] = str(data[-1][1][1])

                dataTerm = aFile.create_dataset('term_' + str(termCounter), data =data[1])
                
                dataTerm.attrs['Date'] = str(data[-1][1][0])
                biconCounter += 1
                termCounter += 1
        else:
            break
    #hdf.close()

if __name__ == '__main__':
    
    writeH5 = False
    # Name of the configuration file
    configFile = '/group/tysongrp/ConfigDR.ini'
    # Name of the setup in the configuration file
    configName = 'TEST'
    # Get a parsed list of file names and also saves bounds to a dictionary
    parsedList, configDict = getParsedList(configFile, configName)
    RQTextFile = './RQFiles.txt'
    with(open(RQTextFile, 'w') as f):
        f.write('FILE NUMBER  RUN NUMBER  FREQUENCY RANGE (MHZ)  DATE  TEMPERATURE  ANTENNA POSITION (W/V/S,THETA,PHI)\n')
        for aVal in parsedList:
            aDate = aVal[1].strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write((str(aVal[0][0])+ '  ' + str(aVal[0][1]) + '  ' + str(aDate) + '  ' + str(configDict['Freq'][0]) + '-' + str(configDict['Freq'][1]) + '  ' + str(aVal[2]) + '  ' + str(aVal[3])) + '\n')
            #print(configDict['Freq'])
    
    if writeH5:
        parsedDict = {}
        # Create a dictionary of lists of tuples (see the description for collectData). Doing this
        # makes it possible to parallelize the read operation
        for val in parsedList:
            keyVal = str(int(val[0][0]))
            if keyVal not in parsedDict:
                parsedDict[keyVal] = []
            
            # This will make it a little annoying to add more parameters to cut on
            parsedDict[keyVal].append((int(val[0][1]), (val[1], val[2], val[3], configDict['Freq'], configDict['Choice']))) 

    
        # List of keys (file numbers) in the parameter dictionary
        parsedKeys = [aKey for aKey in parsedDict]
        dataDir = '/group/tysongrp/JulyRun_7-7-22/Data/'
        #for aKey in parsedDict:
        #    aFile = 'data_' + str(aKey) + '.h5'
        #    dataBin = h5.File(dataDir + aFile, 'r')
        #    for anAverage in parsedDict[aKey]:
        #        print(dataBin['measdata_' + str(anAverage[0])])

        # Create the read/write queues
        writeQueue = mp.Queue()
        readQueue = mp.Queue()
        # Add data to the read queue
        [readQueue.put((aKey, parsedDict[aKey])) for aKey in parsedKeys] 
        
        # Start the write process
        proc = mp.Process(target=writeFiles, args=(writeQueue, ))
        jobs = []
        proc.start()

        # Create the read processes
        for i in range(num_processes):
            p = mp.Process(target=collectData, args=(readQueue, writeQueue, dataDir))
            jobs.append(p)
            p.start()
        for i in range(num_processes):
            # Send the sentinal to tell Simulation to end
            readQueue.put(sentinel)
        for p in jobs:
            p.join()
        writeQueue.put(None)
        proc.join()



