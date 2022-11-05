from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import time
#import matplotlib.pyplot as plt 
import h5py as h5
import os
import pandas as pd

def getFiles(dirLoc):
    dataFiles = np.asarray(sorted([f for f in os.listdir(dirLoc) if f.endswith('.h5')], key = lambda x: int(x[x.index('_') + 1:x.index('.')]), reverse = False))
    return dataFiles

class avgSpec:
    '''
    Inputs:
        fileList(list):
        List of file names of form `data_<n>.h5`

        dataDir(string): 
        Realitive path to data directory

        numProc(int):
        Number of processes to open data files.

    Methods:
        avgMesDataAll: worker. averages together all measData in a single data file
        compAvgAll: calls avgMesData in parallel
    '''
    
    def __init__(self,
        fileList,
        dataDir,
        numProc = 28,
        verbose = False,
        antFile = ''):

        self.fileList = fileList
        self.dataDir = dataDir
        self.numProc = numProc
        self.verbose = verbose
        self.antPos = self.getAllAntPos(antFile)

        self.antData = np.zeros(2**23 - 1)
        self.termData = np.zeros(2**23 - 1)
        
        #can pull these out of data files later
        self.startFreq = 0
        self.stopFreq = 300
        numBins = 2**23 - 1
        self.freqs = np.linspace(self.startFreq, self.stopFreq, numBins)

    def getAllAntPos(self, antFile):
        antArr = []
        AntLoc = namedtuple('AntPos', 'run west vertical south phi theta')
        if os.path.exists(antFile):
            with open(antFile, 'r') as f:
                f.readline()
                for line in f:
                    antArr.append(AntLoc(*tuple(line.split())))
        
            print(antArr)
            return np.asarray(antArr)  
        else:
            print('ANTENNA FILE NOT FOUND')
            return np.asarray([])        

    def getAntPos(self, runNum):
        print(self.antPos[runNum])
        return self.antPos[runNum]

    def avgMesDataAll(self, aFile):
        '''
        Worker function to open a data file and average ALL mesData. Called by mProcAvgAll

        Parameters:
            self.fileList (list of strings): list of file names to be averaged 
            self.dataDir (str): Location of files. Realitive path.

        Returns:
            antData (1D np arr): averaged off spectrum (linear FFT units)
            termData (1D np arr): averaged on spectrum (linear FFT units)
            numAvgInDataFile (int): number of averages in data file.
        '''
        #init some parameters
        antData = np.zeros(2**23 - 1)
        termData = np.zeros(2**23 - 1)
        numDataSets = 0
        numAvgInDataFile = 0

        #iterate through file list and open h5 files in 'r(ead)' mode
        if self.verbose == True:
            print('ON FILE: ' + str(aFile))
        dataBin = h5.File(self.dataDir + aFile, 'r') #makes pointer to h5
        numAvgInMesDataSet = int(dataBin.attrs['averages']) #total number of average for dataset
        allKeys = [aKey for aKey in dataBin.keys()] #key is measdata. 16 per dataset

        for aKey in allKeys[:]:
            dataset = pd.read_hdf(dataDir + aFile, key = aKey) #dataframe of ROACH data (check)
        # need to print and look at this
            termData +=  np.asarray(dataset[dataset.keys()[0]]) #term or bicon
            antData += np.asarray(dataset[dataset.keys()[1]]) #the other one
            numDataSets += 1
            numAvgInDataFile += numAvgInMesDataSet

        antData /= numDataSets
        termData /= numDataSets
        
        return antData, termData, numAvgInDataFile

    def avgDataList(self):
        '''
        Parameters:
            self.dataList (list of len=self.fileList): Each element of dataList contains a len=3 tuple
                which contains (antData (arr of len=numBins), 
                                termData (arr of len=numBins), 
                                numAvgInDataFile (int))
        
        Sets:
            self.antData (1D np arr): averaged off spectrum (linear FFT units)
            self.termData (1D np arr): averaged on spectrum (linear FFT units)
        '''
        antData = np.zeros(2**23)
        termData = np.zeros(2**23)
        numDataFiles = 0

        for dataTuple in self.dataList:
            antData += dataTuple[0]
            termData += dataTuple[1]
            numDataFiles += 1
        
        self.antData = antData / numDataFiles
        self.termData = termData / numDataFiles
        

    def getDates(self, aFile):
       #for aFile in self.fileList:
        dateList = []
        dataBin = h5.File(self.dataDir + aFile, 'r') #makes pointer to h5
        for aKey in sorted(dataBin.keys(), key = lambda x: int(x[x.index('_')+1:]), reverse = False):
            dataset = pd.read_hdf(self.dataDir + aFile, key = aKey) #dataframe of ROACH data (check)
            dateList.append(datetime.strptime(dataset.keys()[0][0], '%Y-%m-%d %H:%M:%S.%f'))
            #print(dateList[-1])
        #print(dateList)
        return dateList

    def getTemperature(self, aFile):
        dataBin = h5.File(self.dataDir + aFile, 'r') #makes pointer to h5
        #print(float(dataBin.attrs['temperature'][:-1]))
        return float(dataBin.attrs['temperature'][:-1])

    def getRunNumbers(self, aFile):
        #for aFile in self.fileList:
        runNums = []
        dataBin = h5.File(self.dataDir + aFile, 'r')
        for aKey in sorted(dataBin.keys(), key = lambda x: int(x[x.index('_')+1:]), reverse = False):
            runNums.append(aKey[aKey.index('_') + 1:])
            #print(int(runNums[-1]))
        #print(runNums)
        return runNums
    
    def getFileNumber(self, aFile):
        #print(int(aFile[aFile.index('_') + 1:-3]))
        return int(aFile[aFile.index('_') + 1:-3])

    def genDatabase(self, dbName):
        totalFiles = len(self.fileList)
        totalAntPos = len(self.antPos)
        with open(dbName, 'w') as f:
            f.write('(File#, Run#)\tDate\tTemperature\tAntPos(West, Vertical, South, Phi, Theta)\n')
            firstDate = self.getDates(self.fileList[0])[0]
            antCount = 0
            antPosLookupArr = []
            for aFile in (self.fileList):
                print('ON FILE ' + str(aFile))
                runNums = self.getRunNumbers(aFile)      
                fileNum = [self.getFileNumber(aFile)]*len(runNums)
                temps = [self.getTemperature(aFile)]*len(runNums)
                dates = np.asarray(self.getDates(aFile))
                #print('ALL DATES: ' + str(dates))
                #print('FIRST DATE: ' + str(firstDate))
                #print('LAST DATE: ' + str(dates[-1]))
                print('TEST: '  + str((dates[0] - firstDate).total_seconds()))
                if (dates[0] - firstDate).total_seconds() > 60*5:
                    antCount += 1
                antPos = [self.getAntPos(antCount)]*len(runNums)
                firstDate = dates[-1]
                for val in zip(fileNum, runNums, dates, temps, antPos):
                    antString = '(' + val[4][1] + ', ' + val[4][2] + ', ' + \
                                 val[4][3] + ', ' + val[4][4] + ', ' + val[4][5] + ')'
                    f.write('(' + str(val[0]) + ',' + str(val[1]) + '): ' + \
                           str(val[2]) + ', ' + str(val[3]) + ', ' + antString + '\n')   
    def computeAvgAll(self):
        '''
        Opens each file in fileList and computes an average of ALL measData in each data file.
        Then averages each of these averages together. 

        Parameters:
            self.fileList (list of strings): list of file names to be averaged

        Returns:
            None

        Sets:
            self.dataList (list of len=self.fileList): Each element of dataList contains a len=3 tuple
                which contains (antData (arr of len=numBins), 
                                termData (arr of len=numBins), 
                                numAvgInDataFile (int))
             
        Calls:
            avgMesDataAll(): worker function. Averages together ALL mesData in a data file
        '''
        tStart = time.time()

        print('Starting to open', len(self.fileList),
                'files and avgerage mesData with', self.numProc, 'processes')
        if __name__ == '__main__':
            #Multiprocessing. Each process (pool) computes an 
            #average spectra on ALL of a single data file's measData.
            pool = Pool(processes = self.numProc)
            self.dataList = pool.map(self.avgMesDataAll, self.fileList)

            #Calculate total number of FFTs in returned spectra. Each data 
            #file returns number of FFTs in dataList[data file index][2]  
            self.totalNumAvg = 0
            for runningAverages in self.dataList:
                self.totalNumAvg += runningAverages[2]
        tMesData = time.time()

        #print('Done opening. Now averaging dataFile arrays together') 
        self.avgDataList()
        tDone = time.time()

        #print('time to open and avg mes data =', tMesData - tStart)
        print('Done. Total time =', tDone - tStart)

dirPath = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/'
dirName = 'DataRun_7-7-22'
antFile = '/home/dark-radio/HASHPIPE/ROACH/PYTHON/antPos.txt'
allFiles = getFiles(dirPath + dirName + '/')
specObj = avgSpec(allFiles, dirPath + dirName + '/', antFile = antFile)
specObj.genDatabase('SearchableDatabase.txt')
#specObj.getRunNumber()


