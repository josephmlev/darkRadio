from scipy.signal import butter, filtfilt, find_peaks, freqz, sosfilt
import glob
import os
from fnmatch import fnmatch
import h5py
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sys 
import multiprocessing
from multiprocessing import Pool
import time


################
# Functions
################

def fft2dBm(fftSpec):
    '''
    Converts fft units to dBm
    '''
    return 10*np.log10(2 * 1000 * fftSpec/ 2**48 / 50) 


def mkFileList(dataDir):
    '''
    Get list of files in dataDir, pack data*.h5 named files into fileList
    '''
    dataDirContents = os.listdir(dataDir)
    fileList = [file for file in dataDirContents if fnmatch(file, 'data*.h5')]
    fileList.sort()
    return fileList

################
# Classes
################

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
        verbose = False):

        self.fileList = fileList
        self.dataDir = dataDir
        self.numProc = numProc
        self.verbose = verbose

        self.antData = np.zeros(2**23)
        self.termData = np.zeros(2**23)
        
        #can pull these out of data files later
        self.startFreq = 0
        self.stopFreq = 300
        numBins = 8388607
        self.freqs = np.linspace(self.startFreq, self.stopFreq, numBins)
   
        self.computeAvgAll()

    
    
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
        antData = np.zeros(2**23)
        termData = np.zeros(2**23)
        numDataSets = 0
        numAvgInDataFile = 0

        #itterate through file list and open h5 files in 'r(ead)' mode
        if self.verbose == True:
            print('ON FILE: ' + str(aFile))
        dataBin = h5py.File(self.dataDir + aFile, 'r') #makes pointer to h5
        numAvgInMesDataSet = int(dataBin.attrs['averages']) #total number of average for dataset
        allKeys = [aKey for aKey in dataBin.keys()] #key is measdata. 16 per dataset

        for aKey in allKeys[:]:
            dataset = pd.read_hdf(self.dataDir + aFile, key = aKey) #dataframe of ROACH data (check)
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
        if 1: #__name__ == '__main__': #does not import with name == main. Removing it fixes issue. Come back to this
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

        if self.verbose:
            print('time to open and avg mes data =', tMesData - tStart)
        print('Done. Total time =', tDone - tStart)