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

#########################################################################################
#Analysis
#########################################################################################

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
    Will soon be replaced by getParsedList
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
        parsedList(list):
        List of tuples of file numbers and info
        each entry is tuple of form
        (
        tuple(float(fileNumber), float(spectrumNumber),
        datetime object,
        float(temperature),
        (float(West), float(Vertical), float(South), float(Phi), float(Theta))
        ) 

        dataDir(string): 
        Realitive path to data directory

        numProc(int):
        Number of processes to open data files.

    Methods:
        compAvg: Main. calls avgMesData in parallel to average all files. Then calls
		        avgDataList to average these subspectra together into a master spectrum

		Internal functions to compute average spectra:
            avgMesData: worker. averages selected measData spectra from a single data file
            avgDataList: Takes spectra from avgMesData and computes a master average
            setSpecDictsAndFileList: converts parsedList to dicts and lists
            for use by other methods
		
    '''
    
    def __init__(self,
        parsedList,
        dataDir,
        numProc = 28,
        verbose = False):

        self.parsedList = parsedList
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

        
        self.computeAvg(parsedList = self.parsedList)
     
    def setSpecDictsAndFileList(self):
        '''
        Converts parsedList to a fileList and specDicts (one for names, one for numbers)

        Sets: 
        fileList: list of str(fileName))
        specNumDict: dict. {int(fileNum) : list(int(specNum))}
        specNameDict: dict. {str(fileName) : list(str(specName)}

        Input:
        parsedList: list of tuples (float(fileNum), float(specNum))
        '''
        #tests
        if type(self.parsedList) != list:
            raise TypeError('parsedList is not a list!')
            
        #init stuff
        specList = []
        fileList = []
        specNameDict = {}
        specNumDict = {}

        parsedListOnlyIdx = [spec[0] for spec in self.parsedList]
        #itterate over parsedList, set unique file name/num as keys in dict. Fill
        #each key with lists of names/nums of mes data  
        for fileNumFloat, specNumFloat in parsedListOnlyIdx:
            #generate names/numbers for files/specs
            fileNum = int(fileNumFloat)
            specNum = int(specNumFloat)
            fileName = 'data_' + str(fileNum) + '.h5'
            specName = 'measdata_' + str(specNum)
            #stuff dicts
            specNumDict.setdefault((fileNum), []).append(specNum)
            specNameDict.setdefault((fileName), []).append(specName)
        #set dicts and fileNameList 
        self.fileNameList = list(specNameDict.keys())
        self.specNameDict = specNameDict
        self.specNumDict = specNumDict

    def avgMeasData(self, aFile):
        '''
        Worker function to open a data file and average SELECT measData. Called by computeAvg

        Parameters:
            self.dataDir (str): Location of files. Realitive path.
            aFile (tuple): specifies a spectrum. (fileName(str), measData(int)) 

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
            print('ON FILE: ' + str(aFile[0]))
        dataBin = h5py.File(self.dataDir + aFile[0], 'r') #makes pointer to h5
        numAvgInMeasDataSet = int(dataBin.attrs['averages']) #total number of average for dataset



        for aKey in aFile[1][:]:
            dataset = pd.read_hdf(self.dataDir + aFile[0], key = aKey) #dataframe of ROACH data (check)
        # need to print and look at this
            termData +=  np.asarray(dataset[dataset.keys()[0]]) #term or bicon
            antData += np.asarray(dataset[dataset.keys()[1]]) #the other one
            numDataSets += 1
            numAvgInDataFile += numAvgInMeasDataSet

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
        NOTE: this takes about 2% of the time it takes to open files and avg 
            their spectra even though it is not parallized
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



    def computeAvg(self, parsedList):
        '''
        Opens each file in fileList and computes an average of SELECTED measData in each data file.
        Then averages each of these averages together. 

        Parameters:
            self.fileList - list of strings of file names to be averaged
            self.specNameList - list of tuples of (str(fileName), str(specNam))
            self.specNumList - list of tuples of (int(fileNum), int(specNum))

        Returns:
            None

        Sets:
            self.dataList (list of len=self.fileList): Each element of dataList contains a len=3 tuple
                which contains (antData (arr of len=numBins), 
                                termData (arr of len=numBins), 
                                numAvgInDataFile (int))
             
        Calls:
            avgMeasDataAll(): worker function. Averages together ALL measData in a data file
        '''
        self.setSpecDictsAndFileList()
        print('Starting to open', len(self.fileNameList),
                'files and avgerage measData with', self.numProc, 'processes')
        tStart = time.time()
        if 1: #__name__ == '__main__': #does not import with name == main. Removing it fixes issue. Come back to this
            #Multiprocessing. Each process (pool) computes an 
            #average spectra on select spectra of a single data file's measData.
            pool = Pool(processes = self.numProc)
            self.dataList = pool.map(self.avgMeasData, list(self.specNameDict.items()))

            #Calculate total number of FFTs in returned spectra. Each data 
            #file returns number of FFTs in dataList[data file index][2]  
            self.totalNumAvg = 0
            for runningAverages in self.dataList:
                self.totalNumAvg += runningAverages[2]
        tMeasData = time.time()

        #print('Done opening. Now averaging dataFile arrays together') 
        self.avgDataList()
        tDone = time.time()

        print('time to open and avg mes data =', tMeasData - tStart)
        print('Done. Total time =', tDone - tStart)
