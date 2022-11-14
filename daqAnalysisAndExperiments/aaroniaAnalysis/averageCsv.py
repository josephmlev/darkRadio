import numpy as np
import pandas as pd
import h5py
import time
import os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../../drlib')
import drlib as dr

'''
This script takes a csv file name (of the form output by aaronia), 
and a list of tuples of time ranges and computes a linear average.
More testing is needed as I don't think it works well with very large files.
h5 files still may be the way to go.
'''


#parameters
################################################################
csv_filename = '8ghz0_gitignore.csv'
cpu_count = cpu_count() #default to total number of cpus, can change to save RAM

#Larger chunkSize uses more RAM. 
#Example: 1M pt fft storing 32 bit floats uses 4MB per spectra. 
#chunksize = 1000 would require 4GB of ram (allow atleast a factor of 2 overhead)
chunkSize = 10

#time values to include in average.
#must be a list of tuples
#example: rangeList = [(1,2), (3,4)]
#would include all time samples between 1 and 2 AND between 3 and 4 seconds
#an empty list averages all data in csv
#NOTE: This may be buggy, don't give overlapping ranges 
rangeList = [(10,10.5)]

#functions    
################################################################
def getFreqs(csv_filename):
    '''
    Reads csv and extracts column names (throwing out "time" name of first column)
    returns array of frequencies
    '''
    return np.asarray(((pd.read_csv(csv_filename, nrows=1)).columns)[1:], dtype=float)

def inRange(timeArr, rangeList):
    '''
    checks which values of a timeArr are in ranges of rangeList
    inputs:
    timeArr (np 1d array):
        aray of times to check
    rangeList (list of tuples): Ranges of times
        [(start time 1, stop time 1), ..., (start time N, stop time N)]
    Returns:
        boolArr (np array of bools): mask array equal in length to timeArr
        True if the index is in a range, False otherwise
    '''
    boolArr = np.zeros((len(rangeList), len(timeArr)))
    for i, tup in enumerate(rangeList):
        boolArr[i, :] = (np.logical_and(timeArr >= tup[0], timeArr <= tup[1]))
    return np.any(boolArr, axis = 0)

def chunkCsv(chunkIdx):
    '''
    reads a chunk of the CSV containing N spectra,
    converts from dBm to watts and averages.
    Inputs:
       chunk (pandas dataFrame): chunk of spectra from csv
       shape is (num spectra, num frequencies)
    Returns:
       avgSpec (np array): averaged spectrum of N sub-spectra.
    NOTE: rangeList is passed in as global variable. This is a little janky
    but hard to use pool to call multple arguements

    There should be a feature to count how many spectra have been added, I can add this if needed
    '''
    #get times and compute boolian mask array for chunk given rangeList
    chunkIdx = chunkIdx.index.tolist()
    chunk = pd.read_csv(csv_filename, index_col=0, skiprows=chunkIdx[0], nrows=chunkSize)
    times = np.asarray(chunk.index)

    #avoid calling inRange if no rangeList given (for speed)
    if rangeList:
        boolArr = inRange(times, rangeList)
    else:
        boolArr = np.full(len(times), True)

    #if no times found, return false
    if boolArr.sum() == 0:
        return False

    #else convert to watts, average and return avgSubSpec
    else:
        chunk_w = dr.dBm2Watts(np.asarray(chunk))
        #use boolArr to index chunk to only include values in a time range. Then average.
        avgSubSpec = chunk_w[boolArr,:].mean(axis = 0)
        return avgSubSpec


#MAIN
#multi process the averaging. 
################################################################
startTime = time.time()
with Pool(processes=cpu_count) as p:
    #subSpecs_list_temp = p.map(chunkCsv, pd.read_csv(csv_filename, index_col=0, chunksize=chunkSize))
    subSpecs_list_temp = p.map(chunkCsv, pd.read_csv(csv_filename, usecols=[0] , chunksize=chunkSize))
print('time to average sub spectra from raw data = ', time.time() - startTime, 'ms')
subSpecs_list = []
for spec in subSpecs_list_temp:
    if type(spec) == bool:
        continue
    else:
        subSpecs_list.append(spec) 


subSpec_arr = np.asarray(subSpecs_list)
avgSpec = subSpec_arr.mean(axis = 0)
print('time to average sub spectra together = ', time.time() - startTime, 'ms')

#extract frequencies from csv
################################################################
#this is slow for 1M pt fft and large files. Comment it out and use np.linspace if that's a problem
freqs = getFreqs(csv_filename)

#plotting
################################################################
plt.close('all')
if 0:
    plt.figure()
    plt.plot(freqs/1e9, avgSpec)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power (watts)')
    plt.show()

if 1:
    plt.figure()
    plt.plot(freqs/1e9, dr.watts2dBm(avgSpec))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power (dBm)')
    plt.show()
################################################################