import h5py
import pandas as pd
import numpy as np
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, '../../drlib')
import drlib as dr

try:
    f.close()
except:
    pass
ti = datetime.now()

rawDataDir      = '/drBiggerBoy/drData/Data/'
fileList        = os.listdir(rawDataDir)
fileList.sort(key=lambda f: int(re.sub('\D', '', f)))
newDataDir      = '/drBiggerBoy/drData/run1Data/'
fileName        = 'preProcDataSet.hdf5'
f               = h5py.File(newDataDir + fileName, 'w')

diffSpecH5        = f.create_dataset('diffSpec_W', (8388608, 11667), chunks = (2**16,2**0), dtype = 'f')
antSpecH5         = f.create_dataset('antSpec_W', (8388608, 11667), chunks = (2**16, 2**0), dtype='f')
vetoSpecH5        = f.create_dataset('vetoSpec_W', (10000, 11667), chunks = True, dtype='f')

##############################################################
#pack "searchable database" df
#indexed by int which counts spectra
#Columns: 
    #File, measData, Date, Temperture, West, Vertical, South, Phi, Theta, VetoSpec(array, len=1e4)
##############################################################

database = pd.read_csv('./databaseForPandas.txt', skiprows=0)#False
database['Date'] = pd.to_datetime(database['Date'], infer_datetime_format=True)
database.set_index('Date', inplace=True)
#print(database.head())

#check that dates are consecutive
dates = database.index
date_ints = set([d.toordinal() for d in dates])
if max(date_ints) - min(date_ints) == len(date_ints) - 1:
    print("Dates are consecutive in database")
else:
    raise Exception('Dates are not consecutive in database')

##############################################################
#Main loop
##############################################################
numFiles        = 0
numMeasData     = 0
numRigolSpec    = 0

compType = False
chunks = None #must add chunks arg to create dataset calls

for fileidx, file in enumerate(fileList):
    if fileidx % 10 == 0:
        print(file)
    dataset         = h5py.File(rawDataDir +  file, 'r')
    numFiles        += 1
    measDataKeys    = list(dataset.keys())
    measDataKeys.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    for measData in measDataKeys:
        measDataInt = int(re.search(r'\d{0,3}$', measData).group())
        measDataSubKeys = dataset[measData].keys()

        #dumb check. Is the date in raw h5 measData the same as what the database
        #thinks it is? Since we previously check that dates are consectutive this 
        #also checks that in raw h5
        datasetDf   = pd.read_hdf(rawDataDir + file, key = measData)
        dateTimeStr = datasetDf.columns[0][0]
        dateTime    = datetime.strptime(dateTimeStr, '%Y-%m-%d %H:%M:%S.%f')
        if not (dateTime==database[database['measData'] == measDataInt].index):
            raise Exception('Dates got jumbled at measData', measDataInt)

        antSpec     = np.float32(dr.fft2Watts(datasetDf.iloc[:,1][:]))
        termSpec    = np.float32(dr.fft2Watts(datasetDf.iloc[:,0][:]))
        diffSpec    = np.float32((antSpec-termSpec))

        diffSpecH5[:, measDataInt]    = diffSpec
        antSpecH5[:, measDataInt]     = antSpec


        #write other things#####
        if 0:
            diffSpecFFT = np.float32((np.asarray(datasetDf.iloc[:,2][1:])))
            dsetFFT     = grp.create_dataset('diffSpec_fft', data = diffSpecFFT, compression=compType)
            colA        = np.float32((np.asarray(datasetDf.iloc[:,0][1:])))
            dsetA       = grp.create_dataset('termSpec_fft', data = colA, compression=compType)
            colB        = np.float32((np.asarray(datasetDf.iloc[:,1][1:])))
            dsetB       = grp.create_dataset('antSpec_fft', data = colB, compression=compType)
        #######################
        

        #take Rigol max and write max Watts
        numMeasData     += 1
        rigolIdx        = 0
        rigolTempArr    = np.zeros((10000,4))
        for key in measDataSubKeys:
            if key[0]=='R':
                rigolTempArr[:, rigolIdx] = dr.dBm2Watts((np.float64(dataset[measData][key])))
                rigolIdx += 1
        rigolMaxArr = np.zeros(10000)
        for i, row in enumerate(rigolTempArr):
            rigolMaxArr[i] = ((row).max())
        vetoSpecH5[:,measDataInt] = np.float32(rigolMaxArr)
        

print('number of files = ', numFiles)
print('number of meas data = ', numMeasData)

f.close()
#put database DF in hdf
database.to_hdf(newDataDir+fileName, 'database_DF')

print('Time:', datetime.now() - ti)


