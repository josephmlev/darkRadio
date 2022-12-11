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

rawDataDir = '/drBiggerBoy/drData/Data/'
fileList = os.listdir(rawDataDir)
fileList.sort(key=lambda f: int(re.sub('\D', '', f)))
newDataDir = '/drBiggerBoy/drData/run1Data/'
fileName = 'preProcDataSet.hdf5'
f = h5py.File(newDataDir + fileName, 'r')


##############################################################
#pack "searchable database" df
#indexed by int which counts spectra
#Columns: 
    #File, measData, Date, Temperture, West, Vertical, South, Phi, Theta, VetoSpec(array, len=1e4)
##############################################################

database = pd.read_csv('./databaseForPandas.txt', skiprows=0)#False
database['Date'] = pd.to_datetime(database['Date'], infer_datetime_format=True)
database.set_index('Date', inplace=True)
print(database.head())

#check that dates are consecutive
dates = database.index
date_ints = set([d.toordinal() for d in dates])
if max(date_ints) - min(date_ints) == len(date_ints) - 1:
    print("Dates are consecutive")
else:
    raise Exception('Dates are not consecutive')

#put database DF in hdf
database.to_hdf(newDataDir+fileName, 'database_DF')

##############################################################
#Main loop
##############################################################
numFiles        = 0
numMeasData     = 0
numRigolSpec    = 0

compType = False
chunks = None #must add chunks arg to create dataset calls

for file in fileList:
    print(file)
    dataset         = h5py.File(rawDataDir +  file, 'r')
    numFiles        += 1
    measDataKeys    = list(dataset.keys())
    measDataKeys.sort(key=lambda f: int(re.sub('\D', '', f)))

    for measData in measDataKeys:
        measDataSubKeys = dataset[measData].keys()
        #print(measDataSubKeys)
        grp         = f.create_group(str(numMeasData))

        datasetDf   = pd.read_hdf(rawDataDir + file, key = measData)
        dateTimeStr = datasetDf.columns[0][0]
        dateTime    = datetime.strptime(dateTimeStr, '%Y-%m-%d %H:%M:%S.%f')
        antSpec     = np.asarray(datasetDf.iloc[:,1][1:])
        termSpec    = np.asarray(datasetDf.iloc[:,0][1:])
        diffSpec    = np.float32(dr.fft2Watts(antSpec-termSpec))
        dset        = grp.create_dataset('diffSpec_W', data = diffSpec, compression=compType, chunks=chunks)

        #write other things#####
        if 0:
            diffSpecFFT = np.float32((np.asarray(datasetDf.iloc[:,2][1:])))
            dsetFFT     = grp.create_dataset('diffSpec_fft', data = diffSpecFFT, compression=compType)
            colA        = np.float32((np.asarray(datasetDf.iloc[:,0][1:])))
            dsetA       = grp.create_dataset('termSpec_fft', data = colA, compression=compType)
            colB        = np.float32((np.asarray(datasetDf.iloc[:,1][1:])))
            dsetB       = grp.create_dataset('antSpec_fft', data = colB, compression=compType)
        #######################
        
        grp.attrs.create('Datetime', dateTimeStr)
        for attr in database.keys():
            grp.attrs.create(attr, database.loc[dateTime][attr])
        
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
            
        rigolDest    = grp.create_dataset('rigolSpec_W', data = np.float32(rigolMaxArr), compression=compType)


print('number of files = ', numFiles)
print('number of meas data = ', numMeasData)

f.close()


