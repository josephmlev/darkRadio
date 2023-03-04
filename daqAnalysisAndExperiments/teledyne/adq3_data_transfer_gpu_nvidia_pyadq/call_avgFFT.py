import avgFft_module
import settings as s
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import time
import datetime

if __name__ == "__main__":
    startFlag = 0
    while startFlag == 0:
        avgSpec = avgFft_module.avgFft(s)
        ti = time.time()
        avgSpec.avgFft()
        if time.time()-ti < 1.05*s.CH0_RECORD_LEN/s.SAMPLE_RATE:
            startFlag = 1
            print('Starting Run')
        else:
            print('Starting again')
            print('TOOK:', time.time()-ti)

    for acqNum in range(s.NOF_ACQUISITIONS_TO_TAKE):
        date_time = str(datetime.datetime.now())
        start_time = time.time()
        if acqNum%10 ==0:
            print(f'STARTING ACQUISITION NUMBER: {acqNum} AT {date_time}')
        avgSpec.avgFft()

        runInfoDict     = {'Acquisition Number' : acqNum, #number of spectra since start of run
                            'Datetime'          : date_time}
        #include units in dict key
        specDict        = {'antSpec_W'        : avgSpec.avgSpec_W_Ch0}

        numSpecPerFile  = 16
        dataDir         = '/drBiggerBoy/h5Testing_gitignore/'

        #avgFft_module.writeH5(specDict, runInfoDict, acqNum, numSpecPerFile, dataDir)
        print(f'ACQUISITION TOOK:{(time.time()-start_time)}')
    avgSpec.exit()
    #np.save('./powSpec0', avgSpec.avgPowSpecCh0)
    #np.save('./powSpec1', avgSpec.avgPowSpecCh1)
    #print('SAVED')


    #avgPowSpec1 = np.asarray(avgPowSpec[0::2]).mean(axis=0)
    #avgPowSpec2 = np.asarray(avgPowSpec[1::2]).mean(axis=0)
    #np.save('avgPowSpec1_3000avg_6switchPerSide_2_17_23', avgPowSpec1)
    #np.save('avgPowSpec2_3000avg_6switchPerSide_2_17_23', avgPowSpec2)
    if 1:
        plt.figure()
        plt.title(f"{s.NOF_BUFFERS_TO_RECEIVE} FFTs Averaged")
        plt.plot(np.linspace(0,1250/s.CH0_SAMPLE_SKIP_FACTOR,s.CH0_RECORD_LEN//2),10*np.log10(avgSpec.avgSpec_W_Ch0[1:]*1000), label = 'CH A')
        #plt.plot(np.linspace(0,1250/s.CH0_SAMPLE_SKIP_FACTOR,s.CH0_RECORD_LEN//2),10*np.log10(avgSpec.avgPowSpecCh1[1:]), alpha = .9, label='CH B')
        plt.xlabel('Freq(MHz)')
        plt.ylabel('Power (dBm)')
        plt.legend(loc='upper right')
        plt.plot()
        plt.show()
    if 0:
        plt.figure()
        plt.plot(np.linspace(0,1250/s.CH0_SAMPLE_SKIP_FACTOR,s.CH0_RECORD_LEN//2),((avgPowSpec1-avgPowSpec2)[1:]))
        plt.xlabel('Freq(MHz)')
        plt.ylabel('Power (mW)')
        plt.plot()
        plt.show()