import avgFft_module
import settings as s
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import time
import datetime
import threading


def procFftAndSave(fftSum, 
                    numFft,
                    date_time_done,
                    switchPos,
                    s):
    ti = time.time()

    # Get FFT^2 sum(s) from GPU buffer into RAM. 
    # Divide by number of FFTs to compute average
    # Return dictionary of spectra  
    specDict =  avgFft_module.avgAndConvertFFT(fftSum,
                                                numFft,
                                                s)

    #auto extract acquisition number
    if not os.path.exists(s.SAVE_DIRECTORY+'database.txt'):
        acqNum = 0
        print(f'set to 0')
    else:
        acqNum = int(open(s.SAVE_DIRECTORY + 'database.txt', 'r'
                        ).readlines()[-1].split(',')[0].strip()) + 1 


    print(f'acq num {acqNum}')
    runInfoDict     = {'ACQ NUM' : acqNum, #number of spectra since start of run. Must be first
                        'DATETIME'          : date_time_done,
                        'ANT POS IDX' : s.ANT_POS_IDX,
                        'LEN FFT LOG2' : int(np.log2(s.CH0_RECORD_LEN)),
                        'SAMPLE RATE MHZ' : s.SAMPLE_RATE/1e6,
                        }
    if s.SAVE_AMP_CHAIN == 1: #option to make text file smaller without amp chain info
        runInfoDict.update(s.SETUP_DICT)

    if s.SAVE_W_SPEC == 1:
        np.save(s.PATH_TO_SAVE_W_DICT, specDict)

    avgFft_module.writeH5(specDict,
        runInfoDict,
        acqNum,
        s.SAVE_DIRECTORY)
    
    np.save(s.SAVE_DIRECTORY + 'plottingSpec/chA_W_switch' + str(switchPos), specDict['chASpec_W'])
    np.save(s.SAVE_DIRECTORY + 'plottingSpec/chB_W_switch' + str(switchPos), specDict['chBSpec_W'])

    # compute average spec for plotting
    if acqNum == 0:
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch' + str(switchPos) + '.npy', specDict['chASpec_W'])
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch' + str(switchPos) + '.npy', specDict['chASpec_W'])
    else:
        pastSpecA = np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch' + str(switchPos) + '.npy')
        pastSpecB = np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch' + str(switchPos) + '.npy')
        avgSpecA  = (((acqNum - 1) * pastSpecA) + specDict['chASpec_W'])/acqNum
        avgSpecB  = (((acqNum - 1) * pastSpecB) + specDict['chBSpec_W'])/acqNum
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch' + str(switchPos) + '.npy', avgSpecA)
        np.save(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch' + str(switchPos) + '.npy', avgSpecB)

    print(f"Processed in {time.time() -ti}")

if __name__ == "__main__":
    avgSpec = avgFft_module.avgFft(s)

    if s.SWITCH:
        ######### switch arduino here ########
        numLoops = s.NOF_ACQUISITIONS_TO_TAKE//2
    else:
        numLoops = s.NOF_ACQUISITIONS_TO_TAKE

    for acqNum in range(numLoops):
        switchPos = 0
        date_time = str(datetime.datetime.now())
        start_time = time.time()
        if acqNum%10 ==0:
            print(f'STARTING ACQUISITION NUMBER: {acqNum} AT {date_time}')
        ti = time.time()
        print()

        avgSpec.collectAndSumFft()
        date_time_done = str(datetime.datetime.now())
        fftSumCopy = avgSpec.fftSum.clone()
        numFftCopy = avgSpec.numFft[:]

        t = threading.Thread(target=procFftAndSave, 
                               args=(fftSumCopy, 
                                    numFftCopy,
                                    date_time_done,
                                    switchPos, 
                                    s)
                            )
        t.start()

        if s.SWITCH:
            switchPos = 1
            ######### switch arduino here ########
            avgSpec.collectAndSumFft()
            date_time_done = str(datetime.datetime.now())
            fftSumCopy = avgSpec.fftSum.clone()
            numFftCopy = avgSpec.numFft[:]

            t = threading.Thread(target=procFftAndSave, 
                                args=(fftSumCopy, 
                                        numFftCopy,
                                        date_time_done,
                                        switchPos,
                                        s)
                                )
            t.start()

        #avgFft_module.avgAndConvertFFT(fftSumCopy, numFftCopy, s)

        #ti = time.time()
        #avgSpecCopy = avgSpec.copy()#copy instance to avoid next call overwriting
        #print(f'Time to copy {time.time()-ti}')
        '''
        runInfoDict     = {'Acquisition Number' : acqNum, #number of spectra since start of run
                            'Datetime'          : date_time}
        #include units in dict key
        specDict        = {'antSpec_W'        : avgSpec.avgSpec_W_Ch0}

        numSpecPerFile  = 16
        dataDir         = '/drBiggerBoy/h5Testing_gitignore/'
        '''

        #avgFft_module.writeH5(specDict, runInfoDict, acqNum, numSpecPerFile, dataDir)
        print(f'ACQUISITION TOOK:{(time.time()-ti)}')
    avgSpec.exit()
    #np.save('./powSpec0', avgSpec.avgPowSpecCh0)
    #np.save('./powSpec1', avgSpec.avgPowSpecCh1)
    #print('SAVED')


    #avgPowSpec1 = np.asarray(avgPowSpec[0::2]).mean(axis=0)
    #avgPowSpec2 = np.asarray(avgPowSpec[1::2]).mean(axis=0)
    #np.save('avgPowSpec1_3000avg_6switchPerSide_2_17_23', avgPowSpec1)
    #np.save('avgPowSpec2_3000avg_6switchPerSide_2_17_23', avgPowSpec2)
    if 1:
        avgSpec_W_plotting = avgFft_module.avgAndConvertFFT(
            avgSpec.fftSum,
            avgSpec.numFft,
            s
        )

        plt.figure()
        plt.title(f"{s.NOF_BUFFERS_TO_RECEIVE} FFTs Averaged")
        plt.plot(np.linspace(0,s.SAMPLE_RATE/2/1e6,s.CH0_RECORD_LEN//2),10*np.log10(avgSpec_W_plotting['chASpec_W'][1:]*1000), label = 'CH A')
        plt.plot(np.linspace(0,s.SAMPLE_RATE/2/1e6,s.CH0_RECORD_LEN//2),10*np.log10(avgSpec_W_plotting['chBSpec_W'][1:]*1000), alpha = 0.8, label = 'CH B')
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