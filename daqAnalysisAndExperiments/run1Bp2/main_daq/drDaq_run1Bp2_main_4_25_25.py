#import avgFft_module
import settings as s
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import time
import datetime
import threading
import daqHelpers as daqh




if s.SWITCH:
    print('Setting up switch/temp Arduino')
    arduino = daqh.arduino()

if s.STIRRER:
    print('Setting up mode stirrer Arduino')
    stirArduino = daqh.stirArduino()
    time.sleep(3) #Arduino needs to sleep after contact or it gets upset

temperature = 0
switchPos = 0
switchSleep = 0

##### testing notes: 4/25/25. teledyne is broken. Disable all teledyne stuff to write stepper code. 11 # comments, 2 ''' comment
#added print temp on line 60


if __name__ == "__main__":
    #avgSpec = avgFft_module.avgFft(s)

    if s.INIT_SLEEP_TIME != 0:
        print(f'Sleeping for {s.INIT_SLEEP_TIME} seconds')
        time.sleep(s.INIT_SLEEP_TIME)

    if s.SWITCH:
        numLoops = s.NOF_ACQUISITIONS_TO_TAKE - (s.NOF_ACQUISITIONS_TO_TAKE//s.SWITCH_DUTYCYCLE) + 1
    else:
        numLoops = s.NOF_ACQUISITIONS_TO_TAKE

    # acqLoopNum is index for this loop
    # while acqNum remembers how many times 
    # drDaq has been called
    date_time_start = (datetime.datetime.now())
    for acqLoopNum in range(numLoops):
        if switchPos == 1:
            switchPos = 0
            switchSleep = 1
        if s.SWITCH:
            arduino.switch(switchPos)
            if switchSleep == 1:
                time.sleep(s.SWITCH_SLEEP_TIME)
                print('sleeping')
                switchSleep = 0
        if s.TEMPERATURE:
            temperature = arduino.getTemp()
            print(f'Temp: {temperature}')
        if s.STIRRER:
            pass
            #stirArduino.writeCmd('step 500')
        date_time_loop_start = (datetime.datetime.now())
        start_time = time.time()
        print(f'STARTING ACQUISITION LOOP NUMBER: {acqLoopNum} AT {str(date_time_loop_start)}')
        ti = time.time()

        #avgSpec.collectAndSumFft()
        date_time_done = str(datetime.datetime.now())
        #fftSumCopy = avgSpec.fftSum.clone()
        #numFftCopy = avgSpec.numFft[:]

        '''t = threading.Thread(target=daqh.procFftAndSave, 
                               args=(fftSumCopy, 
                                    numFftCopy,
                                    date_time_done,
                                    switchPos,
                                    temperature, 
                                    s)
                            )
        t.start()'''
        print(f'Acquisition of single buffer took   : {round((time.time()-ti),4)}')
        print(f'Theoretical time for single buffer  : {round((1/s.SAMPLE_RATE * s.CH0_RECORD_LEN * s.NOF_BUFFERS_TO_RECEIVE * s.DEAD_TIME_RATIO), 4)} seconds \n')

        if s.SWITCH and (acqLoopNum % s.SWITCH_DUTYCYCLE == 0):
            switchPos = 1
            arduino.switch(switchPos)
            print('sleeping')
            time.sleep(s.SWITCH_SLEEP_TIME)
            #avgSpec.collectAndSumFft()
            date_time_done = str(datetime.datetime.now())
            #fftSumCopy = avgSpec.fftSum.clone()
            #numFftCopy = avgSpec.numFft[:]
            if s.TEMPERATURE:
                temperature = arduino.getTemp()

            '''t = threading.Thread(target=daqh.procFftAndSave, 
                                args=(fftSumCopy, 
                                        numFftCopy,
                                        date_time_done,
                                        switchPos,
                                        temperature,
                                        s)
                                )'''
            #t.start()
            print(f'Acquisition of TWO buffers took   : {round((time.time()-ti),4)}')
            print(f'Theoretical time for TWO buffer  : {round((2/s.SAMPLE_RATE * s.CH0_RECORD_LEN * s.NOF_BUFFERS_TO_RECEIVE* s.DEAD_TIME_RATIO), 4)} seconds \n')

    #avgSpec.exit()
    date_time_exit = datetime.datetime.now()
    
    #save battery life
    if s.SWITCH:
        arduino.switch(0)

    time.sleep(1.5)
    print('\n#############################')
    print('RUN STATS\n')
    print(f'Started run at     : {str(date_time_start)}')
    print(f'Ended run at       : {str(date_time_exit)}')
    print(f'Called avgFFT()    : {s.NOF_ACQUISITIONS_TO_TAKE} times.')
    print(f'Each call recieved : {s.NOF_BUFFERS_TO_RECEIVE} buffers for a total of {s.NOF_ACQUISITIONS_TO_TAKE * s.NOF_BUFFERS_TO_RECEIVE} FFTs.')
    print(f'Total time         : {str(round((date_time_exit - date_time_start).total_seconds(), 4))} seconds')
    print(f'Theoretical time   : {round((1/s.SAMPLE_RATE * s.CH0_RECORD_LEN * s.NOF_BUFFERS_TO_RECEIVE * s.NOF_ACQUISITIONS_TO_TAKE* s.DEAD_TIME_RATIO), 4)} seconds')
    print('#############################\n')
    print('it is safe to ctrl-c out (or just close any plots)\n')


    # Simple plot of last acquisition 
    if s.PLOT_FINAL_SPEC:
        avgSpec_W_plotting = daqh.avgAndConvertFFT(
            avgSpec.fftSum,
            avgSpec.numFft,
            s
        )
        RBW    = s.SAMPLE_RATE/s.CH0_RECORD_LEN
        plt.figure()
        plt.title(f"{s.NOF_BUFFERS_TO_RECEIVE} FFTs Averaged, {round(RBW,1)} HZ RBW")
        #plt.plot(np.linspace(0,s.SAMPLE_RATE/2/1e6,s.CH0_RECORD_LEN//2),10*np.log10(avgSpec_W_plotting['chASpec_W'][1:]*1000), label = 'CH A')
        #plt.plot(np.linspace(0,s.SAMPLE_RATE/2/1e6,s.CH0_RECORD_LEN//2),10*np.log10(avgSpec_W_plotting['chBSpec_W'][1:]*1000), alpha = 0.8, label = 'CH B')
        plt.xlabel('Freq(MHz)')
        plt.ylabel('Power (dBm)')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.plot()
        plt.show()


