import serial
import serial.tools.list_ports
import getpass
import os
import pyvisa as visa
import matplotlib.pyplot as plt
import numpy as np
import time

rm = visa.ResourceManager()
allDevices = rm.list_resources()
print(allDevices)

for aDev in allDevices:
    if 'RSA5' in aDev:
        rigolName = aDev
        break

startFreq = 45 * 10**6 #Hz
stopFreq =  1005 * 10**6 #Hz
nPoints = 10000 
rbw = 10000 #Hz
freqs = np.linspace(startFreq, stopFreq, nPoints)

INST = rm.open_resource(rigolName)

if 0: #write params. Otherwise just collect spectrum
    INST.write(':TRAC:AVER:COUN 10')
    INST.write(':SWE:POIN ' + str(nPoints))
    INST.write('FREQ:STAR ' + str(startFreq))
    INST.write('FREQ:STOP ' + str(stopFreq))
    INST.write('SENS:BAND:RES ' + str(rbw))


    INST.write(':INIT:CONT OFF')
    INST.write(':INITiate:IMMediate')
    print('scanning...')
    time.sleep(5)

specStr = (INST.query('TRAC:DATA? TRACE1'))

specArr = np.array([float(i) for i in specStr.split(',')])
print('got it')
print()


INST.close()



specAndFreqArr = np.zeros((nPoints, 2))
specAndFreqArr[:, 0] = freqs
specAndFreqArr[:, 1] = specArr

directory   = '/drBigBoy/darkRadio/daqAnalysisAndExperiments/run1p4/thermalNoiseVsH/data/'
fileName    = 'rigolNoiseFloor_ampOff_RBW10k_freq45_1005MHz_8_17_23'
#fileName    = 'calibration_5_8_23'
np.save(directory + fileName, specAndFreqArr)
print('saved')

plt.close('all')
plt.figure()
plt.plot(freqs/1e6, specArr)
plt.show()