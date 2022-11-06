from audioop import avg
import h5py
import numpy as np

hdf_filename = '8GHz0.h5'

f = h5py.File(hdf_filename, 'r')
numFreqs = 999 #sorry you have to hard code this

#choose if you want linear or log averaging
logAvg = True

#init some things for the loop
totalSpec = np.zeros(numFreqs)
numSpec = 0

if logAvg:
    for time in f.keys():
        #compute totalSpec
        totalSpec += f[time]['block0_values'][0,:]
        #count how many are averaged
        numSpec += 1
    #avgSpec is the total/ number you averaged together
    avgSpec = totalSpec/numSpec

else:
    for time in f.keys():
        #load dBm spec
        spec_dBm = f[time]['block0_values'][0,:]
        #convert to a spectum in Watts
        spec_W = [(10**(pow_dBm/10))/1000 for pow_dBm in spec_dBm]
        #compute totalSpec
        totalSpec += spec_W
        #count how many are averaged
        numSpec += 1
    #avgSpec is the total / number you averaged together
    avgSpec = totalSpec/numSpec

print('num spectra = ', numSpec)
print('head avgSpec = ', avgSpec[0:5])

    