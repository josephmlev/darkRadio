import numpy as np
import pandas as pd
import h5py
import time
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

hdf_filename = '8ghz0.h5'
csv_filename = '8ghz0_gitignore.csv'


startTime = time.time()


def dBm2Watts(spec_dBm):
    return 10**(spec_dBm/10)/1000

def chunkCsv(chunk):
    #append the chunk to the h5 file at the given key.
    #note it writes to 'block0_values' dataset
    spec = chunk.iloc[: , 1:]
    spec_dBm = dBm2Watts(spec)
    #print(spec.mean(axis = 0))

    return np.array(spec_dBm.mean(axis = 0))

p = Pool(processes = 16)
spec = p.map(chunkCsv, pd.read_csv(csv_filename, chunksize=100))

print('done')
spec = np.asarray(spec)
print(len(spec[0]))

#print(specConcat.mean(axis = 0))
#print(specConcat)

plt.close('all')
plt.figure()
#plt.plot(specConcat.mean())
#plt.show()



'''
print('Time to make h5 file = ', round(time.time() - startTime, 3), 'seconds')
print('csv filesize = ', os.path.getsize('./' + csv_filename)/1e6, 'MB')
print('h5 filesize = ', os.path.getsize('./' + hdf_filename)/1e6, 'MB')


#clean up unused data sets that pandas added
f = h5py.File(hdf_filename, 'a')

for time in f.keys():
    del(f[time]['block0_items'])
    del(f[time]['axis0'])
    del(f[time]['axis1'])
f.close()


print('h5 filesize (after clean up) = ', os.path.getsize('./' + hdf_filename)/1e6, 'MB')
'''