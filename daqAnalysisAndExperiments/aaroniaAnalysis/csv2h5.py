import numpy as np
import pandas as pd
import h5py
import time
import os

hdf_filename = '8ghz0.h5'
csv_filename = '8ghz0_small.csv'

startTime = time.time()
#chunk through csv and append to hdf5 file
for chunk in pd.read_csv(csv_filename, chunksize=1):
    print(len(chunk['time'].iloc[0]))
    #name key with the time. Rounded to 3 digits
    hdf_key = str(round(chunk['time'].iloc[0], 3))
    #append the chunk to the h5 file at the given key.
    #note it writes to 'block0_values' dataset
    chunk.iloc[: , 1:].to_hdf(hdf_filename, key = hdf_key, mode = 'a')

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
