import numpy as np
import pandas as pd
import h5py

hdf_filename = '8GHz0.h5'
csv_filename = '8GHz0.csv'


#chunk through csv and append to hdf5 file
for chunk in pd.read_csv(csv_filename, chunksize=1):
    #name key with the time
    hdf_key = str(chunk['time'].iloc[0])
    #append the chunk to the h5 file at the given key.
    #note it writes to 'block0_values' dataset
    chunk.iloc[: , 1:].to_hdf(hdf_filename, key = hdf_key, mode = 'a')


#clean up unused data sets that pandas added
f = h5py.File(hdf_filename, 'a')
freqs = f
for time in f.keys():
    del(f[time]['block0_items'])
    del(f[time]['axis0'])
    del(f[time]['axis1'])
