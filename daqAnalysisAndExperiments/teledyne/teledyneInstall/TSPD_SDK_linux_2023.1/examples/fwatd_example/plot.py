#!/usr/bin/env python

"""
 Copyright 2019 Teledyne Signal Processing Devices Sweden AB
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

file_name = ["dataA.out"]
nsamples = 0  # Number of samples to plot, set to 0 to plot all samples

# Data type to use for binary data. For ASCII this value is ignored.
# Possible values are:
# 'int64' : 64-bit signed integer
# 'int32' : 32-bit signed integer
# 'int16' : 16-bit signed integer

binary_dtype = "int64"

for filen in file_name:
    try:
        data = np.loadtxt(filen)
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
        # The data is probably binary. Decode according to specified dtype
        data = np.fromfile(filen, dtype=binary_dtype)
    except FileNotFoundError as e:
        print("ERROR: '{}' not found".format(e.filename))
        sys.exit()

    if nsamples:
        data = np.array(data[:nsamples])

    fig = plt.figure(1)
    plt.clf()
    plt.plot(data, "-*")
    plt.show()
