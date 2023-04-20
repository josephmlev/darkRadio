#!/usr/bin/env python3
"""
    Copyright 2020 Teledyne Signal Processing Devices Sweden AB

    Plot data collected with the data transfer example.

    Requires:
        - Python 3
        - Numpy
        - Matplotlib

    Change the update the 'file_list' with files to plot.
"""

import numpy as np
import matplotlib.pyplot as plt

# List of files to plot
file_list = ["data/data_ch0_r0.bin"]

# Data type to use for binary data. For ASCII this value is ignored.
# Possible values are:
# 'int64' : 64-bit signed integer
# 'int32' : 32-bit signed integer
# 'int16' : 16-bit signed integer
binary_dtype = "int16"

for k, filen in enumerate(file_list):
    try:
        data = np.loadtxt(filen)
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
        # The data is probably binary. Decode according to specified dtype.
        data = np.fromfile(filen, dtype=binary_dtype)
    except FileNotFoundError as e:
        print("WARNING: '{}' not found".format(e.filename))
        continue

    fig = plt.figure(k)
    plt.clf()
    plt.plot(data, ".-")
    plt.title("Result from {}".format(filen))
    plt.grid(which="both")

plt.show()
