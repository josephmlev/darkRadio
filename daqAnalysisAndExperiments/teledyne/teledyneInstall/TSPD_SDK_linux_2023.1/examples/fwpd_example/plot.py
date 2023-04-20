#!/usr/bin/env python3
# vim:fileencoding=utf-8
#
# Copyright 2019 Teledyne Signal Processing Devices Sweden AB
#

import os
import numpy as np
import matplotlib.pyplot as plt
from parse_header import parse_headers_from_file

file_name = ["data_ch_A.bin"]
header_name = ["headers_ch_A.bin"]
# file_name = ["data_ch_A.bin", "data_ch_B.bin"]
# header_name = ["headers_ch_A.bin", "headers_ch_B.bin"]

# Data type to use for binary data. For ASCII this value is ignored.
# Possible values are:
# 'int64' : 64-bit signed integer
# 'int32' : 32-bit signed integer
# 'int16' : 16-bit signed integer

binary_dtype = "int16"

headers = []

for k, filen in enumerate(header_name):
    if not os.path.exists(filen):
        print("WARNING: '{}' not found".format(filen))
        continue

    headers += [parse_headers_from_file(filen)]

    # Parse header information

    print("{} records for {}".format(len(headers[k]), filen))
    print("-------------")
    for header in headers[k]:
        print("Channel {}".format(header.Channel))
        print("Record num {}".format(header.RecordNumber))
        print("Record length {}".format(header.RecordLength))
        print("Timestamp {}".format(hex(header.Timestamp)))
        print()

        if header.UserID & 0x1:
            print("Padding record")

        print()

for k, filen in enumerate(file_name):
    try:
        data = np.loadtxt(filen)
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
        # The data is probably binary. Decode according to specified dtype
        data = np.fromfile(filen, dtype=binary_dtype)
    except FileNotFoundError as e:
        print("WARNING: '{}' not found".format(e.filename))
        continue

    record_length = np.array([], dtype=int)
    record_start = np.array([], dtype=int)

    record_start_idx = 0
    for hdr in headers[k]:
        if hdr.UserID & 0x2:
            continue
        rec_len = hdr.RecordLength
        record_start = np.append(
            record_start, -hdr.RecordStart // hdr.SamplePeriod + sum(record_length)
        )
        record_length = np.append(record_length, rec_len)

    if not (headers[k][0].UserID & 0x2):
        fig = plt.figure(k)
        plt.clf()
        plt.plot(data, ".-")
        plt.plot(record_start, data[record_start], "o")
        plt.title("Result from {}".format(filen))
        plt.legend(["Data", "Trigger"])

plt.show()
