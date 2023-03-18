# vim:fileencoding=utf-8
#
# Copyright 2019 Teledyne Signal Processing Devices Sweden AB
#

import ctypes as ct
import os


def to_signed(x, nbits):
    if x >= 2 ** (nbits - 1):
        x -= 2**nbits
    return x


class ADQRecordHeader(ct.Structure):
    _fields_ = [
        ("RecordStatus", ct.c_uint8),
        ("UserID", ct.c_uint8),
        ("Channel", ct.c_uint8),
        ("DataFormat", ct.c_uint8),
        ("SerialNumber", ct.c_uint32),
        ("RecordNumber", ct.c_uint32),
        ("SamplePeriod", ct.c_int32),
        ("Timestamp", ct.c_uint64),
        ("RecordStart", ct.c_int64),
        ("RecordLength", ct.c_uint32),
        ("GeneralPurpose0", ct.c_uint16),
        ("GeneralPurpose1", ct.c_uint16),
    ]


def parse_headers_from_file(filename):
    nof_headers = os.path.getsize(filename) // ct.sizeof(ADQRecordHeader())
    headers = []

    with open(filename, "rb") as f:
        headers = [ADQRecordHeader() for x in range(nof_headers)]
        i = 0
        while i < nof_headers and f.readinto(headers[i]) is not None:
            i += 1

    return headers
