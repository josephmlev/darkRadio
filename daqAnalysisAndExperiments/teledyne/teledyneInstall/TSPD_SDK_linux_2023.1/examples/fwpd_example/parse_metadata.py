#!/usr/bin/env python3
# vim:fileencoding=utf-8

"""
 Copyright 2018 Teledyne Signal Processing Devices Sweden AB
"""

import numpy as np
import glob

from parse_header import parse_headers_from_file, to_signed

# "Constants"
SIZE_METADATA_PACKET = 8


class Error(Exception):
    pass


class MetadataPacket:
    record_number = 0
    record_timestamp = 0
    time_over_threshold = 0
    peak_value = 0
    peak_value_timestamp = 0

    def __init__(
        self,
        record_number,
        record_timestamp,
        time_over_threshold,
        peak_value,
        peak_value_timestamp,
    ):
        self.record_number = record_number
        self.record_timestamp = record_timestamp
        self.time_over_threshold = time_over_threshold
        self.peak_value = peak_value
        self.peak_value_timestamp = peak_value_timestamp


def parse_metadata(data_file, header_file, device_type, print_data=True):
    """
    Parse metadata file collected by ADQ14-FWPD example.

    Arguments:
        data_file:
            path to file containg the data records
        header_file:
            path to file containg the headers
    """

    data = np.fromfile(data_file, dtype="uint8")

    # Parse header information
    headers = parse_headers_from_file(header_file)

    if print_data:
        print("\nParsing data file {}".format(data_file))
        print("Found {} record headers".format(len(headers)))

    # Iterate over records
    record_start_idx = 0
    metadata_packets = []
    for k in range(len(headers)):
        metadata_packets_record = []

        if device_type == "adq7":
            if not headers[k].UserID & (1 << 1):
                if print_data:
                    print("Detected raw data record, skipping")
                continue

        rec_num = headers[k].RecordNumber
        rec_len = headers[k].RecordLength * 2  # 1 sample is 2 bytes
        rec_ts = headers[k].Timestamp

        # Extract data corresponding to record
        rec_data = data[record_start_idx : record_start_idx + rec_len]

        nof_metadata_packets = len(rec_data) // SIZE_METADATA_PACKET

        if print_data:
            print(
                "\nRecord number {}, UserID {}".format(headers[k].RecordNumber, headers[k].UserID)
            )
            print("Found {} metadata packets".format(nof_metadata_packets))
            print("Record timestamp {}".format(hex(headers[k].Timestamp)))

        # Iterate over metadata packets
        # (first 8 bytes are gpio, second 8 bytes are zero padding)
        nof_padding_packets = 0
        for i in range(nof_metadata_packets):
            metadata_packet = rec_data[i * SIZE_METADATA_PACKET : (i + 1) * SIZE_METADATA_PACKET]
            extreme_value_timestamp_bytes = metadata_packet[0:4]
            extreme_value_bytes = metadata_packet[4:6]
            time_over_threshold_bytes = metadata_packet[6:8]

            peak_value_timestamp = 0
            for bidx, b in enumerate(extreme_value_timestamp_bytes):
                peak_value_timestamp += b << (8 * bidx)

            peak_value = 0
            for bidx, b in enumerate(extreme_value_bytes):
                peak_value += b << (8 * bidx)

            time_over_threshold = 0
            for bidx, b in enumerate(time_over_threshold_bytes):
                time_over_threshold += b << (8 * bidx)

            # peak_value_timestamp = to_signed(peak_value_timestamp, 32)
            peak_value = to_signed(peak_value, 16)
            # time_over_threshold = to_signed(time_over_threshold, 16)

            if not (peak_value == 0 and peak_value_timestamp == 0 and time_over_threshold == 0):

                if print_data:
                    print("**** Parsing packet {}, record {} ****".format(i, k))
                    print("Peak value timestamp: {}".format(peak_value_timestamp))
                    print("Peak value:           {}".format(peak_value))
                    print("Time over threshold:  {}".format(time_over_threshold))
                    print("**************************")

                pkg = MetadataPacket(
                    rec_num,
                    rec_ts,
                    time_over_threshold,
                    peak_value,
                    peak_value_timestamp,
                )

                metadata_packets_record += [pkg]
            else:
                nof_padding_packets += 1

        if print_data:
            print("{} padding packets".format(nof_padding_packets))

        record_start_idx += rec_len
        metadata_packets += [metadata_packets_record]

    return metadata_packets


if __name__ == "__main__":
    data_files = glob.glob("./metadata_ch_*.bin")
    header_files = glob.glob("./headers_ch_*.bin")

    for df, hf in zip(data_files, header_files):
        if "adq7" in df.lower():
            parse_metadata(df, hf, "adq7")
        elif "adq14" in df.lower():
            parse_metadata(df, hf, "adq14")
        else:
            raise Error("Failed to identify a device type (ADQ14/ADQ7) from the filename.")
