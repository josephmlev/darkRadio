/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */

#ifndef DISK_STREAMING_SETTINGS_H
#define DISK_STREAMING_SETTINGS_H

/* The serial numbers of the disks to be read */
static char *disk_serials[16] = {
  "S5HSNC0N300433V", "S5HSNC0N300437M", NULL
};

/* The maximum amount of metadata blocks to read out from each data set */
#define MAX_METADATA_BLOCK_READS_PER_DATASET 8

/* The maximum amount of data bytes to read out from each data set */
static uint64_t MAX_DATA_READ_BYTES_PER_DATASET = 1024 * 1024;

/* Set to true to finish parsing the raw metadata blocks into finalized
   metadata */
static bool PARSE_RAW_METADATA = true;

/* Set to true to print all the read data */
static bool PRINT_DATA_BYTES = false;

/* Set to true to write the read data to a file */
static bool WRITE_DATA_BYTES_TO_FILE = true;

/* File name for writing read data to file */
static char *DATA_FILE_NAME = "./stored_data.bin";

/* Limit ADNVDS device attachment attempts. Multiple rapid attach calls might
   cause the OS to throttle/temporarily blacklist operations. */
#define ADNVDS_ATTACH_ATTEMPTS 5
#define ADNVDS_ATTACH_WAIT_TIME 1000

#endif
