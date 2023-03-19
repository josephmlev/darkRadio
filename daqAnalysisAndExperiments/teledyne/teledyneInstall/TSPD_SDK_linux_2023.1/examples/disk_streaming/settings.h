/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */

#ifndef DISK_STREAMING_SETTINGS_H
#define DISK_STREAMING_SETTINGS_H

#include "ADQAPI.h"
#include "adnvds.h"

struct DiskStorage
{
  void *adnvds_handle;
  unsigned int *ep_group_handle;
  int drive_interleaving;
  int n_drives_total;
  int n_drives_per_channel[ADQ_MAX_NOF_CHANNELS];
  char *serials[ADQ_MAX_NOF_CHANNELS][256];
};

// Initialize the struct with values according to use case and system
struct DiskStorage disk_information = {
  .adnvds_handle = NULL,
  .ep_group_handle = NULL,
  .drive_interleaving = 1,
  .n_drives_total = 1,
  .n_drives_per_channel = {
    1,
    0,
  },
  .serials = {
    // ADNVDS requires a null-terminated list of disk device
    // serial numbers for each channel
    {"S5GXNF0R781470D", NULL},
    {NULL}
  }
};

/* The trigger event source for each channel. Valid values are:
   - ADQ_EVENT_SOURCE_SOFTWARE
   - ADQ_EVENT_SOURCE_TRIG
   - ADQ_EVENT_SOURCE_LEVEL
   - ADQ_EVENT_SOURCE_SYNC
   - ADQ_EVENT_SOURCE_PERIODIC */
static const enum ADQEventSource TRIGGER_SOURCE[ADQ_MAX_NOF_CHANNELS] = {
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
  ADQ_EVENT_SOURCE_PERIODIC,
};

/* The trigger event source for the monitoring channels */
static const enum ADQEventSource MONITOR_CHANNELS_TRIGGER_SOURCE = ADQ_EVENT_SOURCE_PERIODIC;

/* The edge sensitivity of the selected trigger event source. The support for
   these values varies between the event sources, e.g. ADQ_EVENT_SOURCE_SOFTWARE
   only supports ADQ_EDGE_RISING. */
static const enum ADQEdge TRIGGER_EDGE[ADQ_MAX_NOF_CHANNELS] = {
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
};

/* Periodic event source settings. The period is measured in ADC sampling
   periods. */
static const int PERIODIC_EVENT_SOURCE_PERIOD = 2000 * 1000;

/* Horizontal offset of the trigger point. Offset zero places the trigger point
   at the first sample in the record. */
static const int64_t HORIZONTAL_OFFSET[ADQ_MAX_NOF_CHANNELS] = {
  0, 0, 0, 0, 0, 0, 0, 0
};

/* The number of records to acquire for each channel. Set the value to zero to
   disable the channel. Set to ADQ_INFINITE_NOF_RECORDS to specify an unbounded
   acquisition. */
static const int64_t NOF_RECORDS[ADQ_MAX_NOF_CHANNELS] = {
  ADQ_INFINITE_NOF_RECORDS,
  0,
  0,
  0,
  0,
  0,
  0,
  0
};

/* Record lengths per channel. Set to ADQ_INFINITE_RECORD_LENGTH for
   continuous streaming. */
static const int64_t RECORD_LENGTH[ADQ_MAX_NOF_CHANNELS] = {
  65536,
  0,
  0,
  0,
  0,
  0,
  0,
  0
};

/* If metadata should be transferred more often than just once per record,
   set this interval to a value lower than the record length. */
static const int64_t FORCED_METADATA_INTERVAL[ADQ_MAX_NOF_CHANNELS] = {
  65536,
  65536,
  65536,
  65536,
  65536,
  65536,
  65536,
  65536
};

/* The following constant can be used to stop transfers after a certain amount
   of data in bytes has been stored for each channel. Use this when RECORD_LENGTH
   is set to ADQ_INFINITE_RECORD_LENGTH, or when NOF_RECORDS is set to
   ADQ_INFINITE_NOF_RECORDS as a stopping point if you do not want
   to fill the entire disk drives. */
static const int64_t DISK_STORAGE_LIMIT_BYTES[ADQ_MAX_NOF_CHANNELS] = {
  4 * 1024 * 1024 * 1024ll,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};

/* Set to 1 to enable streaming of records to the host application for monitoring */
static const int32_t HOST_STREAMING_RECORD_ENABLED[ADQ_MAX_NOF_CHANNELS] = {
  0, 0, 0, 0, 0, 0, 0, 0
};

/* Set to 1 to enable streaming of record data to disk */
static const int32_t DISK_STREAMING_RECORD_ENABLED[ADQ_MAX_NOF_CHANNELS] = {
  1, 0, 0, 0, 0, 0, 0, 0
};

/* Set to 1 to enable streaming of record metadata to disk*/
static const int32_t DISK_STREAMING_METADATA_ENABLED[ADQ_MAX_NOF_CHANNELS] = {
  1, 0, 0, 0, 0, 0, 0, 0
};

/* Choose reformatting strategy for the disk drives. */
static const int32_t FORMAT_DRIVE[ADQ_MAX_NOF_CHANNELS] = {
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
  ADNVDS_FORMAT_DEFAULT,
};

/* Choose a unique ID for each channel that will be stored in the disk metadata. */
static const int32_t CHANNEL_UUID[ADQ_MAX_NOF_CHANNELS] = {
  1, 2, 3, 4, 5, 6, 7, 8
};

/* Sample skip factor per channel. Note that it is not possible to set
   a separate sample skip factor for the monitoring channels compared to
   the standard channels. */
static const int SAMPLE_SKIP_FACTOR[ADQ_MAX_NOF_CHANNELS] = {
  1, 1, 1, 1, 1, 1, 1, 1
};

/* Amount of seconds to wait during which no new data is transferred
   before flushing and aborting the transfer. */
#define TIMEOUT_DSU_TRANSFER 2.0

/* Amount of seconds to wait between periodic printouts of disk data rate,
   total stored data, and temperature. */
#define PERIODIC_STATUS_PRINT_TIME 5.0

/* Limit ADNVDS device attachment attempts. Multiple rapid attach calls might
   cause the OS to throttle/temporarily blacklist operations. */
#define ADNVDS_ATTACH_ATTEMPTS 5
#define ADNVDS_ATTACH_WAIT_TIME 1000

#endif
