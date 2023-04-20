#ifndef DISK_STREAMING_HELPERS_H
#define DISK_STREAMING_HELPERS_H

#include "ADQAPI.h"
#include "adnvds.h"

#define TIMER_NO_DATA_RECEIVED 0
#define TIMER_PERIODIC_PRINTOUT 1
#define TIMER_TOTAL_ACQUISITION 2

void timer_start(unsigned int timer_no);
double timer_time_seconds(unsigned int timer_no);
void print_bytes(double value);
void print_time(int seconds);
void print_disk_health_info(void* adnvds_handle, char* serial);
void print_status(
  double interval_sec,
  int nof_channels,
  int overflow_status,
  int64_t *datarate_tracker_data_bytes,
  int64_t *datarate_tracker_metadata_bytes,
  int64_t *datarate_tracker_monitor_bytes,
  int64_t *stored_bytes_data,
  int64_t *stored_bytes_metadata
);

#endif
