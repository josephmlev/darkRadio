/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */
#include <stdio.h>
#include <time.h>
#include "ADQAPI.h"
#include "adnvds.h"
#include "helpers.h"

#ifdef LINUX
#include <sys/time.h>
#include <string.h>
#endif

#ifdef LINUX
struct timeval timer_start_t[3];
struct timeval timer_stop_t;

void timer_start(unsigned int timer_no)
{// TODO, error if timer_no to high.
  gettimeofday(&timer_start_t[timer_no], NULL);
}

double timer_time_seconds(unsigned int timer_no)
{
  double t_start,t_stop;
  gettimeofday(&timer_stop_t, NULL);
  t_start = (double)(timer_start_t[timer_no].tv_sec) + (double)(timer_start_t[timer_no].tv_usec) / 1000000.0;
  t_stop = (double)(timer_stop_t.tv_sec) + (double)(timer_stop_t.tv_usec) / 1000000.0;
  return(t_stop-t_start);
}
#else
clock_t clk_ref[3];
void timer_start(unsigned int timer_no)
{
  clk_ref[timer_no] = clock();
  return;
}

double timer_time_seconds(unsigned int timer_no)
{
  clock_t t = clock();
  return ((double)(t - clk_ref[timer_no]) / CLOCKS_PER_SEC);
}
#endif

/* Helper for printing values in B / kB / MB / GB / TB */
void print_bytes(double value)
{
  if (value > 1e12)
    printf("%.2f TB", value / 1e12);
  else if (value > 1e9)
    printf("%.2f GB", value / 1e9);
  else if (value > 1e6)
    printf("%.2f MB", value / 1e6);
  else if (value > 1e3)
    printf("%.2f kB", value / 1e3);
  else
    printf("%.0f B", value);
}

void print_time(int seconds)
{
  int seconds_mod = seconds;
  if (seconds >= 3600.0)
  {
    printf("%dh ", seconds_mod / 3600);
    seconds_mod -= (seconds_mod / 3600);
  }
  if (seconds >= 60.0)
  {
    printf("%dm ", seconds_mod / 60);
    seconds_mod -= (seconds_mod / 60);
  }
  printf("%ds", seconds_mod);
}

void print_disk_health_info(void* adnvds_handle, char* serial)
{
  struct adnvds_device_health dev_health;
  memset(&dev_health, 0, sizeof(dev_health));

  int result = adnvds_dev_get_health_info(
    adnvds_handle, (void *)serial, &dev_health
  );

  if (result != ADNVDS_STATUS_OK)
  {
    printf("[STATUS] ERROR: Failed to get health info for disk %s, code %d\n", serial, result);
    return;
  }

  printf("[STATUS] Disk %s - ", serial);
  printf("Composite Temperature: %d [degC]\n", (int)dev_health.temperature - 273);
}

void print_status(
  double interval_sec,
  int nof_channels,
  int overflow_status,
  int64_t *datarate_tracker_data_bytes,
  int64_t *datarate_tracker_metadata_bytes,
  int64_t *datarate_tracker_monitor_bytes,
  int64_t *stored_bytes_data,
  int64_t *stored_bytes_metadata)
{
  printf("\n");

  printf("[STATUS] Timestamp:                        ");
  print_time((int)timer_time_seconds(TIMER_TOTAL_ACQUISITION));
  printf("\n");

  printf("[STATUS] Disk data rate per channel:       ");
  for (int ch = 0; ch < nof_channels; ch++)
  {
    print_bytes((double)(datarate_tracker_data_bytes[ch] + datarate_tracker_metadata_bytes[ch]) / interval_sec );
    printf("/s  ");
  }
  printf("\n");

  printf("[STATUS] Monitoring data rate per channel: ");
  // FIXME: Print only [nof_channels, 2*nof_channels], or print for full-rate as well?
  for (int ch = nof_channels; ch < 2 * nof_channels; ch++)
  {
    print_bytes((double)datarate_tracker_monitor_bytes[ch] / interval_sec);
    printf("/s  ");
  }
  printf("\n");

  printf("[STATUS] Stored data per channel:          ");
  for (int ch = 0; ch < nof_channels; ch++)
  {
    print_bytes((double)stored_bytes_data[ch]);
    printf("  ");
  }
  printf("\n");

  printf("[STATUS] Stored metadata per channel:      ");
  for (int ch = 0; ch < nof_channels; ch++)
  {
    print_bytes((double)stored_bytes_metadata[ch]);
    printf("  ");
  }
  printf("\n");

  printf("[STATUS] Digitizer streaming overflow:     %d\n", overflow_status);

  for (int ch = 0; ch < ADQ_MAX_NOF_CHANNELS; ch++)
  {
    datarate_tracker_data_bytes[ch] = 0;
    datarate_tracker_metadata_bytes[ch] = 0;
    datarate_tracker_monitor_bytes[ch] = 0;
  }
}
