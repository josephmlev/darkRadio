/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 *
 * This example illustrates the "data transfer" API interface.
 */

#include "ADQAPI.h"
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include <signal.h>
#include <time.h>

/*
 * Enable writing data to disk by setting WRITE_TO_FILE 1. NOTE: The data
 * throughput will be *much* lower when writing to disk
 */
#define WRITE_TO_FILE 0
#define BASE_FILENAME "data/data"

// Handler for CTRL+C interrupts
static volatile int abort_acquisition = 0;
void sigint_handler(int dummy)
{
  (void)dummy;
  printf("Caught Ctrl-C. Aborting..\n");
  abort_acquisition = 1;
}

#if (WRITE_TO_FILE == 1)
static int WriteRecordToFile(int channel, int record, void *buf, size_t len)
{
  char filename[256] = "";
  sprintf(filename, "%s_ch%d_r%d.bin", BASE_FILENAME, channel, record);

  FILE *fp = fopen(filename, "wb");
  if (fp == NULL)
  {
    printf("Failed to open the file '%s' for writing.\n", filename);
    return -1;
  }

  size_t bytes_written = fwrite(buf, 1, len, fp);
  if (bytes_written != len)
  {
    printf("Failed to write %zu bytes to the file '%s', wrote %zu bytes.\n", len, filename,
           bytes_written);
    fclose(fp);
    return -1;
  }

  fclose(fp);
  return 0;
}
#endif

int main()
{
  const int NOF_RECORDS_PER_BUFFER = 1000;
  const int NOF_BUFFERS_TO_RECEIVE = 20;
  const int RECORD_LENGTH = 1024;
  const float PERIODIC_EVENT_SOURCE_FREQUENCY = 800e3;

  // Connect handler for CTRL+C interrupts
  signal(SIGINT, sigint_handler);

  /* Validate ADQAPI version. */
  switch (ADQAPI_ValidateVersion(ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR))
  {
  case 0:
    // ADQAPI is compatible
    break;
  case -1:
    printf("ADQAPI version is incompatible. The application needs to be recompiled and relinked "
           "against the installed ADQAPI.\n");
    return -1;
  case -2:
    printf("ADQAPI version is backwards compatible. It's suggested to recompile and relink the "
           "application against the installed ADQAPI.\n");
    break;
  }

  /* Initialize the a handle to the ADQ control unit object. */
  void *adq_cu = CreateADQControlUnit();
  if (adq_cu == NULL)
  {
    printf("Failed to create a handle to an ADQ control unit object.\n");
    return -1;
  }

  /* Enable the error trace log. */
  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");

  /* List the available devices connected to the host computer. */
  struct ADQInfoListEntry *adq_list = NULL;
  unsigned int nof_devices = 0;
  if (!ADQControlUnit_ListDevices(adq_cu, &adq_list, &nof_devices))
  {
    printf("ListDevices failed!\n");
    return -1;
  }

  if (nof_devices == 0)
  {
    printf("No device connected.\n");
    goto exit;
  }
  else if (nof_devices != 1)
  {
    printf("Only one ADQ is supported by this example.\n");
    goto exit;
  }

  /* Since this example only supports one device, we always open the device at
     list index zero. */
  int device_to_open_idx = 0;

  printf("Configuring device... ");
  if (ADQControlUnit_SetupDevice(adq_cu, device_to_open_idx))
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Device ids for the ADQ_* functions start at 1, representing the first
     device listed by ADQControlUnit_ListDevices(). */
  int adq_num = 1;

  /* Initialize parameters. */
  struct ADQParameters adq;
  if (ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_TOP, &adq) != sizeof(adq))
  {
    printf("Failed to initialize digitizer parameters.\n");
    goto exit;
  }

  /* Cap the number of active channels to the number of available channels. */
  int nof_active_channels = 1;
  if (nof_active_channels > adq.constant.nof_channels)
  {
    printf("Limiting the number of active channels to %d (maximum available).\n",
           adq.constant.nof_channels);
    nof_active_channels = adq.constant.nof_channels;
  }

  adq.event_source.periodic.frequency = PERIODIC_EVENT_SOURCE_FREQUENCY;

  /* Configure data acquisition for channel 0. */
  for (int ch = 0; ch < nof_active_channels; ++ch)
  {
    adq.acquisition.channel[ch].nof_records = -1;
    adq.acquisition.channel[ch].record_length = RECORD_LENGTH;
    adq.acquisition.channel[ch].trigger_source = ADQ_EVENT_SOURCE_PERIODIC;
    adq.acquisition.channel[ch].trigger_edge = ADQ_EDGE_RISING;
    adq.acquisition.channel[ch].horizontal_offset = 0;

    adq.transfer.channel[ch].record_size = adq.acquisition.channel[ch].bytes_per_sample
                                           * adq.acquisition.channel[ch].record_length;
    adq.transfer.channel[ch].record_length_infinite_enabled = 0;
    adq.transfer.channel[ch].record_buffer_size = NOF_RECORDS_PER_BUFFER * adq.transfer.channel[ch].record_size;
    adq.transfer.channel[ch].metadata_buffer_size = NOF_RECORDS_PER_BUFFER * sizeof(struct ADQGen4RecordHeader);
    adq.transfer.channel[ch].metadata_enabled = 1;
    adq.transfer.channel[ch].nof_buffers = ADQ_MAX_NOF_BUFFERS;
    // adq.test_pattern.channel[ch].source = ADQ_TEST_PATTERN_SOURCE_TRIANGLE;
  }

  adq.transfer.common.write_lock_enabled = 1;
  adq.transfer.common.transfer_records_to_host_enabled = 1;

  /*
   * Enable "data transfer" interface by setting marker_mode to
   * ADQ_MARKER_MODE_HOST_MANUAL
   */
  adq.transfer.common.marker_mode = ADQ_MARKER_MODE_HOST_MANUAL;

  printf("Configuring digitizer parameters... ");
  if (ADQ_SetParameters(adq_cu, adq_num, &adq) == sizeof(adq))
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Start the data acquisition. */
  printf("Start acquiring data... ");
  if (ADQ_StartDataAcquisition(adq_cu, adq_num) == ADQ_EOK)
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Data readout loop. */
  bool done = false;
  int total_nof_received_records = 0;
  double total_nof_received_bytes = 0.0;

#ifdef LINUX
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
#else
  /* clock() in windows returns wall clock */
  clock_t start = clock();
#endif

  while (!done && !abort_acquisition)
  {
    const int WAIT_TIMEOUT_MS = 1000;
    struct ADQP2pStatus status;
    int result = ADQ_WaitForP2pBuffers(adq_cu, adq_num, &status, WAIT_TIMEOUT_MS);

    if (result != ADQ_EOK)
    {
      printf("Error:ADQ_WaitForP2pBuffers failed. Status: %d\n", result);
      break;
    }

    for (int ch = 0; ch < adq.constant.nof_channels; ++ch)
    {
      uint64_t unlock_mask = 0;
      for (int i = 0; i < status.channel[ch].nof_completed_buffers; ++i)
      {
        int completed_buffer_index = status.channel[ch].completed_buffers[i];
        unlock_mask |= 1llu << completed_buffer_index;

#if (WRITE_TO_FILE == 1)
        int16_t *data = (int16_t *)adq.transfer.channel[ch].record_buffer[completed_buffer_index];

        /*
         * NOTE: timestamp and record_start header fields are *not* valid when
         * using ADQ_MARKER_MODE_HOST_MANUAL
         */
        struct ADQGen4RecordHeader *headers = (struct ADQGen4RecordHeader *)adq.transfer.channel[ch]
                                                .metadata_buffer[completed_buffer_index];
        for (int rec_idx = 0; rec_idx < NOF_RECORDS_PER_BUFFER; ++rec_idx)
        {
          if (WriteRecordToFile(ch, headers[rec_idx].record_number, data + (rec_idx * RECORD_LENGTH),
              RECORD_LENGTH))
          {
            done = true;
            break;
          }
        }
#endif
      }

      result = ADQ_UnlockP2pBuffers(adq_cu, adq_num, ch, unlock_mask);
      if (result != ADQ_EOK)
      {
        printf("Error: ADQ_UnlockP2pBuffers failed. Status: %d\n", result);
        done = true;
        break;
      }

      total_nof_received_bytes += status.channel[ch].nof_completed_buffers * adq.transfer.channel[ch].record_buffer_size;
      total_nof_received_records += status.channel[ch].nof_completed_buffers * NOF_RECORDS_PER_BUFFER;
    }

    if (total_nof_received_records >= (NOF_BUFFERS_TO_RECEIVE * NOF_RECORDS_PER_BUFFER * nof_active_channels))
      done = true;
  }

#ifdef LINUX
  clock_gettime(CLOCK_REALTIME, &stop);
  double diff_ms = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6;
#else
  clock_t stop = clock();
  double diff_ms = (((double)(stop - start)) / CLOCKS_PER_SEC) * 1000;
#endif

  /* Stop the data acquisition process. */
  printf("Stop acquiring data... ");
  int result = ADQ_StopDataAcquisition(adq_cu, adq_num);
  switch (result)
  {
  case ADQ_EOK:
  case ADQ_EINTERRUPTED:
    printf("success.\n");
    break;
  default:
    printf("failed, code %d.\n", result);
    break;
  }

  printf("\nCollected %d records in %f ms\n", total_nof_received_records, diff_ms);
  printf("Estimated trigger frequency: %f KHz\n", (total_nof_received_records)/diff_ms);
  printf("Data throughput: %.2f MB/s\n\n", (total_nof_received_bytes/diff_ms)/1000);

exit:
  /* Delete the control unit object and the memory allocated by this application. */
  DeleteADQControlUnit(adq_cu);
  printf("Exiting the application.\n");
  fflush(stdout);
  return 0;
}
