/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */

#include "settings.h"

#include "ADQAPI.h"
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include <signal.h>

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
  // Connect handler for CTRL+C interrupts
  signal(SIGINT, sigint_handler);

  /* Validate ADQAPI version. */
  switch (ADQAPI_ValidateVersion(ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR))
  {
  case 0:
    // ADQAPI is compatible
    break;
  case -1:
    printf("ADQAPI version is incompatible. The application needs to be recompiled and relinked against the installed ADQAPI.\n");
    return -1;
  case -2:
    printf("ADQAPI version is backwards compatible. It's suggested to recompile and relink the application against the installed ADQAPI.\n");
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
  int nof_active_channels = NOF_ACTIVE_CHANNELS;
  if (nof_active_channels > adq.constant.nof_channels)
  {
    printf("Limiting the number of active channels to %d (maximum available).\n",
           adq.constant.nof_channels);
    nof_active_channels = adq.constant.nof_channels;
  }

  /* Modify parameters (values from the header file "settings.h"). */

  for (int ch = 0; ch < nof_active_channels; ch++)
  {
    /* Analog frontend parameters */
    adq.afe.channel[ch].dc_offset = DC_OFFSET;

    /* Signal processing parameters */
    adq.signal_processing.dbs.channel[ch].level = DBS_LEVEL;
    adq.signal_processing.dbs.channel[ch].enabled = DBS_ENABLED;

    adq.signal_processing.sample_skip.channel[ch].skip_factor = SAMPLE_SKIP_FACTOR;

    /* Test pattern parameters */
    adq.test_pattern.channel[ch].source = TEST_PATTERN_SOURCE;

    /* Channel-specific event source parameters */
    adq.event_source.level.channel[ch].level = LT_LEVEL;
    adq.event_source.level.channel[ch].arm_hysteresis = LT_ARM_HYSTERESIS;

    /* Data acquisition parameters */
    adq.acquisition.channel[ch].nof_records = NOF_RECORDS_PER_BUFFER * NOF_BUFFERS_TO_RECEIVE;

    if (adq.constant.firmware.type == ADQ_FIRMWARE_TYPE_FWATD)
    {
      /* If the digitizer is running FWATD firmware, the acquisition engine must
         generate NOF_ACCUMULATIONS more records than we expect to received
         through data readout */
      adq.acquisition.channel[ch].nof_records *= NOF_ACCUMULATIONS;
    }

    adq.acquisition.channel[ch].record_length = RECORD_LENGTH;
    adq.acquisition.channel[ch].trigger_source = TRIGGER_SOURCE;
    adq.acquisition.channel[ch].trigger_edge = TRIGGER_EDGE;
    adq.acquisition.channel[ch].horizontal_offset = HORIZONTAL_OFFSET;

    /* Data transfer parameters */
    adq.transfer.channel[ch].record_size = adq.acquisition.channel[ch].bytes_per_sample
                                          * adq.acquisition.channel[ch].record_length;
    adq.transfer.channel[ch].record_buffer_size = NOF_RECORDS_PER_BUFFER
                                                * adq.transfer.channel[ch].record_size;
    adq.transfer.channel[ch].metadata_buffer_size = NOF_RECORDS_PER_BUFFER
                                                  * sizeof(struct ADQGen4RecordHeader);
    adq.transfer.channel[ch].record_length_infinite_enabled = 0;
    adq.transfer.channel[ch].metadata_enabled = 1;
    adq.transfer.channel[ch].nof_buffers = NOF_TRANSFER_BUFFERS;
  }

  /* Event source parameters */
  adq.event_source.periodic.period = PERIODIC_EVENT_SOURCE_PERIOD;
  adq.event_source.periodic.frequency = PERIODIC_EVENT_SOURCE_FREQUENCY;

  adq.event_source.port[ADQ_PORT_TRIG].pin[0].threshold = TRIGGER_THRESHOLD_V;
  adq.event_source.port[ADQ_PORT_SYNC].pin[0].threshold = TRIGGER_THRESHOLD_V;

  /* If the digitizer is running FWATD firmware, set up the ATD signal processing module */
  if (adq.constant.firmware.type == ADQ_FIRMWARE_TYPE_FWATD)
  {
    adq.signal_processing.atd.common.nof_accumulations = NOF_ACCUMULATIONS;
  }

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

  /* Send software triggers. This structure sends more triggers than required
     since all channels share a common source. */
  for (int ch = 0; ch < nof_active_channels; ++ch)
  {
    if (adq.acquisition.channel[ch].trigger_source == ADQ_EVENT_SOURCE_SOFTWARE)
    {
      if (adq.acquisition.channel[ch].nof_records != 0)
        printf("Generating software events on channel %d.\n", ch);

      for (int i = 0; i < adq.acquisition.channel[ch].nof_records; ++i)
      {
        if (ADQ_SWTrig(adq_cu, adq_num) != ADQ_EOK)
        {
          printf("Error: SWTrig failed.\n");
          goto exit;
        }
      }
    }
  }

  /* Data readout loop. */
  bool done = false;
  int nof_received_records[ADQ_MAX_NOF_CHANNELS] = {0};
  while (!done && !abort_acquisition)
  {
    struct ADQGen4Record *record = NULL;
    int channel = ADQ_ANY_CHANNEL;
    /* Wait for a record buffer. */
    int64_t bytes_received = ADQ_WaitForRecordBuffer(adq_cu, adq_num, &channel,
                                                     (void **)(&record), WAIT_TIMEOUT_MS, NULL);
    /* Negative values are errors. */
    if (bytes_received == ADQ_EAGAIN)
    {
      printf("Timeout while waiting %d ms for new record buffers.\n", WAIT_TIMEOUT_MS);
      continue;
    }
    else if (bytes_received < 0)
    {
      printf("Error: %" PRId64 ".\n", bytes_received);
      break;
    }

    /* Process the data. */
    printf("Got record %d w/ %" PRId64 " bytes, channel %d.\n",
           nof_received_records[channel], bytes_received, channel);

#if (WRITE_TO_FILE == 1)
    WriteRecordToFile(channel, nof_received_records[channel], record->data,
                      (size_t)bytes_received);
    printf("First 8 samples:\n");
    for (int i = 0; i < 10; ++i)
    {
      if (i > 0)
        printf(", ");

      if (record->header->data_format == ADQ_DATA_FORMAT_INT32)
        printf("%d", *((int32_t *)record->data + i));
      else
        printf("%d", *((int16_t *)record->data + i));
    }
    printf("\n\n");
#endif

    /* Return the buffer to the API. */
    int result = ADQ_ReturnRecordBuffer(adq_cu, adq_num, channel, record);
    if (result != ADQ_EOK)
    {
      printf("Failed to return a record buffer, code %d.\n", result);
      break;
    }

    /* Check if the acquisition should end. */
    ++nof_received_records[channel];
    done = true;
    for (int ch = 0; ch < nof_active_channels; ++ch)
    {
      if (adq.constant.firmware.type == ADQ_FIRMWARE_TYPE_FWATD)
      {
        if (adq.acquisition.channel[ch].nof_records != (nof_received_records[ch] * NOF_ACCUMULATIONS))
          done = false;
      }
      else
      {
        if (adq.acquisition.channel[ch].nof_records != nof_received_records[ch])
          done = false;
      }
    }
  }

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

exit:
  /* Delete the control unit object and the memory allocated by this application. */
  DeleteADQControlUnit(adq_cu);
  printf("Exiting the application.\n");
  fflush(stdout);
  return 0;
}
