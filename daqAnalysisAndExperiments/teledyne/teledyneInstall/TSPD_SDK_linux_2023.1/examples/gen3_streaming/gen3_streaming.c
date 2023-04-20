/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */

#include "settings.h"
#include "ADQAPI.h"
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>
#include <signal.h>

#ifdef LINUX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
#endif

/* Handler for CTRL+C interrupts. */
static volatile bool abort_acquisition = false;
void sigint_handler(int dummy)
{
  (void)dummy;
  printf("Caught Ctrl-C. Aborting..\n");
  abort_acquisition = true;
}

static char label(int channel)
{
  static char labels[] = "ABCDEFGH";
  if ((channel >= 0) && (channel < (int)sizeof(labels)))
    return labels[channel];
  return '-';
}

#if (WRITE_TO_FILE != 0)
static int write_record_to_file(int channel, int record, void *buf, size_t len)
{
  char filename[256] = "";
  FILE *fp = NULL;
  size_t bytes_written = 0;
  sprintf(filename, "%s_ch%c_r%d.bin", BASE_FILENAME, label(channel), record);

  fp = fopen(filename, "wb");
  if (fp == NULL)
  {
    printf("Failed to open the file '%s' for writing.\n", filename);
    return -1;
  }

  bytes_written = fwrite(buf, 1, len, fp);
  if (bytes_written != len)
  {
    printf("Failed to write %zu bytes to the file '%s', wrote %zu bytes.\n", len, filename,
           bytes_written);
    fclose(fp);
    return -1;
  }

  printf("Wrote record %d to file '%s'.\n", record, filename);
  fclose(fp);
  return 0;
}
#endif

static void process_data(const struct ADQRecord *const record,
                         const struct ADQDataReadoutStatus *status, int channel,
                         int64_t bytes_received, enum ADQProductID_Enum pid)
{
  /* Any data processing would be performed in this step. This example prints
     some information about the record and optionally writes the data to file. */
  int i = 0;
  int16_t *record_data = (int16_t *)record->data;
  bool is_incomplete = status->flags & ADQ_DATA_READOUT_STATUS_FLAGS_INCOMPLETE;

  printf("Got %srecord %u from channel %c (%" PRId64 " B).\n", is_incomplete ? "part of " : "",
         record->header->RecordNumber, label(channel), bytes_received);
  printf("  Timestamp (trigger): %" PRIu64 " (%d ps units)\n", record->header->Timestamp,
         (pid == PID_ADQ14) ? 125 : 25);
  printf("  Timestamp (first sample): %" PRId64 " (%d ps units)\n",
         record->header->Timestamp + record->header->RecordStart, (pid == PID_ADQ14) ? 125 : 25);

  uint8_t lost_data = record->header->RecordStatus & 0xFu;
  if (lost_data > 0)
  {
    printf("  Lost data: (0x%02X)\n", record->header->RecordStatus & 0xFu);
    if (lost_data & 0x8u)
      printf("    At the end of the record.\n");
    if (lost_data & 0x4u)
      printf("    Within the record.\n");
    if (lost_data & 0x2u)
      printf("    At the beginning of the record.\n");
    if (lost_data & 0x1u)
      printf("    One or several preceding records.\n");
  }

  uint8_t dram_fill = (record->header->RecordStatus >> 4) & 0x7u;
  if (dram_fill > 0)
  {
    printf("  The DRAM is filled to more than %.1f%% of its maximum capacity.\n",
           12.5 * (double)dram_fill);
  }

  printf("  The first 8 samples are:\n    ");
  for (i = 0; i < 8; ++i)
  {
    if (i > 0)
      printf(", ");
    printf("%d", record_data[i]);
  }
  printf("\n  ");

#if (WRITE_TO_FILE != 0)
  write_record_to_file(channel, record->header->RecordNumber, record->data, bytes_received);
#endif
  printf("\n");
}

static void streaming(void *adq_cu, int adq_num, enum ADQProductID_Enum pid)
{
  /* This function implements the program flow suggested in the user guide
     (document id 20-2465). Please refer to that document for more information. */

  /* We need the following parameters to direct the flow in this functions. */
  int nof_channels = (int)ADQ_GetNofChannels(adq_cu, adq_num);
  struct ADQDataAcquisitionParameters acquisition;
  int result = ADQ_GetParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_ACQUISITION, &acquisition);
  if (result != sizeof(acquisition))
  {
    printf("Failed to get the data acquisition parameters.\n");
    return;
  }

  /* If MEMORY_OWNER_USER is defined, this example adds code to demonstrate how
     to manage user-owned record buffers. */
#if (MEMORY_OWNER_USER != 0)
  struct ADQRecord *record_buffers[NOF_RECORD_BUFFERS * ADQ_MAX_NOF_CHANNELS] = {NULL};
  printf("Allocating memory for user-owned record buffers.\n");
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    for (int i = 0; i < NOF_RECORD_BUFFERS; ++i)
    {
      struct ADQRecord *record = malloc(sizeof(struct ADQRecord));
      if (record == NULL)
      {
        printf("Failed to allocate memory for user-owned record buffers.\n");
        goto exit;
      }
      record->data = malloc(USER_RECORD_BUFFER_SIZE);
      record->header = malloc(sizeof(struct ADQRecordHeader));
      record->size = USER_RECORD_BUFFER_SIZE;
      if (record->data == NULL || record->header == NULL)
      {
        printf("Failed to allocate memory for user-owned record buffers.\n");
        free(record->data);
        free(record->header);
        goto exit;
      }

      record_buffers[ch * NOF_RECORD_BUFFERS + i] = record;
      result = ADQ_ReturnRecordBuffer(adq_cu, adq_num, ch, record);
      if (result != ADQ_EOK)
      {
        printf("Failed to register a record buffer for channel %c, code %d.\n", label(ch), result);
        goto exit;
      }
    }
  }
#endif

  /* Start the data acquisition. */
  printf("Start acquiring data... ");
  result = ADQ_StartDataAcquisition(adq_cu, adq_num);
  if (result != ADQ_EOK)
  {
    printf("failed, code %d.\n", result);
    goto exit;
  }
  printf("success.\n");

  /* Send software triggers. This structure sends more triggers than required
     since all channels share a common source. */
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    if (acquisition.channel[ch].trigger_source == ADQ_EVENT_SOURCE_SOFTWARE)
    {
      if (acquisition.channel[ch].nof_records != 0)
        printf("Generating software events on channel %c.\n", label(ch));
      for (int i = 0; i < acquisition.channel[ch].nof_records; ++i)
        ADQ_SWTrig(adq_cu, adq_num);
    }
  }

  /* Data readout loop. */
  bool done = false;
  int nof_received_records[ADQ_MAX_NOF_CHANNELS] = {0};
  while (!done && !abort_acquisition)
  {
    /* Wait for a record buffer on any channel. */
    struct ADQDataReadoutStatus status = {0};
    struct ADQRecord *record = NULL;
    int channel = ADQ_ANY_CHANNEL;
    int64_t bytes_received = ADQ_WaitForRecordBuffer(adq_cu, adq_num, &channel, (void **)(&record),
                                                     WAIT_TIMEOUT_MS, &status);
    /* Negative values are errors. Zero bytes received indicates a successful
       call, but only that only the status parameter can be read. */
    if (bytes_received == 0)
    {
      printf("Status event from channel %c, flags: 0x%08X.\n", label(channel), status.flags);
      if (status.flags & ADQ_DATA_READOUT_STATUS_FLAGS_INCOMPLETE)
        printf("  ADQ_DATA_READOUT_STATUS_FLAGS_INCOMPLETE\n");
      if (status.flags & ADQ_DATA_READOUT_STATUS_FLAGS_STARVING)
        printf("  ADQ_DATA_READOUT_STATUS_FLAGS_STARVING\n");
      printf("\n");
      continue;
    }
    else if (bytes_received < 0)
    {
      if (bytes_received == ADQ_EAGAIN)
      {
        /* Flushing on a timeout is not a requirement. This call aims to ensure
           that we transfer all the data from the digitizer when it's no longer
           acquiring records. */
        printf("Timeout, initiating a flush.\n");
        ADQ_FlushDMA(adq_cu, adq_num);
        continue;
      }
      printf("Waiting for a record buffer failed, code '%" PRId64 "'.\n", bytes_received);
      goto stop_then_exit;
    }

    /* Process the data. */
    process_data(record, &status, channel, bytes_received, pid);

    /* Return the buffer to the API. */
    result = ADQ_ReturnRecordBuffer(adq_cu, adq_num, channel, record);
    if (result != ADQ_EOK)
    {
      printf("Failed to return a record buffer, code %d.\n", result);
      break;
    }

    /* Check if the acquisition should end. We only increment the counter if the */
    if (!(status.flags & ADQ_DATA_READOUT_STATUS_FLAGS_INCOMPLETE))
      ++nof_received_records[channel];

    done = true;
    for (int ch = 0; ch < nof_channels; ++ch)
    {
      if (NOF_RECORDS[ch] != nof_received_records[ch])
        done = false;
    }
  }

stop_then_exit:
  /* Stop the data acquisition process. */
  printf("Stop acquiring data... ");
  result = ADQ_StopDataAcquisition(adq_cu, adq_num);
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

  int overflow_status = ADQ_GetStreamOverflow(adq_cu, adq_num);
  if (overflow_status != 0)
    printf("The device reports an overflow condition.\n");

exit:
#if (MEMORY_OWNER_USER != 0)
  printf("Returning memory for user-owned record buffers.\n");
  for (i = 0; i < sizeof(record_buffers) / sizeof(record_buffers[0]); ++i)
  {
    if (record_buffers[i] == NULL)
      continue;

    free(record_buffers[i]->data);
    free(record_buffers[i]->header);
  }
#endif
  return;
}

static int configure(void *adq_cu, int adq_num, enum ADQProductID_Enum pid)
{
  /* Read out static configuration. */
  int nof_channels = (int)ADQ_GetNofChannels(adq_cu, adq_num);
  if (nof_channels < 1)
  {
    printf("Read invalid number of channels %d.\n", nof_channels);
    return -1;
  }

  /* TRIG port */
  printf("Configuring the threshold for the TRIG port...");
  if (ADQ_HasVariableTrigThreshold(adq_cu, adq_num, 1))
  {
    if (pid == PID_ADQ14)
    {
      if (!ADQ_SetExtTrigThreshold(adq_cu, adq_num, 1, TRIG_THRESHOLD))
      {
        printf("failed.\n");
        return -1;
      }
    }
    else
    {
      if (!ADQ_SetTriggerThresholdVoltage(adq_cu, adq_num, ADQ_EVENT_SOURCE_TRIG, TRIG_THRESHOLD))
      {
        printf("failed.\n");
        return -1;
      }
    }
    printf("success.\n");
  }

  /* DC offset */
  if (ADQ_HasAdjustableBias(adq_cu, adq_num))
  {
    for (int ch = 0; ch < nof_channels; ++ch)
    {
      printf("Configuring DC offset for channel %c...", label(ch));
      if (!ADQ_SetAdjustableBias(adq_cu, adq_num, ch + 1, DC_OFFSET[ch]))
      {
        printf("failed.\n");
        return -1;
      }
      printf("success.\n");
    }
  }

  /* Digital baseline stabilization (DBS) */
  unsigned int nof_dbs_instances;
  if (!ADQ_GetNofDBSInstances(adq_cu, adq_num, &nof_dbs_instances))
  {
    printf("Failed to get the number of DBS instances.\n");
    return -1;
  }

  int nof_dbs_instances_per_channel = (int)nof_dbs_instances / nof_channels;
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    for (int i = 0; i < nof_dbs_instances_per_channel; ++i)
    {
      unsigned char instance = (unsigned char)(ch * nof_dbs_instances_per_channel + i);
      printf("Configuring DBS instance %d (channel %c)...", instance, label(ch));
      if (!ADQ_SetupDBS(adq_cu, adq_num, instance, DBS_BYPASS[ch], DBS_LEVEL[ch], 0, 0))
      {
        printf("failed.\n");
        return -1;
      }
      printf("success.\n");
    }
  }
  Sleep(1000);

  /* Periodic event generator. */
  printf("Configuring the periodic event source...");
  if (!ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, PERIODIC_EVENT_SOURCE_PERIOD))
  {
    printf("failed.\n");
    return -1;
  }
  printf("success.\n");

  /* Sample skip */
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    printf("Configuring sample skip for channel %c...", label(ch));
    if (!ADQ_SetChannelSampleSkip(adq_cu, adq_num, ch + 1, SAMPLE_SKIP_FACTOR[ch]))
    {
      printf("failed.\n");
      return -1;
    }
    printf("success.\n");
  }

  /* Initialize data acquisition parameters. */
  struct ADQDataAcquisitionParameters acquisition;
  int result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_ACQUISITION,
                                        &acquisition);
  if (result != sizeof(acquisition))
  {
    printf("Failed to initialize data acquisition parameters.\n");
    return -1;
  }

  /* Initialize data transfer parameters. */
  struct ADQDataTransferParameters transfer;
  result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_TRANSFER, &transfer);
  if (result != sizeof(transfer))
  {
    printf("Failed to initialize data transfer parameters.\n");
    return -1;
  }

  /* Initialize data readout parameters. */
  struct ADQDataReadoutParameters readout = {0};
  result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_READOUT, &readout);
  if (result != sizeof(readout))
  {
    printf("Failed to initialize data readout parameters.\n");
    return -1;
  }

  /* Adjust the transfer buffer size if needed. These are the default values: */
  transfer.channel[0].nof_buffers = 8;
  if (pid == PID_ADQ8)
    transfer.channel[0].record_buffer_size = 64 * 9216;
  else
    transfer.channel[0].record_buffer_size = 512 * 1024;

  /* Configure the acquisition parameters. */
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    acquisition.channel[ch].horizontal_offset = HORIZONTAL_OFFSET[ch];
    acquisition.channel[ch].trigger_edge = TRIGGER_EDGE[ch];
    acquisition.channel[ch].trigger_source = TRIGGER_SOURCE[ch];
    acquisition.channel[ch].record_length = RECORD_LENGTH[ch];
    acquisition.channel[ch].nof_records = NOF_RECORDS[ch];
  }

  /* Configure the readout parameters. */
#if (MEMORY_OWNER_USER != 0)
  readout.common.memory_owner = ADQ_MEMORY_OWNER_USER;
#endif
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    if ((INCOMPLETE_RECORDS != 0)
        || (acquisition.channel[ch].record_length == ADQ_INFINITE_RECORD_LENGTH))
      readout.channel[ch].incomplete_records_enabled = 1;
  }

  /* Write the parameters to the digitizer. */
  printf("Configuring data acquisition... ");
  result = ADQ_SetParameters(adq_cu, adq_num, &acquisition);
  if (result != sizeof(acquisition))
  {
    printf("failed, code %d. See the log file for more information.\n", result);
    return -1;
  }
  printf("success.\n");

  printf("Configuring data transfer... ");
  result = ADQ_SetParameters(adq_cu, adq_num, &transfer);
  if (result != sizeof(transfer))
  {
    printf("failed, code %d. See the log file for more information.\n", result);
    return -1;
  }
  printf("success.\n");

  printf("Configuring data readout... ");
  result = ADQ_SetParameters(adq_cu, adq_num, &readout);
  if (result != sizeof(readout))
  {
    printf("failed, code %d. See the log file for more information.\n", result);
    return -1;
  }
  printf("success.\n");

  printf("Configuring the level trigger... ");
  if (!ADQ_SetupLevelTrigger(adq_cu, adq_num, LT_LEVEL, (const int *)TRIGGER_EDGE, LT_RESET_LEVEL,
                             0, 1))
  {
    printf("failed.\n");
    return -1;
  }
  printf("success.\n");

  return 0;
}

int main()
{
  /* Connect handler for CTRL+C interrupts. */
  signal(SIGINT, sigint_handler);

  int revision = ADQAPI_GetRevision();
  printf("Streaming example for ADQ14, ADQ7 and ADQ8.\n");
  printf("API Revision: %6d\n", revision);

  /* Initialize the a handle to the ADQ control unit object. */
  void *adq_cu = CreateADQControlUnit();
  if (!adq_cu)
  {
    printf("Failed to create the ADQ control unit.\n");
    return -1;
  }

  /* Enable the error trace log. */
  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");

  /* List the available devices connected to the host computer. */
  struct ADQInfoListEntry *adq_list = NULL;
  unsigned int nof_devices = 0;
  if (!ADQControlUnit_ListDevices(adq_cu, &adq_list, &nof_devices))
  {
    printf("ListDevices failed.\n");
    goto exit;
  }

  if (nof_devices == 0)
  {
    printf("No devices found.\n");
    goto exit;
  }

  for (unsigned int i = 0; i < nof_devices; ++i)
  {
    printf("Entry #%u - ", i);
    switch (adq_list[i].ProductID)
    {
    case PID_ADQ14:
      printf("ADQ14");
      break;
    case PID_ADQ7:
      printf("ADQ7");
      break;
    case PID_ADQ8:
      printf("ADQ8");
      break;
    default:
      printf("Unsupported");
      break;
    }

    switch (adq_list[i].HWIFType)
    {
    case HWIF_PCIE:
      printf("-PCIe/PXIe\n");
      break;
    case HWIF_USB3:
      printf("-USB\n");
      break;
    case HWIF_ETH_ADQ14:
    case HWIF_ETH_ADQ7:
      printf("-10GbE\n");
      break;
    default:
      printf("\n");
      break;
    }
  }

  int device_to_open = 0;
  if (nof_devices > 1)
  {
    for (;;)
    {
      printf("\nTarget device: ");
      scanf("%d", &device_to_open);

      if ((device_to_open < 0) || (device_to_open >= (int)(nof_devices)))
      {
        printf("Invalid device '%d', valid range is [0, %u].\n", device_to_open, nof_devices - 1);
      }
      else
      {
        break;
      }
    }
  }

  printf("Initializing device #%d...", device_to_open);
  if (ADQControlUnit_SetupDevice(adq_cu, device_to_open))
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Device ids for the ADQ_* functions start at 1, representing the first
     device targeted by ADQControlUnit_SetupDevice(). */
  int adq_num = 1;
  enum ADQProductID_Enum product_id = adq_list[device_to_open].ProductID;
  int result = configure(adq_cu, adq_num, product_id);
  if (result != 0)
  {
    printf("Configuration failed.\n");
    goto exit;
  }

  /* Configuration and acquisition is handled by the function 'streaming()'.  */
  for (;;)
  {
    streaming(adq_cu, adq_num, product_id);
    if (abort_acquisition)
      break;

    printf("Press 0 followed by ENTER to exit, press any other character then ENTER to repeat the "
           "acquisition.\n");
    char c = 0;
    scanf(" %c", &c);
    if (c == '0')
      break;
  }

exit:
  /* Delete the control unit object and the memory allocated by this application. */
  DeleteADQControlUnit(adq_cu);
  printf("Exiting the application.\n");
  fflush(stdout);
  return 0;
}
