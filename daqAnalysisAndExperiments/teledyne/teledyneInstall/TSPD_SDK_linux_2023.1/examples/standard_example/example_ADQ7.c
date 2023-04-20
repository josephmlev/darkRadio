// File: ADQ_simple_example.cpp
// Description: A simple example of how to use the ADQAPI.
// This example is generic for all SP Devices Data Acquisition Boards (ADQs)
// The example sets some basic settings and collects data.

#define _CRT_SECURE_NO_WARNINGS // This define removes warnings for printf

#include "ADQAPI.h"
#include "os.h"
#include <stdio.h>
#include <time.h>
//#define VERBOSE

#ifdef LINUX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
#endif

void adq7_raw_streaming(void *adq_cu, int adq_num, unsigned int use_binary);
void adq7_triggered_streaming_offline(void *adq_cu, int adq_num);
void adq7_multirecord(void *adq_cu, int adq_num);
void adq7_read_binary_packet_stream(void);
int handle_headers_and_data(unsigned int *header_status, const unsigned int *headers_added,
                            unsigned int *headers_done, unsigned int *records_completed,
                            unsigned int *samples_remaining, const unsigned int *samples_added,
                            unsigned int *samples_extradata, short **target_buffers,
                            short **target_buffers_extradata, size_t *buffer_size,
                            struct ADQRecordHeader **target_headers, unsigned int write_to_file,
                            FILE **outfile_data, FILE **outfile_headers, unsigned int channelsmask);

#define CHECKADQ(f)              \
  if (!(f))                      \
  {                              \
    printf("Error in " #f "\n"); \
    goto error;                  \
  }

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void adq7(void *adq_cu, int adq_num)
{
  int mode;
  char *serialnumber;

  int *revision;

  unsigned int tlocal;
  unsigned int tr1;
  unsigned int tr2;
  unsigned int tr3;
  unsigned int tr4;

  double fs;
  unsigned int nofchannels;

  if (adq_cu)
  {
    serialnumber = ADQ_GetBoardSerialNumber(adq_cu, adq_num);

    revision = ADQ_GetRevision(adq_cu, adq_num);

    tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0) / 256;
    tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1) / 256;
    tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2) / 256;
    tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3) / 256;
    tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4) / 256;

    fs = 0.0;
    nofchannels = ADQ_GetNofChannels(adq_cu, adq_num);

    ADQ_GetSampleRate(adq_cu, adq_num, 0, &fs);

    printf("\nConnected to ADQ7 #1\n\n");

    printf("Device Serial Number: %s\n", serialnumber);
    printf("Firmware Revision: %d\n", revision[0]);
    printf("%u channels, %.2f GSPs\n", nofchannels, fs / 1000 / 1000 / 1000);
    printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
           tlocal, tr1, tr2, tr3, tr4);

    // Checking for in-compatible firmware
    if (ADQ_HasFeature(adq_cu, adq_num, "FWATD") == 1)
    {
      printf("ERROR: This device is loaded with FWATD firmware and cannot be used for this "
             "example. Please see FWATD examples.\n");
      return;
    }
    if (ADQ_HasFeature(adq_cu, adq_num, "FWPD") == 1)
    {
      printf("ERROR: This device is loaded with FWPD firmware and cannot be used for this example. "
             "Please see FWPD examples.\n");
      return;
    }
  }

  mode = 0;

  while (1)
  {
    printf("\nSelect:\n"
           " 0 = Exit\n");
    if (adq_cu)
      printf(" 1 = Triggered streaming\n"
             " 2 = Continuous streaming\n"
             " 3 = Raw streaming (text)\n"
             " 4 = Multirecord\n"
             " 5 = Raw streaming (binary)\n"
             " 6 = Triggered streaming to disk for offline parsing\n");
    printf(" 7 = Offline parsing of stream from disk\n\n");

    scanf("%d", &mode);

    switch (mode)
    {
    case 1:
      printf("This example is deprecated for the triggered streaming acquisition mode. Refer to "
             "the example 'gen3_streaming' for an updated reference implementation and to the user "
             "guide (20-2465) for its documentation.\n");
      break;
    case 2:
      printf("This example is deprecated for the continuous streaming acquisition mode. Refer to "
             "the example 'gen3_streaming' for an updated reference implementation and to the user "
             "guide (20-2465) for its documentation.\n");
      break;
    case 3:
      if (adq_cu)
        adq7_raw_streaming(adq_cu, adq_num, 0);
      break;
    case 4:
      if (adq_cu)
        adq7_multirecord(adq_cu, adq_num);
      break;
    case 5:
      if (adq_cu)
        adq7_raw_streaming(adq_cu, adq_num, 1);
      break;
    case 6:
      if (adq_cu)
        adq7_triggered_streaming_offline(adq_cu, adq_num);
      break;
    case 7:
      adq7_read_binary_packet_stream();
      break;
    default:
      return;
    }
  }
}

void adq7_raw_streaming(void *adq_cu, int adq_num, unsigned int use_binary)
{
  unsigned int n_samples_collect = 8 * 64 * 1024;
  unsigned int buffers_filled;
  int collect_result;
  unsigned int samples_to_collect;
  signed short *data_stream_target;
  unsigned int LoopVar;
  FILE *outfile = NULL, *outfileBin = NULL;

  unsigned int en_A = 1;
  unsigned int en_B = 0;

  unsigned int sample_skip = 8;

  int exit = 0;
  unsigned int test_pattern = 0; // 0 or 2
  unsigned int timeout = 0;

  if (use_binary)
    outfile = fopen("data_raw.bin", "wb");
  else
    outfile = fopen("data_raw.out", "w");

  if (outfile == NULL)
  {
    printf("Error: Failed to open output file.\n");
    return;
  }
  printf("\nSetting up streaming...");

  // Enable streaming
  CHECKADQ(ADQ_SetSampleSkip(adq_cu, adq_num, sample_skip));

  CHECKADQ(ADQ_SetTestPatternMode(adq_cu, adq_num, test_pattern));
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 2, 1)); // RAW mode
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 3, 1 * en_A + 2 * en_B)); // mask
  CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, ADQ_SW_TRIGGER_MODE));

  printf("Collecting data, please wait...\n");

  // Created temporary target for streaming data
  data_stream_target = NULL;

  // Allocate temporary buffer for streaming data
  CHECKADQ(data_stream_target = (signed short *)malloc(n_samples_collect * sizeof(signed short)));

  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));
  CHECKADQ(ADQ_StartStreaming(adq_cu, adq_num));

  samples_to_collect = n_samples_collect;

  while (samples_to_collect > 0)
  {
    unsigned int samples_in_buffer;
    timeout = 0;
    do
    {
      collect_result = ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled);
      printf("Filled: %2u\n", buffers_filled);
      if (timeout > 100)
        goto error;
      timeout++;

    } while ((buffers_filled == 0) && (collect_result));
    collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
    samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Warning: Streaming Overflow!\n");
      collect_result = 0;
    }

    if (collect_result)
    {
      // Buffer all data in RAM before writing to disk, if streaming to disk is need a high
      // performance procedure could be implemented here. Data format is set to 16 bits, so buffer
      // size is Samples*2 bytes
      memcpy((void *)&data_stream_target[n_samples_collect - samples_to_collect],
             ADQ_GetPtrStream(adq_cu, adq_num), samples_in_buffer * sizeof(signed short));
      samples_to_collect -= samples_in_buffer;
    }
    else
    {
      printf("Collect next data page failed!\n");
      samples_to_collect = 0;
    }
  }

  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));

  // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
  printf("Writing stream data in RAM to disk.\n");

  samples_to_collect = n_samples_collect;

  if (use_binary)
  {
    fwrite(data_stream_target, samples_to_collect, sizeof(signed short), outfile);
  }
  else
  {
    for (LoopVar = 0; LoopVar < samples_to_collect; LoopVar += 1)
    {
      // fprintf(outfile, "%04hx\n", data_stream_target[LoopVar]);
      fprintf(outfile, "%d\n", data_stream_target[LoopVar]);
    }
  }

  printf("\n\n Done. Samples stored.\n");

error:

  if (NULL != outfile)
    fclose(outfile);
  if (NULL != outfileBin)
    fclose(outfileBin);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

void adq7_triggered_streaming_file_writer(unsigned int write_to_file, short *data_buffer,
                                          unsigned int samples_to_write, FILE *file)
{
  unsigned int i;

  switch (write_to_file)
  {
  case 1: // Write ASCII
    for (i = 0; i < samples_to_write; ++i)
      fprintf(file, "%d\n", data_buffer[i]);
    break;
  case 2: // Write binary
    fwrite(data_buffer, sizeof(short), samples_to_write, file);
    break;
  default:
    break;
  }
}

void adq7_triggered_streaming_offline(void *adq_cu, int adq_num)
{
  // Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  unsigned int samples_per_record;
  unsigned int pretrig_samples;
  unsigned int holdoff_samples;
  unsigned int success;
  unsigned int nof_records = 0;
  const unsigned int records_completed[4] = {0, 0, 0, 0};
  unsigned char channelsmask;
  unsigned int maskinput;
  unsigned int trig_channel;
  unsigned int tr_buf_size = 256 * 1024;
  unsigned int tr_buf_no = 8;
  unsigned int trig_freq = 0;
  unsigned int trig_period = 0;
  unsigned int timeout_ms = 1000;
  unsigned int write_to_file = 2;

  unsigned int nof_records_sum = 0;
  unsigned int nof_received_records_sum = 0;
  unsigned int received_all_records = 0;

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0) / 256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1) / 256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2) / 256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3) / 256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4) / 256;
  char *serialnumber;
  unsigned int nof_channels;
  int exit = 0;
  unsigned int ch = 0;
  unsigned int i;
  unsigned int buffers_filled = 0;
  unsigned int flush_performed = 0;
  unsigned int retval;

  // Bias ADC codes
  int adjustable_bias = 0;

  // DBS settings
  unsigned int dbs_nof_inst = 0;
  unsigned char dbs_inst = 0;
  int dbs_bypass = 1;
  int dbs_dc_target = adjustable_bias;
  int dbs_lower_saturation_level = 0;
  int dbs_upper_saturation_level = 0;

  FILE *outfile_data[4] = {NULL, NULL, NULL, NULL};
  FILE *outfile_headers[4] = {NULL, NULL, NULL, NULL};
  char outfile_mode[8];
  char filename[256];

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
         tlocal, tr1, tr2, tr3, tr4);

  serialnumber = ADQ_GetBoardSerialNumber(adq_cu, adq_num);
  nof_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  printf("Device Serial Number: %s\n", serialnumber);

  // Setup adjustable bias
  if (ADQ_HasAdjustableBias(adq_cu, adq_num))
  {
    for (ch = 0; ch < nof_channels; ++ch)
    {
      success = ADQ_SetAdjustableBias(adq_cu, adq_num, ch + 1, adjustable_bias);
      if (success == 0)
        printf("Failed setting adjustable bias for channel %c.\n", "ABCD"[ch]);
      else
        printf("Adjustable bias for channel %c set to %d codes.\n", "ABCD"[ch], adjustable_bias);
    }

    printf("Waiting for bias settling...\n");
    Sleep(1000);
  }

  // Setup DBS
  ADQ_GetNofDBSInstances(adq_cu, adq_num, &dbs_nof_inst);
  for (dbs_inst = 0; dbs_inst < dbs_nof_inst; ++dbs_inst)
  {
    printf("Setting up DBS instance %u ...\n", dbs_inst);
    success = ADQ_SetupDBS(adq_cu, adq_num, dbs_inst, dbs_bypass, dbs_dc_target,
                           dbs_lower_saturation_level, dbs_upper_saturation_level);
    if (success == 0)
      printf("Failed setting up DBS instance %d.", dbs_inst);
  }
  Sleep(1000);

  printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level "
         "Trigger Mode\n %d = Internal trigger mode\n",
         ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE,
         ADQ_INTERNAL_TRIGGER_MODE);
  scanf("%d", &trig_mode);

  switch (trig_mode)
  {
  case ADQ_SW_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
  }
  case ADQ_EXT_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
  }
  case ADQ_LEVEL_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n -32768 <= level <= 32767\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    if (nof_channels > 1)
    {
      printf("\nChoose level trig channel.\n A->%c: 1->%u\n", "ABCD"[nof_channels - 1],
             nof_channels);
      scanf("%u", &trig_channel);
    }
    else
    {
      trig_channel = 1;
    }
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    printf("\n");
    break;
  }
  case ADQ_INTERNAL_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trigger frequency in Hz\n");
    scanf("%u", &trig_freq);
    trig_period = 1000000000 / trig_freq;
    CHECKADQ(ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, trig_period));
    printf("\n");
    break;
  }
  default:
    return;
    break;
  }

  printf("\nChoose number of records (-1 for infinite mode).\n");
  scanf("%u", &nof_records);

  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

  printf("\nChoose number of milliseconds for exit timeout (e.g. 1000 ms).\n");
  scanf("%u", &timeout_ms);

  channelsmask = 0x00;
  printf("\nEnable channel A data collection? (0 or 1)\n");
  scanf("%u", &maskinput);
  if (maskinput > 0)
    channelsmask |= 0x01;

  if (nof_channels > 1)
  {
    printf("\nEnable channel B data collection? (0 or 1)\n");
    scanf("%u", &maskinput);
    if (maskinput > 0)
      channelsmask |= 0x02;
  }

  // Compute the sum of the number of records specified by the user
  for (ch = 0; ch < 4; ++ch)
  {
    if (!((1 << ch) & channelsmask))
      continue;
    nof_records_sum += nof_records;
  }

  // Open output files for data
  switch (write_to_file)
  {
  case 1:
    // Save in ASCII
    sprintf(outfile_mode, "w");
    break;
  case 2:
    // Save in binary
    sprintf(outfile_mode, "wb");
    break;
  default:
    break;
  }

  {
    unsigned int structsize;
    char *buffer;
    FILE *fp;
    const char *ds_filename = "./adq_device_struct.bin";
    if (ADQ_GetADQDataDeviceStructSize(adq_cu, adq_num, &structsize))
    {
      fp = fopen(ds_filename, "wb");
      if (!fp)
      {
        printf("Couldn't open %s.\n", ds_filename);
        goto error;
      }
      buffer = (char *)malloc(structsize);
      if (ADQ_GetADQDataDeviceStruct(adq_cu, adq_num, (void *)buffer))
      {
        fwrite(buffer, structsize, 1, fp);
        fclose(fp);
        free(buffer);
      }
      else
      {
        printf("Failed to get ADQData device struct.\n");
        free(buffer);
        goto error;
      }
    }
    else
    {
      printf("Failed to get ADQData device struct size.\n");
      goto error;
    }
  }
  sprintf(filename, "data_packet_stream.out");
  outfile_data[0] = fopen(filename, outfile_mode);

  pretrig_samples = 0;
  holdoff_samples = 0;

  // Use triggered streaming for data collection.
  CHECKADQ(ADQ_TriggeredStreamingSetup(adq_cu, adq_num, nof_records, samples_per_record,
                                       pretrig_samples, holdoff_samples, channelsmask));

  // Commands to start the triggered streaming mode after setup
  CHECKADQ(ADQ_ResetWriteCountMax(adq_cu, adq_num));
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, tr_buf_no, tr_buf_size));
  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));
  CHECKADQ(ADQ_StartStreaming(adq_cu, adq_num));
  // When StartStreaming is issued device is armed and ready to accept triggers

  if (trig_mode == ADQ_SW_TRIGGER_MODE)
  {
    // Send software triggers
    for (i = 0; i < nof_records; ++i)
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
  }

  // Collection loop
  do
  {
    buffers_filled = 0;
    success = 1;

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Streaming overflow detected...\n");
      goto error;
    }
    // Wait for one or more transfer buffers
    while (!buffers_filled)
    {
#ifdef POLLING_FOR_DATA
      for (i = 0; i < timeout_ms; i++)
      {
        CHECKADQ(ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled));
        if (buffers_filled)
        {
          break;
        }
        Sleep(1);
      }
#else
      CHECKADQ(ADQ_WaitForTransferBuffer(adq_cu, adq_num, &buffers_filled, timeout_ms));
#endif
      if (!buffers_filled)
      {
        if (flush_performed)
          break;
        printf("Timeout, flushing DMA...\n");
        CHECKADQ(ADQ_FlushDMA(adq_cu, adq_num));
        flush_performed = 1;
      }
    }

    if (buffers_filled)
    {
      ADQ_CollectDataNextPage(adq_cu, adq_num);
      if (ADQ_GetStreamOverflow(adq_cu, adq_num))
        printf("Warning: Streaming Overflow!\n");
      fwrite(ADQ_GetPtrStream(adq_cu, adq_num), sizeof(signed short),
             ADQ_GetSamplesPerPage(adq_cu, adq_num), outfile_data[0]);
    }
    else
    {
      printf("\n");
      break;
    }

    // Update received_all_records
    nof_received_records_sum = 0;
    for (ch = 0; ch < 4; ++ch)
      nof_received_records_sum += records_completed[ch];

    // Determine if collection is completed
    received_all_records = (nof_received_records_sum >= nof_records_sum);
  } while (!received_all_records);

  CHECKADQ(ADQ_GetWriteCountMax(adq_cu, adq_num, &retval));
  printf(" Peak: %.3f MiB\n", (double)((unsigned int)retval * 128) / (double)(1024 * 1024));

  if (success)
    printf("Done.\n");
  else
    printf("Error occurred.\n");

error:
  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));

  // Close any open output files
  if (write_to_file > 0)
  {
    for (ch = 0; ch < 4; ch++)
    {
      if (!((1 << ch) & channelsmask))
        continue;

      if (outfile_data[ch])
        fclose(outfile_data[ch]);
      if (outfile_headers[ch])
        fclose(outfile_headers[ch]);
    }
  }

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);
  return;
}

void adq7_multirecord(void *adq_cu, int adq_num)
{

  // Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  unsigned int samples_per_record;
  unsigned int number_of_records;
  unsigned int buffersize;
  unsigned int channel;
  unsigned char channelsmask;
  // unsigned int maskinput;
  unsigned int trig_channel;
  unsigned int write_to_file = 1;
  unsigned int records_to_collect;

  short *buf_a = NULL;
  short *buf_b = NULL;
  void *target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused
                           // pointers should be null pointers

  FILE *outfile[2] = {NULL, NULL};
  int exit = 0;

  printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level "
         "Trigger Mode\n",
         ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE);
  scanf("%d", &trig_mode);

  switch (trig_mode)
  {
  case ADQ_SW_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
  }
  case ADQ_EXT_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
  }
  case ADQ_LEVEL_TRIGGER_MODE:
  {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n -32768 <= level <= 32767\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    scanf("%u", &trig_channel);
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    printf("\n");
    break;
  }
  default:
    return;
    break;
  }

  printf("\nChoose number of records.\n");
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

  channelsmask = 0x03;

  // Use only multirecord mode for data collection.
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num, number_of_records, samples_per_record));

  if (trig_mode == ADQ_SW_TRIGGER_MODE)
  {
    int trigged;
    printf("Issuing software trigger(s).\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    } while (trigged == 0);
  }
  else
  {
    int trigged;
    printf("\nPlease trigger your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
    } while (trigged == 0);
  }
  printf("\nAll records triggered\n");

  // collect data
  printf("\nSave collected data to file?\n 0 = Do not save\n 1 = Save in ascii format (slow)\n 2 = "
         "Save in binary format (fast)\n");
  scanf("%u", &write_to_file);

  records_to_collect = number_of_records;

  printf("Collecting data, please wait...\n");

  // Each data buffer must contain enough samples to store all the records consecutively
  buffersize = records_to_collect * samples_per_record;

  buf_a = (short *)calloc(buffersize, sizeof(short));
  buf_b = (short *)calloc(buffersize, sizeof(short));

  // Create a pointer array containing the data buffer pointers
  target_buffers[0] = (void *)buf_a;
  target_buffers[1] = (void *)buf_b;
  if (buf_a == NULL)
    goto error;
  if (buf_b == NULL)
    goto error;

  // Use the GetData function
  CHECKADQ(ADQ_GetData(adq_cu, adq_num, target_buffers, buffersize, sizeof(short), 0,
                       records_to_collect, channelsmask, 0, samples_per_record,
                       ADQ_TRANSFER_MODE_NORMAL));

  switch (write_to_file)
  {
  case 0: // Do not save
    break;
  case 1: // Save as ascii
    outfile[0] = fopen("dataA.out", "w");
    outfile[1] = fopen("dataB.out", "w");
    for (channel = 0; channel < 2; channel++)
    {
      if (outfile[channel] != NULL)
      {
        unsigned int j;
        for (j = 0; j < buffersize; j++)
        {
          if (channelsmask & (0x01 << channel))
            fprintf(outfile[channel], "%d\n", ((short *)target_buffers[channel])[j]);
        }
      }
      else
      {
        printf("Error: Failed to open output files.\n");
      }
    }
    break;
  case 2: // Save as binary
    outfile[0] = fopen("dataA.out", "wb");
    outfile[1] = fopen("dataB.out", "wb");
    for (channel = 0; channel < 2; channel++)
    {
      if (outfile[channel] != NULL)
      {
        if (channelsmask & (0x01 << channel))
          fwrite((short *)target_buffers[channel], sizeof(short), buffersize, outfile[channel]);
      }
      else
        printf("Error: Failed to open output files.\n");
    }
    break;
  default:
    printf("Error: Unknown format!\n");
    break;
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  if (write_to_file > 0)
    printf("\n\nDone. Samples stored in data.out.\n");

error:

  for (channel = 0; channel < 2; channel++)
  {
    if (outfile[channel] != NULL)
      fclose(outfile[channel]);
  }

  if (buf_a != NULL)
    free(buf_a);
  if (buf_b != NULL)
    free(buf_b);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

/* This function reads back a previously recorded binary
   file containing the raw binary data from the ADQ.*/
#define PACKET_BUFFER_SIZE 25 * 1024
void adq7_read_binary_packet_stream(void)
{
  const char *ds_filename = "adq_device_struct.bin";

  int ch;

  void *adq_device_struct = NULL;
  void *ref;
  long filesize;
  FILE *fp = NULL;
  char packets[PACKET_BUFFER_SIZE];

  FILE *outfile_data[4] = {NULL, NULL, NULL, NULL};
  FILE *outfile_headers[4] = {NULL, NULL, NULL, NULL};
  char outfile_mode[8];
  char filename[256];
  int write_to_file = 2; // Use binary mode
  int nof_channels = 4;

  // Buffers to handle the stream output (both headers and data)
  size_t buffer_size[8];
  short *target_buffers[8] = {NULL};
  short *target_buffers_extradata[8] = {NULL};
  struct ADQRecordHeader *target_headers[8] = {NULL};

  // Variables to handle the stream output (both headers and data)
  unsigned int records_completed[8] = {0};
  unsigned int header_status[8] = {0};
  unsigned int samples_added[8] = {0};
  unsigned int headers_added[8] = {0};
  unsigned int samples_extradata[8] = {0};
  unsigned int samples_remaining;
  unsigned int headers_done = 0;
  unsigned int channelsmask = 0x3;
  // Allocate memory for parsed data stream
  for (ch = 0; ch < 8; ch++)
  {
    target_buffers[ch] = (short int *)malloc(sizeof(short int) * (PACKET_BUFFER_SIZE));
    if (!target_buffers[ch])
    {
      printf("Failed to allocate memory for target_buffers\n");
      goto error;
    }
    target_headers[ch] = (struct ADQRecordHeader *)malloc(
      (PACKET_BUFFER_SIZE / ADQ7_STREAM_RECORD_MIN_BYTES) * sizeof(struct ADQRecordHeader));
    if (!target_headers[ch])
    {
      printf("Failed to allocate memory for target_headers\n");
      goto error;
    }
    target_buffers_extradata[ch] = (short int *)malloc(sizeof(short int) * (PACKET_BUFFER_SIZE));
    if (!target_buffers_extradata[ch])
    {
      printf("Failed to allocate memory for target_buffers_extradata\n");
      goto error;
    }
    buffer_size[ch] = PACKET_BUFFER_SIZE;
  }
  strcpy(outfile_mode, "wb");

  // Create ADQData reference for this packet stream
  CHECKADQ(ADQData_Create(&ref));
  // Enable error tracing for ADQData
  CHECKADQ(ADQData_EnableErrorTrace(ref, LOG_LEVEL_INFO, "ADQData.log", 0));

  // Read ADQ device struct from disk
  // Open device struct file
  fp = fopen(ds_filename, "rb");
  if (!fp)
  {
    printf("Couldn't open %s.\n", ds_filename);
    goto error;
  }
  // Get file size
  fseek(fp, 0L, SEEK_END);
  filesize = ftell(fp);
  rewind(fp);
  adq_device_struct = malloc(filesize);
  fread(adq_device_struct, filesize, 1, fp);
  fclose(fp);
  fp = NULL;

  // Open output files for headers
  if (write_to_file > 0)
  {
    for (ch = 0; ch < nof_channels; ++ch)
    {
      if (!((1 << ch) & channelsmask))
        continue;

      sprintf(filename, "data%c.out", "ABCDEFGH"[ch]);
      outfile_data[ch] = fopen(filename, outfile_mode);
      if (!outfile_data[ch])
      {
        printf("Failed to open output file %s.\n", filename);
        goto error;
      }

      sprintf(filename, "headers%c.out", "ABCDEFGH"[ch]);
      outfile_headers[ch] = fopen(filename, "wb");
      if (!outfile_headers[ch])
      {
        printf("Failed to open output file %s.\n", filename);
        goto error;
      }
    }
  }

  // Initialize packet stream with metadata from ADQ device struct
  CHECKADQ(ADQData_InitPacketStream(ref, adq_device_struct, NULL));

  // Open packet stream stored to disk
  fp = fopen("data_packet_stream.out", "rb");
  if (!fp)
  {
    printf("Couldn't open %s.\n", filename);
    goto error;
  }

  // Get file size
  fseek(fp, 0L, SEEK_END);
  filesize = ftell(fp);
  rewind(fp);
  // Read packets from disk, needs to be done in multiples of ADQ7_STREAM_CHUNK_BYTES bytes
  while (filesize >= PACKET_BUFFER_SIZE)
  {
    fread(packets, PACKET_BUFFER_SIZE, 1, fp);
    filesize -= PACKET_BUFFER_SIZE;
    // Parse packets
    CHECKADQ(ADQData_ParsePacketStream(ref, // ref,
                                       (void *)packets, // raw_data_buffer
                                       PACKET_BUFFER_SIZE, // raw_data_size,
                                       (void **)target_buffers, // target_buffers,
                                       (void **)target_headers, // target_headers,
                                       samples_added, // samples_added,
                                       headers_added, // headers_added,
                                       header_status, // header_status,
                                       0xf)); // channels_mask
    handle_headers_and_data(header_status, headers_added, &headers_done, records_completed,
                            &samples_remaining, samples_added, samples_extradata, target_buffers,
                            target_buffers_extradata, buffer_size, target_headers, write_to_file,
                            outfile_data, outfile_headers, channelsmask);
  }
  while (filesize >= ADQ7_STREAM_CHUNK_BYTES)
  {
    // Read remaining packets from disk, needs to be done in multiples of ADQ7_STREAM_CHUNK_BYTES
    // bytes
    fread(packets, ADQ7_STREAM_CHUNK_BYTES, 1, fp);
    filesize -= ADQ7_STREAM_CHUNK_BYTES;
    // Parse packets
    CHECKADQ(ADQData_ParsePacketStream(ref, // ref,
                                       (void *)packets, // raw_data_buffer
                                       ADQ7_STREAM_CHUNK_BYTES, // raw_data_size,
                                       (void **)target_buffers, // target_buffers,
                                       (void **)target_headers, // target_headers,
                                       samples_added, // samples_added,
                                       headers_added, // headers_added,
                                       header_status, // header_status,
                                       0xf)); // channels_mask
    handle_headers_and_data(header_status, headers_added, &headers_done, records_completed,
                            &samples_remaining, samples_added, samples_extradata, target_buffers,
                            target_buffers_extradata, buffer_size, target_headers, write_to_file,
                            outfile_data, outfile_headers, channelsmask);
  }

  for (ch = 0; ch < 8; ch++)
    if (records_completed[ch])
      printf("Parsed %u records for channel %c and stored to file.\n", records_completed[ch],
             "ABCDEFGH"[ch]);

  fclose(fp);
  fp = NULL;

  // Destroy reference for packet stream
  CHECKADQ(ADQData_Destroy(ref));

error:
  if (fp)
    fclose(fp);
  for (ch = 0; ch < 8; ch++)
  {
    if (target_buffers[ch])
      free(target_buffers[ch]);
    if (target_headers[ch])
      free(target_headers[ch]);
    if (target_buffers_extradata[ch])
      free(target_buffers_extradata[ch]);
  }
  // Close any open output files
  if (write_to_file > 0)
  {
    for (ch = 0; ch < 4; ch++)
    {
      if (!((1 << ch) & channelsmask))
        continue;

      if (outfile_data[ch])
        fclose(outfile_data[ch]);
      if (outfile_headers[ch])
        fclose(outfile_headers[ch]);
    }
  }
  if (adq_device_struct)
    free(adq_device_struct);
  return;
}

int handle_headers_and_data(unsigned int *header_status, const unsigned int *headers_added,
                            unsigned int *headers_done, unsigned int *records_completed,
                            unsigned int *samples_remaining, const unsigned int *samples_added,
                            unsigned int *samples_extradata, short **target_buffers,
                            short **target_buffers_extradata, size_t *buffer_size,
                            struct ADQRecordHeader **target_headers, unsigned int write_to_file,
                            FILE **outfile_data, FILE **outfile_headers, unsigned int channelsmask)
{
  int ch;
  unsigned int i;
  // Parse the data
  for (ch = 0; ch < 4; ++ch)
  {
    if (!((1 << ch) & channelsmask))
      continue;

    if (headers_added[ch] > 0)
    {
      if (header_status[ch])
        *headers_done = headers_added[ch];
      else
        // One incomplete record in the buffer,
        // header is copied to the front of the buffer later
        *headers_done = headers_added[ch] - 1;

      // If there is at least one complete header
      records_completed[ch] += *headers_done;
    }

    // Parse the added samples
    if (samples_added[ch] > 0)
    {
      *samples_remaining = samples_added[ch];
      // Handle incomplete record at the start of the buffer
      if (samples_extradata[ch] > 0)
      {
        if (*headers_done == 0)
        {
          // There is not enough data in the transfer buffer to complete
          // the record. Add all the samples to the extradata buffer.
          if (sizeof(short) * (samples_added[ch] + samples_extradata[ch]) > buffer_size[ch])
          {
#ifdef VERBOSE
            printf("Enlarging buffer size to %zu.\n",
                   sizeof(short) * (samples_added[ch] + samples_extradata[ch]));
#endif
            // Enlarge record buffer
            target_buffers_extradata[ch] = (short *)
              realloc(target_buffers_extradata[ch],
                      sizeof(short) * (samples_added[ch] + samples_extradata[ch]));
            if (!target_buffers_extradata[ch])
            {
              printf("Out of memory while trying to allocate buffer for record.\n");
              return 1;
            }
            buffer_size[ch] = sizeof(short) * (samples_added[ch] + samples_extradata[ch]);
          }
          memcpy(&(target_buffers_extradata[ch][samples_extradata[ch]]), target_buffers[ch],
                 sizeof(short) * samples_added[ch]);
          *samples_remaining -= samples_added[ch];
          samples_extradata[ch] += samples_added[ch];
        }
        else
        {
          // There is enough data in the transfer buffer to complete
          // the record. Add RecordLength-samples_extradata samples
          adq7_triggered_streaming_file_writer(write_to_file, target_buffers_extradata[ch],
                                               samples_extradata[ch], outfile_data[ch]);
          adq7_triggered_streaming_file_writer(write_to_file, target_buffers[ch],
                                               target_headers[ch][0].RecordLength
                                                 - samples_extradata[ch],
                                               outfile_data[ch]);
          if (write_to_file)
            fwrite(&target_headers[ch][0], sizeof(struct ADQRecordHeader), 1, outfile_headers[ch]);

          *samples_remaining -= target_headers[ch][0].RecordLength - samples_extradata[ch];
          samples_extradata[ch] = 0;
#ifdef VERBOSE
          printf("Completed record %u on channel %d, %u samples.\n",
                 target_headers[ch][0].RecordNumber, ch, target_headers[ch][0].RecordLength);
#endif
        }
      }
      else
      {
        if (*headers_done == 0)
        {
          // The samples in the transfer buffer begin a new record, this
          // record is incomplete.
          if (sizeof(short) * (samples_added[ch]) > buffer_size[ch])
          {
#ifdef VERBOSE
            printf("Enlarging buffer size to %zu.\n", sizeof(short) * (samples_added[ch]));
#endif
            // Enlarge record buffer
            target_buffers_extradata[ch] = (short *)realloc(target_buffers_extradata[ch],
                                                            sizeof(short) * (samples_added[ch]));
            if (!target_buffers_extradata[ch])
            {
              printf("Out of memory while trying to allocate buffer for record.\n");
              return 1;
            }
            buffer_size[ch] = sizeof(short) * (samples_added[ch]);
          }
          memcpy(target_buffers_extradata[ch], target_buffers[ch],
                 sizeof(short) * samples_added[ch]);
          *samples_remaining -= samples_added[ch];
          samples_extradata[ch] = samples_added[ch];
        }
        else
        {
          // The samples in the transfer buffer begin a new record, this
          // record is complete.
          adq7_triggered_streaming_file_writer(write_to_file, target_buffers[ch],
                                               target_headers[ch][0].RecordLength,
                                               outfile_data[ch]);
          if (write_to_file)
            fwrite(&target_headers[ch][0], sizeof(struct ADQRecordHeader), 1, outfile_headers[ch]);

          *samples_remaining -= target_headers[ch][0].RecordLength;

#ifdef VERBOSE
          printf("Completed record %u on channel %d, %u samples.\n",
                 target_headers[ch][0].RecordNumber, ch, target_headers[ch][0].RecordLength);
#endif
        }
      }
      // At this point: the first record in the transfer buffer or the entire
      // transfer buffer has been parsed.

      // Loop through complete records fully inside the buffer
      for (i = 1; i < *headers_done; ++i)
      {
        adq7_triggered_streaming_file_writer(
          write_to_file, (&target_buffers[ch][samples_added[ch] - (*samples_remaining)]),
          target_headers[ch][i].RecordLength, outfile_data[ch]);
        if (write_to_file)
          fwrite(&target_headers[ch][i], sizeof(struct ADQRecordHeader), 1, outfile_headers[ch]);

        *samples_remaining -= target_headers[ch][i].RecordLength;

#ifdef VERBOSE
        printf("Completed record %u on channel %d, %u samples.\n",
               target_headers[ch][i].RecordNumber, ch, target_headers[ch][i].RecordLength);
#endif
      }

      if (*samples_remaining > 0)
      {
        // There is an incomplete record at the end of the transfer buffer
        // Copy the incomplete header to the start of the target_headers buffer
        memcpy(target_headers[ch], &target_headers[ch][*headers_done],
               sizeof(struct ADQRecordHeader));

        // Copy any remaining samples to the target_buffers_extradata buffer,
        // they belong to the incomplete record
        if (sizeof(short) * (*samples_remaining) > buffer_size[ch])
        {
#ifdef VERBOSE
          printf("Enlarging buffer size to %zu.\n", sizeof(short) * (*samples_remaining));
#endif
          // Enlarge record buffer
          target_buffers_extradata[ch] = (short *)realloc(target_buffers_extradata[ch],
                                                          sizeof(short) * (*samples_remaining));
          if (!target_buffers_extradata[ch])
          {
            printf("Out of memory while trying to allocate buffer for record.\n");
            return 1;
          }
          buffer_size[ch] = sizeof(short) * (*samples_remaining);
        }
        memcpy(target_buffers_extradata[ch],
               &target_buffers[ch][samples_added[ch] - (*samples_remaining)],
               sizeof(short) * (*samples_remaining));
#ifdef VERBOSE
        printf("Incomplete at end of transfer buffer.\n");
        printf("Copying %u samples to the extradata buffer\n", *samples_remaining);
#endif
        samples_extradata[ch] = *samples_remaining;
        *samples_remaining = 0;
      }
    }
  }

  return 0;
}
