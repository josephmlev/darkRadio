/*
 *  Copyright 2019 Teledyne Signal Processing Devices Sweden AB
 *
 *  ADQ14-FWATD example
 */

#ifdef LINUX
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
extern int _kbhit();
#define sprintf_s snprintf
#else
#include <conio.h>
#endif

#include "ADQAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#include "utils.h"

//#define SINGLE_SHOT_COLLECTION

// Number of data buffers to queue up per channel
// Infinite streaming requires a higher number
#ifdef SINGLE_SHOT_COLLECTION
#define NOF_BUFFERS 10
#else
#define NOF_BUFFERS 10
#endif

#define MAX_NOF_CHANNELS (4)

/* Helper macro to handle failed ADQAPI function calls. */
#define CHECKADQ(f)              \
  if (!(f))                      \
  {                              \
    printf("Error in " #f "\n"); \
    goto error;                  \
  }

int adq14(void *adq_cu, unsigned int adq_num)
{
  /* WFA settings */

  /* Data format of the WFA buffers. This example only supports:
   * - ATD_WFA_BUFFER_FORMAT_STRUCT
   */
  const enum ATDWFABufferFormat wfa_buffer_format = ATD_WFA_BUFFER_FORMAT_STRUCT;

  /* Number of accumulations */
  unsigned int wfa_nof_accumulations = 64;

#ifndef SINGLE_SHOT_COLLECTION
  /* Software accumulator implemented in this example code. Extends the data to
   * 64-bit samples. The total number of accumulations will be
   *     wfa_nof_accumulations * nof_software_accumulations
   * and any number of software accumulations >1 must be accounted for by the
   * number of repeats (wfa_nof_repeats).
   */
  const unsigned int nof_software_accumulations = 1;
#endif

  /* Samples, subject to adjustment, see call to
   * ADQ_ATDGetAdjustedRecordLength() */
  unsigned int wfa_record_length = 64512;

  /* Search direction input to ADQ_ATDGetAdjustedRecordLength(). '-1'
   * descending, '1' ascending. */
  int wfa_rlen_search_direction = 1;

  /* Samples before trigger */
  unsigned int wfa_nof_pretrig_samples = 0;
  unsigned int wfa_nof_triggerdelay_samples = 0;
  /* Set wfa_nof_repeats to 0xFFFFFFFF to enable infinite streaming */
  /* set wfa_nof_repeats = repeats * nof_software_accumulations if software
   * accumulator is used */
  unsigned int wfa_nof_repeats = 2;
  unsigned int wfa_bypass = 0;
  char wfa_channel_mask = 0x3;

  // WFA status
  unsigned int wfa_progress_percent = 0;
  unsigned int wfa_records_collected = 0;
  unsigned int stream_status = 0;
  unsigned int wfa_status = 0;

  // Software accumulator (64-bit)
  long long int *acc_data[MAX_NOF_CHANNELS] = {NULL};

  // Program options
  int use_synthetic_data = 1;
  int use_single_channel = 1;

  /* Device-to-host transfer settings */
  const unsigned int transfer_buffer_size = 1 * 1024 * 1024;
  const unsigned int nof_transfer_buffers = 8;

  /*
   * Data storage mode. One file is created per channel. Supported modes:
   *   - FWM_DISABLE (no files will be written)
   *   - FWM_ASCII
   *   - FWM_BINARY
   */
  const enum FileWriterMode write_mode = FWM_ASCII;

  /* Split data from repeated acquisitions into different files. */
  const unsigned int file_split = 0;

  // Trigger type: ADQ_LEVEL_TRIGGER_MODE ADQ_EXT_TRIGGER_MODE
  unsigned int trigger_mode = ADQ_LEVEL_TRIGGER_MODE;

  /* Internal Trigger specific options */
  /* In samples */
  unsigned int int_trigger_period = 200 * 1000;

  // External Trigger specific options
  double ext_trigger_threshold = 0.1;
  unsigned int ext_trigger_edge = 1;

  // Level Trigger specific options
  int level_trigger_level = 1000;
  int level_trigger_resetval = 300;
  int level_trigger_edge = 1;
  unsigned int level_trigger_channel = 1;

  // Bias level
  int adjustable_bias = 0; // Codes

  // DBS settings
  unsigned int dbs_nof_inst = 0;
  unsigned char dbs_inst = 0;
  int dbs_bypass = 1;
  int dbs_dc_target = adjustable_bias;
  int dbs_lower_saturation_level = 0;
  int dbs_upper_saturation_level = 0;

  // Iterators and status variables
  unsigned int ch = 0;
  /* unsigned int sample = 0; */
  unsigned int nof_channels = 0;
  unsigned int i = 0;
  int wfa_record_length_adjusted = -1;
  unsigned int wfa_device_nof_accumulations = 0;
  unsigned int wfa_partition_lower_bound = 0;
  unsigned int wfa_partition_upper_bound = 0;

  struct ATDWFABufferStruct *target_buffers[MAX_NOF_CHANNELS] = {NULL};

#ifndef SINGLE_SHOT_COLLECTION
  // Collection status
  unsigned int nof_received_records_sum = 0;
  unsigned int nof_received_records[MAX_NOF_CHANNELS] = {0};
  unsigned int received_all_records = 1;

  unsigned int acc_ctr[MAX_NOF_CHANNELS] = {0};
  struct ATDWFABufferStruct *current_buffer = NULL;
#endif

  /* Output files */
  FILE *outfile_data[MAX_NOF_CHANNELS] = {NULL};
#define outfile_mode_len 3
  char outfile_mode[outfile_mode_len];
#define filename_len 16
  char filename[filename_len];

  // Time measurement
  time_t t;
  struct tm ts;

  unsigned int nof_adc_cores;

  if (!ADQ_GetNofAdcCores(adq_cu, adq_num, &nof_adc_cores))
  {
    printf("ERROR: Failed to read the number of ADC cores.\n");
    goto error;
  }

  if (wfa_buffer_format != ATD_WFA_BUFFER_FORMAT_STRUCT)
  {
    printf("ERROR: Invalid WFA buffer format. This example only supports the "
           "struct buffer format.\n");
    goto error;
  }

  t = time(NULL);
#ifdef LINUX
  ts = *localtime(&t);
#else
  localtime_s(&ts, &t);
#endif
  printf("FWATD example started %2.2d:%2.2d:%2.2d.\n\n", ts.tm_hour, ts.tm_min, ts.tm_sec);

#ifdef SINGLE_SHOT_COLLECTION
  if (wfa_nof_repeats == 0xFFFFFFFF)
  {
    printf("\nERROR: Single shot mode cannot be combined with infinite "
           "streaming! (wfa_nof_repeats = 0xFFFFFFFF )\n");
    goto error;
  }

  if (NOF_BUFFERS < wfa_nof_repeats)
  {
    printf("NOF_BUFFERS has to be greater than or equal to the number of "
           "repeats in single-shot mode.\n");
    goto error;
  }
#endif

  /* Open output files for data and headers */
  switch (write_mode)
  {
  case FWM_ASCII:
    sprintf_s(outfile_mode, outfile_mode_len, "w");
    break; /* ASCII */
  case FWM_BINARY:
    sprintf_s(outfile_mode, outfile_mode_len, "wb");
    break; /* Binary */
  default:
    break;
  }

  if (use_single_channel)
    wfa_channel_mask = 0x1;

  // Get number of channels
  nof_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  if ((!file_split) && (write_mode != FWM_DISABLE))
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      if (!((1 << ch) & wfa_channel_mask))
        continue;

      sprintf_s(filename, filename_len, "data%c.out", "ABCD"[ch]);
      outfile_data[ch] = fopen(filename, outfile_mode);
      if (!(outfile_data[ch]))
      {
        printf("Failed to open output file %s.\n", filename);
        goto error;
      }
    }
  }

  if (use_single_channel)
    nof_channels = 1;

  CHECKADQ(ADQ_ATDSetWFABufferFormat(adq_cu, adq_num, wfa_buffer_format));

  // Set Clock Source
  CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, ADQ_CLOCK_INT_INTREF));

  // Set Trigger
  CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trigger_mode));

  switch (trigger_mode)
  {
  case 1:
    printf("Trigger setup failed.\n");
    goto cleanup_exit;
  case ADQ_EXT_TRIGGER_MODE:
    CHECKADQ(ADQ_SetExtTrigThreshold(adq_cu, adq_num, 1, ext_trigger_threshold));
    CHECKADQ(ADQ_SetTriggerEdge(adq_cu, adq_num, trigger_mode, ext_trigger_edge));
    break;
  case ADQ_LEVEL_TRIGGER_MODE:
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, level_trigger_level));
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, level_trigger_edge));
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, level_trigger_channel));
    CHECKADQ(ADQ_SetTrigLevelResetValue(adq_cu, adq_num, level_trigger_resetval));
    break;
  case ADQ_INTERNAL_TRIGGER_MODE:
    CHECKADQ(ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, int_trigger_period));
    CHECKADQ(ADQ_SetConfigurationTrig(adq_cu, adq_num, 5, 50, 0));
    break;
  default:
    printf("Trigger setup failed.\n");
    goto cleanup_exit;
  }

  // Enable internal test pattern generation if use_synthetic_data was set
  if (use_synthetic_data)
  {
    printf("Using synthetic data.\n");
    for (ch = 0; ch < nof_adc_cores; ch++)
    {
      CHECKADQ(ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + ch + 1, 1024, 0));
    }
    CHECKADQ(ADQ_SetTestPatternMode(adq_cu, adq_num, 4));
  }
  else
  {
    printf("Using data from ADC.\n");
    for (ch = 0; ch < nof_adc_cores; ch++)
    {
      CHECKADQ(ADQ_SetGainAndOffset(adq_cu, adq_num, ch + 1, 1024, 0));
    }
    CHECKADQ(ADQ_SetTestPatternMode(adq_cu, adq_num, 0));
  }

  // Setup adjustable bias
  if (ADQ_HasAdjustableBias(adq_cu, adq_num))
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      if (!ADQ_SetAdjustableBias(adq_cu, adq_num, ch + 1, adjustable_bias))
      {
        printf("Failed setting adjustable bias for ch %c.\n", "ABCD"[ch]);
        goto error;
      }

      printf("Adjustable bias for ch %c set to %d codes.\n", "ABCD"[ch], adjustable_bias);
    }

    if (!dbs_bypass)
    {
      printf("Waiting for bias settling...\n");
      Sleep(1000);
    }
  }

  // Setup DBS
  ADQ_GetNofDBSInstances(adq_cu, adq_num, &dbs_nof_inst);
  for (dbs_inst = 0; dbs_inst < dbs_nof_inst; ++dbs_inst)
  {
    printf("Setting up DBS instance %u ...\n", dbs_inst);
    if (!ADQ_SetupDBS(adq_cu, adq_num, dbs_inst, dbs_bypass, dbs_dc_target,
                      dbs_lower_saturation_level, dbs_upper_saturation_level))
    {
      printf("Failed setting up DBS instance %d.", dbs_inst);
      goto error;
    }
  }

  if (!dbs_bypass)
    Sleep(1000);

  // Adjust WFA partitioning here if needed (very advanced, please refer to the
  // reference guide before making any changes).
  if (!ADQ_ATDSetWFAPartitionBoundariesDefault(adq_cu, adq_num))
  {
    printf("An error occurred during partition boundary setup.\n");
    goto error;
  }

  // Report partition boundaries.
  if (!ADQ_ATDGetWFAPartitionBoundaries(adq_cu, adq_num, &wfa_partition_lower_bound,
                                        &wfa_partition_upper_bound))
  {
    printf("An error occurred during partition boundary retrieval.\n");
    goto error;
  }
  printf("Current partition boundaries: [%u, %u].\n", wfa_partition_lower_bound,
         wfa_partition_upper_bound);

  // Adjust the record length according to the FWATD rule set.
  // printf("Adjusting the record length...\n");
  wfa_record_length_adjusted = ADQ_ATDGetAdjustedRecordLength(adq_cu, adq_num, wfa_record_length,
                                                              wfa_rlen_search_direction);

  if (wfa_record_length_adjusted > 0)
  {
    wfa_record_length = (unsigned int)wfa_record_length_adjusted;
    printf("Adjusted record length is %u samples.\n", wfa_record_length);
  }
  else
  {
    printf("An error occurred while adjusting the record length. "
           "Error code: %d.\n",
           wfa_record_length_adjusted);
    goto cleanup_exit;
  }

  // Retrieve information on the workload split.
  wfa_device_nof_accumulations = ADQ_ATDGetDeviceNofAccumulations(adq_cu, adq_num,
                                                                  wfa_nof_accumulations);

  printf("Using current partition boundaries, the ADQ will "
         "perform %u accumulations.\n",
         wfa_device_nof_accumulations);
  printf("The incoming data rate will be reduced by this factor when "
         "transferred on the host-interface.\n");

  for (ch = 0; ch < MAX_NOF_CHANNELS; ch++)
  {
    if (!((1 << ch) & wfa_channel_mask))
    {
      target_buffers[ch] = NULL;
      continue;
    }

    target_buffers[ch] = (struct ATDWFABufferStruct *)malloc(NOF_BUFFERS
                                                             * sizeof(struct ATDWFABufferStruct));

    if (!target_buffers[ch])
    {
      printf("Failed to allocate memory for target buffer.\n");
      goto error;
    }

    for (i = 0; i < NOF_BUFFERS; i++)
    {
      target_buffers[ch][i].Data = (int *)malloc(sizeof(int) * wfa_record_length);
      if (!target_buffers[ch][i].Data)
      {
        printf("Failed to allocate memory for target buffer data region.\n");
        goto error;
      }
    }
  }

  // Allocate memory for 64-bit software accumulator
  for (ch = 0; ch < nof_channels; ch++)
  {
    if (!((1 << ch) & wfa_channel_mask))
      continue;

    acc_data[ch] = (long long int *)malloc(sizeof(long long int) * wfa_record_length);
    if (acc_data[ch] == NULL)
    {
      printf("Failed to allocate memory for SW accumulator buffer on channel %c.\n", "ABCD"[ch]);
      goto error;
    }
  }

  // Setup size of transfer buffers.
  printf("\nSetting up streaming...\n");
  printf("wfa_record_length       %u\n", wfa_record_length);
  printf("wfa_nof_accumulations   %u\n", wfa_nof_accumulations);
  printf("wfa_nof_pretrig_samples %u\n", wfa_nof_pretrig_samples);
  printf("wfa_nof_triggerdelay_samples %u\n", wfa_nof_triggerdelay_samples);
  printf("wfa_nof_repeats         %u\n", wfa_nof_repeats);
  printf("wfa_bypass              %u\n", wfa_bypass);

  /* Set transfer buffers */
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, nof_transfer_buffers, transfer_buffer_size));

  // Setup WFA
  CHECKADQ(ADQ_ATDSetupWFA(adq_cu, adq_num, wfa_record_length, wfa_nof_pretrig_samples,
                           wfa_nof_triggerdelay_samples, wfa_nof_accumulations, wfa_nof_repeats));

#ifdef SINGLE_SHOT_COLLECTION
  // Start WFA transfer.
  CHECKADQ(ADQ_ATDStartWFA(adq_cu, adq_num, (void **)target_buffers, wfa_channel_mask, 0));

  do
  {
    CHECKADQ(ADQ_ATDGetWFAStatus(adq_cu, adq_num, &wfa_progress_percent, &wfa_records_collected,
                                 &stream_status, &wfa_status));

    printf("Progress: %u%%\n", wfa_progress_percent);
    if (wfa_status > 0)
      goto cleanup_exit;

    /* Check for any key press indicating an abort */
    if (_kbhit())
    {
      printf("User aborted\n");
      ADQ_ATDStopWFA(adq_cu, adq_num);
      printf("Waiting for ATD completion.\n");
      CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));
      goto cleanup_exit;
    }

    Sleep(100);
  } while (wfa_progress_percent < 100);

  printf("Data collection finished successfully.\n");

  /* Save data to file */
  if (write_mode != FWM_DISABLE)
  {
    for (ch = 0; ch < nof_channels; ++ch)
    {
      if (!((1 << ch) & wfa_channel_mask))
        continue;

      printf("Saving %u records from channel %u.\n", wfa_nof_repeats, ch + 1);
      for (i = 0; i < wfa_nof_repeats; ++i)
      {
        if (file_split)
        {
          sprintf_s(filename, filename_len, "data%c_%u.out", "ABCD"[ch], i);
          fopen_s(&outfile_data[ch], filename, outfile_mode);
          if (!(outfile_data[ch]))
          {
            printf("Failed to open output file %s.\n", filename);
            goto error;
          }
        }

        atd_file_writer(write_mode, target_buffers[ch][i].Data, wfa_record_length,
                        sizeof(target_buffers[ch][i].Data[0]), outfile_data[ch]);

        if (file_split)
        {
          fclose(outfile_data[ch]);
          outfile_data[ch] = NULL;
        }
      }
    }
  }

#else
  // Register all WFA buffers.
  for (i = 0; i < NOF_BUFFERS; i++)
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      if (!((1 << ch) & wfa_channel_mask))
        continue;

      CHECKADQ(ADQ_ATDRegisterWFABuffer(adq_cu, adq_num, ch + 1, &target_buffers[ch][i]));
    }
  }

  // Start WFA transfer.
  CHECKADQ(ADQ_ATDStartWFA(adq_cu, adq_num, NULL, wfa_channel_mask, 0));

  // Fetch each WFA result separately.
  do
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      if (!((1 << ch) & wfa_channel_mask))
        continue;
      if ((wfa_nof_repeats != 0xFFFFFFFF) || (nof_received_records[ch] % 1000 == 0))
      {
        printf("\nWaiting for WFA buffer for channel %c...\n", "ABCD"[ch]);
      }

      if (!ADQ_ATDWaitForWFABuffer(adq_cu, adq_num, ch + 1, (void **)(&current_buffer), 0))
      {
        // Timeout while waiting for buffer.
        if (current_buffer == NULL)
          continue;

        printf("Failed to get data buffer for channel %c.\n", "ABCD"[ch]);

        if (current_buffer == (struct ATDWFABufferStruct *)-1)
          printf("Streaming overflow occurred while waiting for buffer.\n");

        ADQ_ATDStopWFA(adq_cu, adq_num);
        ADQ_ATDWaitForWFACompletion(adq_cu, adq_num);
        goto error;
      }

      printf("Record %u, timestamp 0x%" PRIu64 "\n", nof_received_records[ch],
             current_buffer->Timestamp);

      if (file_split && (write_mode != FWM_DISABLE))
      {
        sprintf_s(filename, filename_len, "data%c_%u.out", "ABCD"[ch], nof_received_records[ch]);
        outfile_data[ch] = fopen(filename, outfile_mode);
        if (!(outfile_data[ch]))
        {
          printf("Failed to open output file %s.\n", filename);
          goto error;
        }
      }

      /* Software accumulator (64-bit) */
      if (acc_ctr[ch] == 0)
      {
        /* Overwrite */
        for (i = 0; i < wfa_record_length; i++)
          acc_data[ch][i] = (long long int)current_buffer->Data[i];
      }
      else
      {
        /* Accumulate */
        for (i = 0; i < wfa_record_length; i++)
          acc_data[ch][i] += (long long int)current_buffer->Data[i];
      }

      if ((++acc_ctr[ch]) == nof_software_accumulations)
      {
        /* Current accumulator is completed. */
        /* Save data to file */
        if (write_mode != FWM_DISABLE)
        {
          atd_file_writer(write_mode, acc_data[ch], wfa_record_length, sizeof(acc_data[ch][0]),
                          outfile_data[ch]);
          if (file_split)
          {
            fclose(outfile_data[ch]);
            outfile_data[ch] = NULL;
          }
        }
        /* Reset current accumulator */
        acc_ctr[ch] = 0;
      }

      // Make buffer available for writing again
      CHECKADQ(ADQ_ATDRegisterWFABuffer(adq_cu, adq_num, ch + 1, current_buffer));

      ++nof_received_records[ch];
    }
    ADQ_ATDGetWFAStatus(adq_cu, adq_num, &wfa_progress_percent, &wfa_records_collected,
                        &stream_status, &wfa_status);

    nof_received_records_sum = 0;
    for (ch = 0; ch < nof_channels; ++ch)
      nof_received_records_sum += nof_received_records[ch];

    if ((wfa_nof_repeats != 0xFFFFFFFF) || (nof_received_records[ch] % 1000 == 0))
    {
      printf("wfa_progress_percent:  %u\n", wfa_progress_percent);
      printf("wfa_records_collected: %u\n", wfa_records_collected);
      printf("stream_status:         %u\n", stream_status);
      printf("wfa_status:            %u\n", wfa_status);
    }

    // Completion criteria
    received_all_records = 1;
    for (ch = 0; ch < nof_channels; ++ch)
    {
      if (!((1 << ch) & wfa_channel_mask))
        continue;
      if ((nof_received_records[ch] < wfa_nof_repeats) || (wfa_nof_repeats == 0xFFFFFFFF))
        received_all_records = 0;
    }

    if (_kbhit())
    {
      printf("\nUser aborted! Shutting down acquisition...\n");
      break;
    }

  } while (!received_all_records);

#endif

  // Always call ATDStopWFA before ATDWaitForWFACompletion
  printf("Stopping WFA...\n");
  ADQ_ATDStopWFA(adq_cu, adq_num);
  // Collection is done, finish by calling ATDWaitForWFACompletion().
  printf("Waiting for ATD completion.\n");
  CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));

  goto cleanup_exit;

error:
  /*
   * Do not use the CHECKADQ macro below this line. It includes 'goto error',
   * which would cause a failing command to go into an infinite loop.
   */
  printf("ERROR: An error occurred\n");
  printf("Please view the trace logs for more information\n");
  // This will help to recover the device on the next start if the device is
  // stuck in some weird streaming state.
  if (ADQ_IsUSBDevice(adq_cu, adq_num) || ADQ_IsUSB3Device(adq_cu, adq_num))
  {
    printf("\nPerforming ResetDevice 18 preemptively...\n");
    ADQ_ResetDevice(adq_cu, adq_num, 18);
  }

cleanup_exit:

  printf("Returning memory.\n");

  for (ch = 0; ch < MAX_NOF_CHANNELS; ch++)
  {
    if (target_buffers[ch] == NULL)
      continue;
    for (i = 0; i < NOF_BUFFERS; i++)
    {
      if (target_buffers[ch][i].Data)
      {
        free(target_buffers[ch][i].Data);
        target_buffers[ch][i].Data = NULL;
      }
    }
    if (target_buffers[ch])
    {
      free(target_buffers[ch]);
      target_buffers[ch] = NULL;
    }
  }

  for (ch = 0; ch < MAX_NOF_CHANNELS; ch++)
  {
    if (acc_data[ch])
      free(acc_data[ch]);
  }

  /* Close any open output files */
  if ((!file_split) && (write_mode != FWM_DISABLE))
  {
    for (ch = 0; ch < MAX_NOF_CHANNELS; ch++)
    {
      if (!((1 << ch) & wfa_channel_mask))
        continue;
      if (outfile_data[ch])
        fclose(outfile_data[ch]);
    }
  }
  return 0;
}
