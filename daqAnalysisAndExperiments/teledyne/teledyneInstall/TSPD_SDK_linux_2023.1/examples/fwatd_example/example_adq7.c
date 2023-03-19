/*
 *  Copyright 2019 Teledyne Signal Processing Devices Sweden AB
 */

#include "ADQAPI.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"

#ifdef LINUX
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
extern int _kbhit();
#define sprintf_s snprintf
#else
/* For kbhit */
#include <conio.h>
#endif

/* Helper macro to handle failed ADQAPI function calls. */
#define CHECKADQ(f)              \
  if (!(f))                      \
  {                              \
    printf("Error in " #f "\n"); \
    goto error;                  \
  }

/*
 * Number of buffers to rotate in the streaming queue for one channel. Higher
 * throughput requires more buffers.
 */
#define NOF_BUFFERS (10)

/*
 * Number of filter coefficients for the linear phase FIR filter in user
 * logic 1.
 */
#define NOF_COEFFS_UL1 (9)

/*
 * Data collection mode
 * The single shot collection should NOT be used together with 'infinite
 * streaming', i.e. to capture an endless stream of data since it requires
 * preallocation of the memory needed to store all the data. The value on
 * NOF_BUFFERS above should be greater or equal to the number of repeats if
 * single-shot collection is activated.
 */

// #define SINGLE_SHOT_COLLECTION

int adq7(void *adq_cu, unsigned int adq_num)
{
  /* WFA Settings */

  /* [1, 266144] for safe scaling */
  const unsigned int wfa_nof_accumulations = 1024;

  /* Set to '-1' for infinite streaming */
  const unsigned int wfa_nof_repeats = 10;

  const unsigned int wfa_record_length = 8192;
  const unsigned int wfa_pretrig_samples = 0;
  const unsigned int wfa_triggerdelay_samples = 0;
  const unsigned char channel_mask = 0x01;

#ifndef SINGLE_SHOT_COLLECTION
  /*
   * Software accumulator implemented in this example code. Extends the data to
   * 64-bit samples. The total number of accumulations will be
   *     wfa_nof_accumulations * nof_software_accumulations
   * and any number of software accumulations >1 must be accounted for by the
   * number of repeats (wfa_nof_repeats).
   */
  const unsigned int nof_software_accumulations = 1;
#endif

  /* Sample skip */
  const unsigned int sample_skip_factor = 1;

  /*
   * Data storage mode. One file is created per channel. Supported modes:
   *   - FWM_DISABLE (no files will be written)
   *   - FWM_ASCII
   *   - FWM_BINARY
   */
  const enum FileWriterMode write_mode = FWM_ASCII;

  /* Split data from repeated acquisitions into different files. */
  const unsigned int file_split = 0;

  /*
   * Trigger configuration. Valid values are defined in the ADQAPI header file
   * and repeated below for convenience.
   *   - ADQ_LEVEL_TRIGGER_MODE
   *   - ADQ_INTERNAL_TRIGGER_MODE
   *   - ADQ_EXT_TRIGGER_MODE
   */
  const int trig_mode = ADQ_LEVEL_TRIGGER_MODE;

  /*
   * Internal trigger synchronization mode.
   *   - 0: Always enabled (default)
   *   - 3: Synchronize on first external trigger event.
   */
  const unsigned int int_trigger_sync_mode = 0;

  /* Level trigger parameters */
  const int trig_level = 0; /* ADC codes */
  const int trig_edge = 1; /* 1: Rising, 0: Falling */
  const int trig_channel = 1;

  /* External trigger parameters */
  const double ext_trigger_threshold = 0.5f; /* In Volts [-0.5, 3.3] */

  /* Internal trigger parameters */
  const unsigned int int_trigger_period = 1000000; /* In ADC clock cycles */

  /* Trigger blocking */
  const int enable_trigger_blocking = 0;
  const unsigned int trigger_blocking_mode = 0; /* ONCE */
  const unsigned int trigger_blocking_source = 9; /* SYNC */
  const uint64_t trigger_blocking_window_length = 0;
  const unsigned int trigger_blocking_tcount_limit = 0;

  /* Timestamp synchronization */
  const int enable_timestamp_sync = 0;
  const unsigned int timestamp_sync_mode = 1;
  const unsigned int timestamp_sync_source = 0;

  /* Threshold for SYNC input. Used if trigger_blocking_source == 9 */
  const double sync_threshold = 0.0f;

  /* Accumulation grid synchronization */
  const unsigned int enable_accumulation_grid_sync = 0;

  /*
   * Test pattern
   *  - 0: Disabled
   *  - 2: Positive sawtooth wave
   *  - 3: Negative sawtooth wave
   *  - 4: Triangle wave
   */
  const int test_pattern_mode = 4;

  /*
   * Clock source
   *  - 0: Internal clock source, internal 10 MHz reference
   *  - 1: Internal clock source, external 10 MHz reference
   *  - 2: External clock source
   */
  const int clock_source = 0;

  /* Enable output of 10 MHz reference clock */
  const unsigned int clock_ref_out_enable = 0;

#ifndef SINGLE_SHOT_COLLECTION
  /* Timeout to use when waiting for a completed record in user-space */
  const unsigned int timeout_ms = 1000; /* 0: Wait indefinitely */
#endif

  /*
   * Bias level in ADC codes [-32768, 32767]
   * ADQ7 has an input range of 1 Vpp. If X is the desired DC offset in mV, the
   * adjustable_bias value is computed as: adjustable_bias = 32768 * X / 500
   */
  int adjustable_bias = 0;

  /* Digital baseline stabilization configuration */
  unsigned int dbs_nof_inst = 0;
  unsigned char dbs_inst = 0;
  const int dbs_bypass = 1;
  const int dbs_dc_target = adjustable_bias;
  const int dbs_lower_saturation_level = 0;
  const int dbs_upper_saturation_level = 0;

  /*
   * Threshold module settings,
   *  - 1-channel mode: Channel X uses the settings at index 0
   *  - 2-channel mode: Channel A and B use the settings at index 0 and 1,
   *                    respectively.
   */
  int thr_threshold[2] = {0, 0};
  int thr_baseline[2] = {0, 0};

  /*
   * Threshold polarity
   *  - 0: Positive, values below the threshold are replaced with the baseline.
   *  - 1: Negative, values above the threshold are replaced with the baseline.
   */
  unsigned int thr_polarity[2] = {0, 0};
  unsigned int thr_bypass[2] = {1, 1};

  /*
   * Threshold filter coefficients
   *  The coefficients use a 16-bit 2's complement representation with
   *  14 fractional bits. The filter is an linear phase FIR filter of order 16.
   *  Thus, the coefficient at index 8 is the point of symmetry in the filter's
   *  impulse response. In 1-channel mode, the filter coefficients for
   *  channel A are used for channel X.
   */
  unsigned int thr_coefficients_cha[9] = {0, 0, 0, 0, 0, 0, 0, 0, (1 << 14)};
  unsigned int thr_coefficients_chb[9] = {0, 0, 0, 0, 0, 0, 0, 0, (1 << 14)};

  /*
   * User logic filter coefficients
   *  The filter is an linear phase FIR filter of order 32. Thus, the
   *  coefficient at index 16 is the point of symmetry in the filter's impulse
   *  response. In 1-channel mode, the filter coefficients for channel A are
   *  used for channel X.

   *  There are two ways to format the coefficients to the filter: Either a
   *  fixed point representation (16-bit 2's complement with 14 fractional
   *  bits) or a floating point representation, subjected to rounding, may
   *  be used. The format is controlled by ul_format and the rounding mode
   *  by ul_rounding_mode.
   */
  unsigned int ul_coefficients_fp_cha[NOF_COEFFS_UL1] = {0, 0, 0, 0, 0, 0, 0, 0, (1 << 14)};
  unsigned int ul_coefficients_fp_chb[NOF_COEFFS_UL1] = {0, 0, 0, 0, 0, 0, 0, 0, (1 << 14)};
  float ul_coefficients_float_cha[NOF_COEFFS_UL1] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 0.0f, 1.0f};
  float ul_coefficients_float_chb[NOF_COEFFS_UL1] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 0.0f, 1.0f};
  /*
   * User logic filter coefficient representation
   *  - 0: Fixed point (2's complement Q16.14)
   *  - 1: Floating point (subjected to rounding to reach Q16.14)
   */
  unsigned int ul_format[2] = {1, 1};
  /*
   * User logic filter coefficient rounding method. Discarded if the format
   * is 'fixed point'. The trace log file reports the rounded value as 'INFO'.
   *  - 0: Symmetric rounding, tie away from zero.
   *  - 1: Symmetric rounding, tie towards zero.
   *  - 2: Symmetric rounding, tie to even.
   */
  unsigned int ul_rounding_method[2] = {0, 0};

  /* Enable or disable the user logic filter. */
  const unsigned int ul_bypass[2] = {1, 1};

  /* Device-to-host transfer settings */
  const unsigned int transfer_buffer_size = 1 * 1024 * 1024;
  const unsigned int nof_transfer_buffers = 8;

  /* Internal variables representing memory and status */
  struct ATDWFABufferStruct **target_buffers = NULL;
#ifndef SINGLE_SHOT_COLLECTION
  struct ATDWFABufferStruct *current_buffer;
  unsigned int received_all_records;

  /* Software accumulator (64-bit) */
  unsigned int acc_ctr[2] = {0};
  long long int *acc_data[2];
#endif
  unsigned int wfa_status;
  unsigned int wfa_progress_percent;
  unsigned int wfa_records_collected;
  unsigned int wfa_stream_status;
  unsigned int nof_received_records[2] = {0};
  unsigned int nof_channels;
  /* Output files */
  FILE *outfile_data[2] = {NULL, NULL};
#define outfile_mode_len 3
  char outfile_mode[outfile_mode_len];
#define filename_len 16
  char filename[filename_len];
  unsigned int ch;
  unsigned int i;
  int exit;

  /* Create indexable array of the pointers. */
  unsigned int *thr_coefficients[2] = {thr_coefficients_cha, thr_coefficients_chb};
  void *ul_coefficients[2] = {NULL};
  ul_coefficients[0] = ul_format[0] ? (void *)ul_coefficients_float_cha
                                    : (void *)ul_coefficients_fp_cha;
  ul_coefficients[1] = ul_format[1] ? (void *)ul_coefficients_float_chb
                                    : (void *)ul_coefficients_fp_chb;

  /* Fetch the number of digitizer channels */
  nof_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  /* Validate channel mask */
  if (!channel_mask)
  {
    printf("The channel mask is zero.\n");
    goto error;
  }

  if (nof_channels == 1 && channel_mask != 0x1)
  {
    printf("The digitizer is running in 1-channel mode so the channel mask "
           "must be set to 0x01.\n");
    goto error;
  }

  /* Validate digitizer firmware */
  if (ADQ_HasFeature(adq_cu, adq_num, "FWATD") == 1)
  {
    printf("Device supports FWATD.\n");
  }
  else
  {
    printf("Target device does not support the advanced time-domain feature. "
           "Either the device is not running the FWATD firmware, or the "
           "license check failed.\n");
    goto error;
  }

#ifdef SINGLE_SHOT_COLLECTION
  if (NOF_BUFFERS < wfa_nof_repeats)
  {
    printf("NOF_BUFFERS has to be greater than or equal to the number of "
           "repeats in single-shot mode.\n");
    goto error;
  }
#endif

  /* Allocate memory */
  printf("Allocating memory for target buffers.\n");
  target_buffers = (struct ATDWFABufferStruct **)malloc(2 * sizeof(struct ATDWFABufferStruct *));
  if (!target_buffers)
  {
    printf("Failed to allocate memory for target buffer.\n");
    goto error;
  }
  for (ch = 0; ch < 2; ch++)
  {
    if (!((1 << ch) & channel_mask))
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
      target_buffers[ch][i].Data = (int *)calloc(sizeof(int), wfa_record_length);
      if (!target_buffers[ch][i].Data)
      {
        printf("Failed to allocate memory for target buffer data region.\n");
        goto error;
      }
    }
#ifndef SINGLE_SHOT_COLLECTION
    /* Buffers used to perform software accumulation */
    acc_data[ch] = (long long int *)calloc(sizeof(long long int), wfa_record_length);
    if (!acc_data[ch])
    {
      printf("Failed to allocate memory for software accumulator.\n");
      goto error;
    }
#endif
  }

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
  if ((!file_split) && (write_mode != FWM_DISABLE))
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      if (!((1 << ch) & channel_mask))
        continue;

      sprintf_s(filename, filename_len, "data%c.out", "AB"[ch]);
      outfile_data[ch] = fopen(filename, outfile_mode);
      if (!(outfile_data[ch]))
      {
        printf("Failed to open output file %s.\n", filename);
        goto error;
      }
    }
  }

  /* Set transfer buffers */
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, nof_transfer_buffers, transfer_buffer_size));

  /* Clocking configuration */
  CHECKADQ(ADQ_EnableClockRefOut(adq_cu, adq_num, clock_ref_out_enable));
  if (!clock_ref_out_enable)
  {
    /* Configure clock source unless reference output is active */
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));
  }

  /* Configure trigger */
  switch (trig_mode)
  {
  case ADQ_INTERNAL_TRIGGER_MODE:
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    /* Sampling frequency is 10 GHz for 1 CH ADQ7 */
    CHECKADQ(ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, int_trigger_period));
    CHECKADQ(ADQ_SetInternalTriggerSyncMode(adq_cu, adq_num, int_trigger_sync_mode));
    break;
  case ADQ_EXT_TRIGGER_MODE:
    CHECKADQ(ADQ_SetTriggerThresholdVoltage(adq_cu, adq_num, trig_mode, ext_trigger_threshold));
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    CHECKADQ(ADQ_SetTriggerEdge(adq_cu, adq_num, trig_mode, trig_edge));
    break;
  case ADQ_LEVEL_TRIGGER_MODE:
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_edge));
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    break;
  default:
    printf("Unsupported trigger mode %d.\n", trig_mode);
    goto error;
  }

  /* Configure test pattern */
  CHECKADQ(ADQ_SetTestPatternMode(adq_cu, adq_num, test_pattern_mode));
  if (test_pattern_mode != 0)
  {
    CHECKADQ(ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + 1, 1024, 0));
    CHECKADQ(ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + 2, 1024, 0));
    CHECKADQ(ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + 3, 1024, 0));
    CHECKADQ(ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + 4, 1024, 0));
  }

  /* Configure sample skip */
  CHECKADQ(ADQ_SetSampleSkip(adq_cu, adq_num, sample_skip_factor));

  /* Configure adjustable bias */
  if (ADQ_HasAdjustableBias(adq_cu, adq_num))
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      CHECKADQ(ADQ_SetAdjustableBias(adq_cu, adq_num, ch + 1, adjustable_bias));
      printf("Adjustable bias for channel %c set to %d codes.\n", "AB"[ch], adjustable_bias);
    }
    printf("Waiting for bias settling...\n");
    Sleep(1000);
  }
  /* Configure the digital baseline stabilization */
  ADQ_GetNofDBSInstances(adq_cu, adq_num, &dbs_nof_inst);
  for (dbs_inst = 0; dbs_inst < dbs_nof_inst; ++dbs_inst)
  {
    printf("Setting up DBS instance %u ...\n", dbs_inst);
    CHECKADQ(ADQ_SetupDBS(adq_cu, adq_num, dbs_inst, dbs_bypass, dbs_dc_target,
                          dbs_lower_saturation_level, dbs_upper_saturation_level));
  }
  Sleep(1000);

  /* Configure user logic filter */
  for (ch = 0; ch < nof_channels; ch++)
  {
    printf("Setting up user logic filter for channel %c ...\n", "AB"[ch]);
    CHECKADQ(ADQ_ResetUserLogicFilter(adq_cu, adq_num, ch + 1));
    CHECKADQ(ADQ_SetUserLogicFilter(adq_cu, adq_num, ch + 1, ul_coefficients[ch], NOF_COEFFS_UL1,
                                    ul_format[ch], ul_rounding_method[ch]));
    CHECKADQ(ADQ_EnableUserLogicFilter(adq_cu, adq_num, ch + 1, !ul_bypass[ch]));
  }
  printf("Disable bypass of user logic 1.\n");
  CHECKADQ(ADQ_BypassUserLogic(adq_cu, adq_num, 1, 0));

  if (enable_trigger_blocking)
  {
    printf("Enable trigger blocking\n");
    CHECKADQ(ADQ_SetupTriggerBlocking(adq_cu, adq_num, trigger_blocking_mode,
                                      trigger_blocking_source, trigger_blocking_window_length,
                                      trigger_blocking_tcount_limit));
    CHECKADQ(ADQ_DisarmTriggerBlocking(adq_cu, adq_num));

    if (trigger_blocking_source == 9)
    {
      CHECKADQ(ADQ_SetTriggerThresholdVoltage(adq_cu, adq_num, 9, sync_threshold));
    }
  }
  else
  {
    /* Mode 4 disables trigger blocking. Other arguments are not used */
    printf("Disable trigger blocking\n");
    CHECKADQ(ADQ_SetupTriggerBlocking(adq_cu, adq_num, 4, 0, 0, 0));
  }

  if (enable_timestamp_sync)
  {
    CHECKADQ(ADQ_SetupTimestampSync(adq_cu, adq_num, timestamp_sync_mode, timestamp_sync_source));
  }

  /* Threshold module configuration */
  for (ch = 0; ch < nof_channels; ch++)
  {
    printf("Setting up threshold module for channel %c ...\n", "AB"[ch]);
    CHECKADQ(ADQ_ATDSetThresholdFilter(adq_cu, adq_num, ch + 1, thr_coefficients[ch]));
    CHECKADQ(ADQ_ATDSetupThreshold(adq_cu, adq_num, ch + 1, thr_threshold[ch], thr_baseline[ch],
                                   thr_polarity[ch], thr_bypass[ch]));
  }

  /* Configure accumulation grid synchronization. */
  CHECKADQ(ADQ_ATDEnableAccumulationGridSync(adq_cu, adq_num, enable_accumulation_grid_sync));

  /* WFA configuration */
  CHECKADQ(ADQ_ATDSetupWFA(adq_cu, adq_num, wfa_record_length, wfa_pretrig_samples,
                           wfa_triggerdelay_samples, wfa_nof_accumulations, wfa_nof_repeats));

  /* Register WFA buffers */
  printf("Registering target buffers...\n");
  for (ch = 0; ch < nof_channels; ch++)
  {
    for (i = 0; i < NOF_BUFFERS; i++)
    {
      if (!((1 << ch) & channel_mask))
        continue;
      CHECKADQ(ADQ_ATDRegisterWFABuffer(adq_cu, adq_num, ch + 1, &target_buffers[ch][i]));
    }
  }

  /* Reset the digitizer's timestamp. Timing is not guaranteed. */
  CHECKADQ(ADQ_ResetTimestamp(adq_cu, adq_num));

  if (enable_timestamp_sync)
  {
    CHECKADQ(ADQ_ArmTimestampSync(adq_cu, adq_num));
  }

#ifndef SINGLE_SHOT_COLLECTION
  /* Start WFA in non-blocking streaming mode */
  CHECKADQ(ADQ_ATDStartWFA(adq_cu, adq_num, NULL, channel_mask, 0));

  if (enable_trigger_blocking)
  {
    CHECKADQ(ADQ_ArmTriggerBlocking(adq_cu, adq_num));
  }

  /* Collection loop handling one completed user-space record at a time. */
  printf("Collecting data...\n");
  do
  {
    for (ch = 0; ch < nof_channels; ch++)
    {
      /* Skip any inactive channel */
      if (!((1 << ch) & channel_mask))
        continue;
      /* Wait for a completed WFA buffer, checking for any errors. */
      if (!ADQ_ATDWaitForWFABuffer(adq_cu, adq_num, ch + 1, (void **)(&current_buffer), timeout_ms))
      {
        /* Timeout while waiting for buffer */
        if (current_buffer == NULL)
          continue;

        printf("Failed to get data buffer for channel %c.\n", "AB"[ch]);

        if ((int *)current_buffer == (int *)(-2))
          printf("Streaming thread not running.\n");

        /*
         * Gracefully abort the WFA and wait for the 'all-clear' signal to exit.
         */
        CHECKADQ(ADQ_ATDStopWFA(adq_cu, adq_num));
        CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));
        goto cleanup_exit;
      }

      if ((file_split) && (write_mode != FWM_DISABLE))
      {
        sprintf_s(filename, filename_len, "data%c_%u.out", "AB"[ch],
                  current_buffer -> RecordNumber);
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
          atd_file_writer(write_mode, acc_data[ch], wfa_record_length, sizeof(long long int),
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

      /* Check meta information */
      if (current_buffer->Status > 0)
      {
        printf("Record %u on channel %u reports %u records accumulated.\n",
               current_buffer->RecordNumber, current_buffer->Channel,
               current_buffer->RecordsAccumulated);
        printf("Status 0x%08X.\n", current_buffer->Status);
      }

      /* Make buffer available for writing again */
      ADQ_ATDRegisterWFABuffer(adq_cu, adq_num, ch + 1, current_buffer);
      ++nof_received_records[ch];

      /* Check status */
      if ((wfa_nof_repeats == 0xFFFFFFFF) ? (nof_received_records[ch] % 10 == 0) : 1)
      {
        ADQ_ATDGetWFAStatus(adq_cu, adq_num, &wfa_progress_percent, &wfa_records_collected,
                            &wfa_stream_status, &wfa_status);
        printf("Progress:    %u%%\n", wfa_progress_percent);
        printf("Raw records: %u.\n", wfa_records_collected);
        if (wfa_status & (1u << 31))
        {
          printf("Queue starving!\n");
        }
        if (wfa_stream_status > 0)
        {
          printf("Streaming overflow, aborting...\n");
          CHECKADQ(ADQ_ATDStopWFA(adq_cu, adq_num));
          printf("Waiting for ATD completion.\n");
          CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));
          goto cleanup_exit;
        }
        for (i = 0; i < nof_channels; i++)
        {
          printf("Received %u records on channel %c.\n", nof_received_records[i], "AB"[i]);
        }
      }
    }
    /*
     * Completion criteria. Keep expecting data if any channel has not yet
     * received all data or if infinite streaming is activated.
     */
    received_all_records = 1;
    for (ch = 0; ch < nof_channels; ++ch)
    {
      if (!((1 << ch) & channel_mask))
        continue;
      if ((nof_received_records[ch] < wfa_nof_repeats) || (wfa_nof_repeats == 0xFFFFFFFF))
        received_all_records = 0;
    }
    /* Check for any key press indicating an abort */
    if (_kbhit())
    {
      printf("User aborted\n");
      CHECKADQ(ADQ_ATDStopWFA(adq_cu, adq_num));
      printf("Waiting for ATD completion.\n");
      CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));
      goto cleanup_exit;
    }
  } while (!received_all_records);

#else
  /*
   * Start WFA in non-blocking single-shot mode, should NOT be used together
   * with the infinite streaming mode.
   */
  CHECKADQ(ADQ_ATDStartWFA(adq_cu, adq_num, (void **)target_buffers, channel_mask, 0));

  if (enable_trigger_blocking)
  {
    CHECKADQ(ADQ_ArmTriggerBlocking(adq_cu, adq_num));
  }

  do
  {
    CHECKADQ(ADQ_ATDGetWFAStatus(adq_cu, adq_num, &wfa_progress_percent, &wfa_records_collected,
                                 &wfa_stream_status, &wfa_status));

    /* Ensure that the collection thread is still running */
    if (wfa_status & (1 << 29) && (wfa_progress_percent < 100))
    {
      printf("Unexpected WFA stop\n");
      CHECKADQ(ADQ_ATDStopWFA(adq_cu, adq_num));
      printf("Waiting for ATD completion.\n");
      CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));
      goto error;
    }

    printf("Progress: %u%%\n", wfa_progress_percent);
    /* Check for any key press indicating an abort */
    if (_kbhit())
    {
      printf("User aborted\n");
      CHECKADQ(ADQ_ATDStopWFA(adq_cu, adq_num));
      printf("Waiting for ATD completion.\n");
      CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));
      goto cleanup_exit;
    }

    /*
     * Mask 'collection thread not running' since a non-zero value is expected.
     */
    if ((wfa_status & ~0x20000000u) > 0)
      goto error;

    Sleep(100);
  } while (wfa_progress_percent < 100);

  /* Save data to file */
  if (write_mode != FWM_DISABLE)
  {
    for (ch = 0; ch < nof_channels; ++ch)
    {
      if (!((1 << ch) & channel_mask))
        continue;
      for (i = 0; i < wfa_nof_repeats; ++i)
      {
        printf("Record number %u reports %u accumulations.\n", target_buffers[ch][i].RecordNumber,
               target_buffers[ch][i].RecordsAccumulated);

        if ((file_split) && (write_mode != FWM_DISABLE))
        {
          sprintf_s(filename, filename_len, "data%c_%u.out", "AB"[ch],
                    target_buffers[ch][i].RecordNumber);
          fopen_s(&outfile_data[ch], filename, outfile_mode);
          if (!(outfile_data[ch]))
          {
            printf("Failed to open output file %s.\n", filename);
            goto error;
          }
        }

        atd_file_writer(write_mode, target_buffers[ch][i].Data, wfa_record_length, sizeof(uint32_t),
                        outfile_data[ch]);

        if (file_split)
        {
          fclose(outfile_data[ch]);
          outfile_data[ch] = NULL;
        }
      }
    }
  }
#endif

  printf("Waiting for ATD completion.\n");
  CHECKADQ(ADQ_ATDWaitForWFACompletion(adq_cu, adq_num));

  goto cleanup_exit;

error:
  /*
   * Do not use the CHECKADQ macro below this line. It includes 'goto error',
   * which would cause a failing command to go into an infinite loop.
   */
  printf("ERROR: An error occurred\n");
cleanup_exit:
  if (enable_timestamp_sync)
  {
    CHECKADQ(ADQ_DisarmTimestampSync(adq_cu, adq_num));
  }

  if (enable_trigger_blocking)
  {
    ADQ_DisarmTriggerBlocking(adq_cu, adq_num);
    /* Mode 4 is required to disable the trigger blocking */
    ADQ_SetupTriggerBlocking(adq_cu, adq_num, 4, 0, 0, 0);
  }

  if (ADQ_ATDGetWFAStatus(adq_cu, adq_num, NULL, NULL, NULL, &wfa_status))
  {
    /*
     * Mask 'collection thread not running' since a non-zero value is expected.
     */
    if ((wfa_status & ~0x20000000u) > 0)
    {
      printf("WFA status non-zero: 0x%08X\n", wfa_status);
    }
    else
    {
      printf("WFA status: OK\n");
    }
  }
  else
  {
    printf("ERROR: Failed to get WFA Status\n");
  }
  for (ch = 0; ch < nof_channels; ch++)
  {
    printf("Received %u records on channel %c.\n", nof_received_records[ch], "AB"[ch]);
  }

  /* Free memory */
  printf("Returning memory.\n");
  for (ch = 0; ch < 2 && target_buffers; ch++)
  {
    if (!target_buffers[ch])
      continue;
    for (i = 0; i < NOF_BUFFERS; i++)
    {
      if (target_buffers[ch][i].Data)
        free(target_buffers[ch][i].Data);
    }
    if (target_buffers[ch])
      free(target_buffers[ch]);
  }

  if (target_buffers)
    free(target_buffers);

  /* Close any open output files */
  if ((!file_split) && (write_mode != FWM_DISABLE))
  {
    for (ch = 0; ch < 2; ch++)
    {
      if (!((1 << ch) & channel_mask))
        continue;
      if (outfile_data[ch])
        fclose(outfile_data[ch]);
    }
  }
  printf("Press 0 followed by ENTER to exit.\n");
#ifdef LINUX
  scanf("%d", &exit);
#else
  scanf_s("%d", &exit);
#endif
  return 0;
}
