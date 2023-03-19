/*
 * Copyright 2018 Teledyne Signal Processing Devices Sweden AB
 */

#ifndef LINUX
/* Remove unsafe function warning */
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ADQAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>

#ifdef LINUX
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include <termios.h>
#define Sleep(x) usleep(1000 * x)
#define _getch getchar
int _kbhit()
{
  static const int STDIN = 0;
  static unsigned int initialized = 0;
  int bytes_waiting;

  if (!initialized)
  {
    struct termios term;
    tcgetattr(STDIN, &term);
    term.c_lflag &= ~ICANON;
    tcsetattr(STDIN, TCSANOW, &term);
    setbuf(stdin, NULL);
    initialized = 1;
  }

  ioctl(STDIN, FIONREAD, &bytes_waiting);
  return bytes_waiting;
}

static struct timespec tsref;
static void timer_start(void)
{
  if (clock_gettime(CLOCK_REALTIME, &tsref) < 0)
  {
    printf("\nFailed to start timer.");
    return;
  }
}
static unsigned int timer_time_ms(void)
{
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
  {
    printf("\nFailed to get system time.");
    return -1;
  }
  return (unsigned int)((int)(ts.tv_sec - tsref.tv_sec) * 1000
                        + (int)(ts.tv_nsec - tsref.tv_nsec) / 1000000);
}
#else
#include <windows.h>
#include <conio.h>
static clock_t tref;
static void timer_start(void)
{
  tref = clock();
  if (tref < 0)
  {
    printf("\nFailed to start timer.");
    return;
  }
}
static unsigned int timer_time_ms(void)
{
  clock_t t = clock();
  if (t < 0)
  {
    printf("\nFailed to get system time.");
    return -1;
  }
  return (unsigned int)((float)(t - tref) * 1000.0f / CLOCKS_PER_SEC);
}
#endif

#include "streaming_header.h"
#include "formatter.h"
#define HISTOGRAM_EXTR_NOF_BINS 16384
#define HISTOGRAM_TOT_NOF_BINS 4096

#define NOF_FULL_RATE_CHANNELS_MAX 2
#define NOF_LOW_RATE_CHANNELS_MAX 2

/*
 * The number of processing channels is the sum of the number of full-rate
 * channels (= analog channels) and the number of low-rate channels (= metadata
 * channels). This number is the highest for a device running the 2-channel
 * firmware.
 */
#define NOF_PROCESSING_CHANNELS_MAX (NOF_FULL_RATE_CHANNELS_MAX + NOF_LOW_RATE_CHANNELS_MAX)

/* Shorter defines */
#define NOF_FRC_MAX NOF_FULL_RATE_CHANNELS_MAX
#define NOF_LRC_MAX NOF_FULL_RATE_CHANNELS_MAX
#define NOF_PROCC_MAX NOF_PROCESSING_CHANNELS_MAX

void adq7(void *adq_cu, unsigned int adq_num)
{
  /*
   * Settings
   */

  /*
   * Set to 1 to don't stop the collection when the number of records collected
   * is equal to nof_records. This is can be useful if padding is enabled. Has
   * no effect when running infinite streaming.
   *
   * The data collection loop may be stopped by pressing any key.
   */

  const unsigned int break_collection_on_key_press_only = 0;

  const unsigned int nof_records[NOF_FRC_MAX] = {10, 10};
  const unsigned int record_length[NOF_FRC_MAX] = {10 * 1024, 10 * 1024};

  const unsigned int detection_window_length[NOF_FRC_MAX] = {1024 * 1024 * 1024,
                                                             1024 * 1024 * 1024};
  const unsigned int padding_grid_offset = 1024 * 1024 * 1024;

  const unsigned int nof_pretrig_samples[NOF_FRC_MAX] = {0, 0};
  const unsigned int nof_triggerdelay_samples[NOF_FRC_MAX] = {0, 0};
  const unsigned int variable_len[NOF_FRC_MAX] = {0, 0};

  /*
   * The channel masked used to enable (1) or disable (0) channels. Each bit
   * corresponds to one channel.
   *
   * For 1 channel:
   * bit 0: Channel X
   * bit 1: Metadata channel X
   *
   * For 2 channels:
   * bit 0: Channel A
   * bit 1: Channel B
   * bit 2: Metadata channel A
   * bit 3: Metadata channel B
   */
  const char channel_mask = (1 << 0);

  const unsigned int reduction_factor[NOF_PROCC_MAX] = {1, 1, 1, 1};

  const unsigned int minimum_frame_length[NOF_PROCC_MAX] = {1, 1, 1, 1};

  /*
   * Program options
   *
   * use_synthetic_data:
   *  - 0: Collect data from ADCs
   *  - 1: Use built in pulse generator the generate data internally
   *       (used for testing and debugging)
   *
   * save_data_to_file:
   *  Save data to (binary) files. The data can be plotted using the plot.py
   *  python script.
   *       Data filename:   data_ch_<N>.bin where N = 'A', 'B'
   *       Header filename: headers_ch_<N>.bin where N = 'A', 'B'
   */
  const int use_synthetic_data = 1;
  const int save_data_to_file = 1;

  /* Data collection modes:
   * 0: Default mode. Pulse data.
   * 2: Every Nth record
   * 3: Pulse Data with padding (with detection window)
   * 4: Pulse Data with padding (without detection window)
   */
  const unsigned int collection_mode[4] = {0, 0, 0, 0};

  /*
   * Trigger setting
   * Available modes are:
   * ADQ_LEVEL_TRIGGER_MODE ADQ_EXT_TRIGGER_MODE ADQ_SW_TRIGGER_MODE ADQ_INTERNAL_TRIGGER_MODE;
   */
  const unsigned int trigger_mode = ADQ_SW_TRIGGER_MODE;
  const unsigned int padding_trigger_mode = ADQ_SW_TRIGGER_MODE;

  /* Internal trigger period in samples */
  const unsigned int int_trig_period = 1024 * 1024 * 1024;

  /* External trigger threshold in Volt */
  const double ext_trig_threshold = 0.2f;

  /* Level trigger specific options */
  const int trigger_level[NOF_FRC_MAX] = {10000, 2000};
  const int reset_hysteresis[NOF_FRC_MAX] = {50, 50};
  const int trigger_arm_hysteresis[NOF_FRC_MAX] = {50, 50};
  const int reset_arm_hysteresis[NOF_FRC_MAX] = {50, 50};

  /* Polarity: 1 - Rising, 0 - Falling */
  const unsigned int trigger_polarity[NOF_FRC_MAX] = {1, 1};
  const unsigned int reset_polarity[NOF_FRC_MAX] = {0, 0};

  /* Trigger blocking, supports external trigger input and synchronization
   * input
   * Modes:
   *   - 0: Once
   *   - 1: Window
   *   - 2: Gate
   *   - 3: Inverse window
   */
  const int trigger_blocking_en = 0;
  const int trigger_blocking_mode = 0;
  const int trigger_blocking_source = ADQ_EXT_TRIGGER_MODE;
  const int trigger_blocking_window_length = 1024;

  /* Timestamp synchronization
   * Modes:
   *  - 0: First event
   *  - 1: Every event
   */
  const int timestamp_sync_en = 0;
  const int timestamp_sync_mode = 1;
  const int timestamp_sync_source = ADQ_EXT_TRIGGER_MODE;

  /* Bias ADC codes */
  const int enable_adjustable_bias = 0;
  const int adjustable_bias = 0;

  /* DBS settings */
  const int dbs_bypass = 1;
  const int dbs_dc_target = adjustable_bias;
  const int dbs_lower_saturation_level = 0;
  const int dbs_upper_saturation_level = 0;

  /* Synthetic pulse generator settings */
  const int pgen_baseline[NOF_FRC_MAX] = {200, 200};
  const int pgen_amplitude[NOF_FRC_MAX] = {12000, 12000};
  const unsigned int pgen_pulse_period[NOF_FRC_MAX] = {512, 512};
  const unsigned int pgen_pulse_width[NOF_FRC_MAX] = {128, 128};
  const unsigned int pgen_nof_pulses_in_burst[NOF_FRC_MAX] = {3, 3};
  const unsigned int pgen_nof_bursts[NOF_FRC_MAX] = {1, 1};
  const unsigned int pgen_burst_period[NOF_FRC_MAX] = {3000, 3000};

  /* Internal pulse generator modes
   * 0: Bypass
   * 1: Regular pulse output (no PRBS)
   * 2: PRBS width
   * 3: PRBS amplitude
   * 4: PRBS width & amplitude
   */
  unsigned int pgen_mode = 1;

  /* Pulse generator trig mode should not be used with level trigger */
  const unsigned int pgen_trig_mode_en = 1;
  const unsigned int pgen_noise_en = 0;

  const unsigned int pgen_prbsw_seed = 6738;
  const int pgen_prbsw_offset = 128 - 32;
  const unsigned int pgen_prbsw_scale_bits = 9; // output = offset + [0, 15]

  const unsigned int pgen_prbsh_seed = 5574;
  const int pgen_prbsh_offset = 10000 - 512;
  const unsigned int pgen_prbsh_scale_bits = 5; // output = offset + [0, 511]

  const unsigned int pgen_prbsn_seed = 5574;
  const int pgen_prbsn_offset = 0;
  const unsigned int pgen_prbsn_scale_bits = 7; // output = offset + [0, 511]

  /* Transfer settings */
  const unsigned int nof_transfer_buffers = 4;
  const unsigned int transfer_buffer_size = 2 * 1024 * 1024;
  const unsigned int transfer_timeout_ms = 60000;
  const unsigned int timeout_ms = 1000;

  /* Set to 1 to print header data (e.g. length) for records collected */
  const unsigned int show_records_collected = 1;

  /*
   * Variables and data arrays
   */
  int infinite_streaming = 0;

  /* Iterators and status variables */
  unsigned int ch, source_channel = 0;
  unsigned int nof_full_rate_channels = 0;
  unsigned int i = 0;

  /* DBS variables */
  unsigned char dbs_inst = 0;
  unsigned int dbs_nof_inst = 0;

  /* Collection status */
  unsigned int records_completed[NOF_PROCC_MAX] = {0};
  unsigned int samples_added[NOF_PROCC_MAX] = {0};
  unsigned int headers_added[NOF_PROCC_MAX] = {0};
  unsigned int header_status[NOF_PROCC_MAX] = {0};

  /* DRAM analysis */
  unsigned int show_dram_status = 0;
  unsigned int dram_fill;
  unsigned int dram_peak_fill;

  /* Output files */
  FILE *data_files[NOF_PROCC_MAX] = {NULL, NULL};
  FILE *header_files[NOF_PROCC_MAX] = {NULL, NULL};
  char file_name[256];

  unsigned int nof_pad_records = 0;
  unsigned int nof_records_max;

  /* Data readout */
  int result;
  int done;
  int channel;
  int64_t bytes_received;
  struct ADQDataReadoutParameters readout;
  struct ADQDataReadoutStatus status;
  struct ADQRecord *record;

  /* Some variable verification */
  if (use_synthetic_data && pgen_trig_mode_en && trigger_mode == ADQ_LEVEL_TRIGGER_MODE)
  {
    printfw("Can not use the pulse generator 'trig mode' togheter with the "
            "level trigger.");
    printfwrastatus("ERR", 2);
    goto close_files_exit;
  }

  /* Get number of channels */
  nof_full_rate_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  for (ch = 0; ch < nof_full_rate_channels; ++ch)
  {
    if (nof_records[ch] == (unsigned int)(-1))
    {
      infinite_streaming = 1;
      break;
    }
  }

  if (trigger_mode == ADQ_SW_TRIGGER_MODE && infinite_streaming)
  {
    printfw("Can not use software trigger and infinite streaming");
    printfwrastatus("ERR", 2);
    goto close_files_exit;
  }

  // Truncate output files.
  if (save_data_to_file)
  {
    printfw("Creating output files...");
    for (ch = 0; ch < 4; ch++)
    {
      if (!((1 << ch) & channel_mask))
        continue;
      if (collection_mode[ch] == 1)
        sprintf(file_name, "metadata_ch_%c_adq7.bin", "ABCD"[ch]);
      else
        sprintf(file_name, "data_ch_%c.bin", "ABCD"[ch]);
      data_files[ch] = fopen(file_name, "wb");
      sprintf(file_name, "headers_ch_%c.bin", "ABCD"[ch]);
      header_files[ch] = fopen(file_name, "wb");
    }
    printfwrastatus("OK", 1);
  }

  /*
   * Digitizer configuration
   */

  /*
   * Enable User ID header from UL2. This is required to get the padding and
   * metadata markers in the user ID
   */
  ADQ_EnableUseOfUserHeaders(adq_cu, adq_num, 0, 0);

  /*
   * Set Clock Source
   * Internal reference is used by default. Uncomment this if any other clock
   * source is used.
   */
  // printfw("Setting clock source..");
  // if (!ADQ_SetClockSource(adq_cu, adq_num, ADQ_CLOCK_INT_EXTREF))
  // {
  //   printfwrastatus("FAILED", 2);
  //   goto cleanup_exit;
  // }
  // printfwrastatus("OK", 1);

  if (trigger_mode == ADQ_EXT_TRIGGER_MODE || padding_trigger_mode == ADQ_EXT_TRIGGER_MODE)
  {
    printfw("Configuring ext. trigger threshold %.3f...", ext_trig_threshold);
    if (!ADQ_SetTriggerThresholdVoltage(adq_cu, adq_num, ADQ_EXT_TRIGGER_MODE, ext_trig_threshold))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
    printfw("Configuring ext. trigger rising edge...");
    if (!ADQ_SetTriggerEdge(adq_cu, adq_num, ADQ_EXT_TRIGGER_MODE, 1))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  /* Enable internal test pattern generation if use_synthetic_data was set */
  if (use_synthetic_data)
  {
    printfw("Bypassing gain & offset compensation...");
    for (ch = 0; ch < 4; ch++)
    {
      ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + ch + 1, 1024, 0);
    }
    printfwrastatus("OK", 1);

    /* Combine settings into the mode parameter */
    pgen_mode |= (pgen_trig_mode_en << 3) | (pgen_noise_en << 4);

    for (ch = 0; ch < nof_full_rate_channels; ++ch)
    {
      printfw("Disarming pulse generator on ch %c...", "ABCD"[ch]);
      if (!ADQ_EnableTestPatternPulseGenerator(adq_cu, adq_num, ch + 1, 0))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      printfwrastatus("OK", 1);

      printfw("Configuring pulse generator on channel %c...", "ABCD"[ch]);
      if (!ADQ_SetupTestPatternPulseGenerator(adq_cu, adq_num, ch + 1, pgen_baseline[ch],
                                              pgen_amplitude[ch], pgen_pulse_period[ch],
                                              pgen_pulse_width[ch], pgen_nof_pulses_in_burst[ch],
                                              pgen_nof_bursts[ch], pgen_burst_period[ch],
                                              pgen_mode))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }

      printfwrastatus("OK", 1);
      if (((pgen_mode & 0x7) == 2) || ((pgen_mode & 0x7) == 4))
      {
        printfw("Configuring pulse generator width PRBS on channel %c...", "ABCD"[ch]);
        if (!ADQ_SetupTestPatternPulseGeneratorPRBS(adq_cu, adq_num, ch + 1, 0, pgen_prbsw_seed,
                                                    pgen_prbsw_offset, pgen_prbsw_scale_bits))
        {
          printfwrastatus("FAILED", 2);
          goto cleanup_exit;
        }
        printfwrastatus("OK", 1);
      }
      if (((pgen_mode & 0x7) == 3) || ((pgen_mode & 0x7) == 4))
      {
        printfw("Configuring pulse generator height PRBS on channel %c...", "ABCD"[ch]);
        if (!ADQ_SetupTestPatternPulseGeneratorPRBS(adq_cu, adq_num, ch + 1, 1, pgen_prbsh_seed,
                                                    pgen_prbsh_offset, pgen_prbsh_scale_bits))
        {
          printfwrastatus("FAILED", 2);
          goto cleanup_exit;
        }
        printfwrastatus("OK", 1);
      }
      if (pgen_mode & 0x10)
      {
        printfw("Configuring pulse generator noise PRBS on channel %c...", "ABCD"[ch]);
        if (!ADQ_SetupTestPatternPulseGeneratorPRBS(adq_cu, adq_num, ch + 1, 2, pgen_prbsn_seed,
                                                    pgen_prbsn_offset, pgen_prbsn_scale_bits))
        {
          printfwrastatus("FAILED", 2);
          goto cleanup_exit;
        }
        printfwrastatus("OK", 1);
      }

      printfw("Set pulse generator test pattern mode on channel %c...", "ABCD"[ch]);
      if (!ADQ_SetTestPatternMode(adq_cu, adq_num, 8))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      printfwrastatus("OK", 1);
    }
  }
  else
  {
    printfw("Using analog input data...");
    for (ch = 0; ch < 4; ch++)
    {
      ADQ_SetGainAndOffset(adq_cu, adq_num, ch + 1, 1024, 0);
    }
    if (!ADQ_SetTestPatternMode(adq_cu, adq_num, 0))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  /* Set up adjustable bias */
  if (ADQ_HasAdjustableBias(adq_cu, adq_num) && enable_adjustable_bias)
  {
    for (ch = 0; ch < nof_full_rate_channels; ch++)
    {
      printfw("Configuring adjustable bias on channel %c.", "ABCD"[ch]);
      if (!ADQ_SetAdjustableBias(adq_cu, adq_num, ch + 1, adjustable_bias))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      printfwrastatus("OK", 1);
    }

    printfw("Waiting for bias settling...");
    Sleep(1000);
    printfwrastatus("OK", 1);
  }

  /* Set up DBS */
  ADQ_GetNofDBSInstances(adq_cu, adq_num, &dbs_nof_inst);
  for (dbs_inst = 0; dbs_inst < dbs_nof_inst; ++dbs_inst)
  {
    printfw("Setting up DBS instance %u ...", dbs_inst);
    if (!ADQ_SetupDBS(adq_cu, adq_num, dbs_inst, dbs_bypass, dbs_dc_target,
                      dbs_lower_saturation_level, dbs_upper_saturation_level))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  /* Wait for DBS to settle */
  if (!dbs_bypass)
    Sleep(1000);

  /* Set up trigger blocking */
  if (trigger_blocking_en)
  {
    printfw("Setting up trigger blocking...");
    if (!ADQ_SetupTriggerBlocking(adq_cu, adq_num, trigger_blocking_mode, trigger_blocking_source,
                                  trigger_blocking_window_length, 0))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  /* Set up timestamp synchronization */
  if (timestamp_sync_en)
  {
    printfw("Setting up timestamp synchronization...");
    if (!ADQ_SetupTimestampSync(adq_cu, adq_num, timestamp_sync_mode, timestamp_sync_source))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  for (ch = 0; ch < nof_full_rate_channels; ch++)
  {
    printfw("Configuring level trigger on channel %c...", "ABCD"[ch]);
    if (!ADQ_PDSetupLevelTrig(adq_cu, adq_num, ch + 1, trigger_level[ch], reset_hysteresis[ch],
                              trigger_arm_hysteresis[ch], reset_arm_hysteresis[ch],
                              trigger_polarity[ch], reset_polarity[ch]))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
    printfw("Configuring acquisition parameters on channel %c...", "ABCD"[ch]);
    if (!ADQ_PDSetupTiming(adq_cu, adq_num, ch + 1, nof_pretrig_samples[ch], 0, 0,
                           record_length[ch], nof_records[ch], variable_len[ch]))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  printfw("Setting internal trigger period");
  if (!ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, int_trig_period))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  for (ch = 0; ch < nof_full_rate_channels; ++ch)
  {
    printfw("Configuring characterization on channel %c...", "ABCD"[ch]);

    /* The record length is set up with PDSetupTiming */

    if (!ADQ_PDSetupCharacterization(adq_cu, adq_num, ch + 1, collection_mode[ch],
                                     reduction_factor[ch], detection_window_length[ch], 0,
                                     padding_grid_offset, minimum_frame_length[ch],
                                     trigger_polarity[ch], trigger_mode, padding_trigger_mode))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }

    printfwrastatus("OK", 1);
  }

  printf("\n");
  printfw("Configuring device-to-host interface...");
  if (!ADQ_SetTransferBuffers(adq_cu, adq_num, nof_transfer_buffers, transfer_buffer_size))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  if (!ADQ_SetTransferTimeout(adq_cu, adq_num, transfer_timeout_ms))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  /* Initialize streaming */
  ADQ_DisarmTriggerBlocking(adq_cu, adq_num);
  if (trigger_blocking_en)
  {
    printfw("Arming trigger blocking...");
    if (!ADQ_ArmTriggerBlocking(adq_cu, adq_num))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  ADQ_DisarmTimestampSync(adq_cu, adq_num);
  if (timestamp_sync_en)
  {
    printfw("Arming timestamp synchronization...");
    if (!ADQ_ArmTimestampSync(adq_cu, adq_num))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  printfw("Enabling trigger...");
  if (!ADQ_PDEnableLevelTrig(adq_cu, adq_num, 1))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  printfw("Set up streaming...");
  if (!ADQ_PDSetupStreaming(adq_cu, adq_num, channel_mask))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  printfw("Initializing data readout parameters...");
  result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_READOUT, &readout);
  if (result != sizeof(readout))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  printfw("Configuring data readout parameters...");
  result = ADQ_SetParameters(adq_cu, adq_num, &readout);
  if (result != sizeof(readout))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  /*
   * This prevents an accidental key press during configuration to abort the
   * collection loop.
   */
  while (_kbhit())
  {
    _getch();
  };

  /* Start of data collection */
  printfw("Start data acquisition...");
  if (ADQ_StartDataAcquisition(adq_cu, adq_num) != ADQ_EOK)
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  if (use_synthetic_data)
  {
    for (ch = 0; ch < nof_full_rate_channels; ++ch)
    {
      printfw("Enabling pulse generator on ch %c...", "ABCD"[ch]);
      if (!ADQ_EnableTestPatternPulseGenerator(adq_cu, adq_num, ch + 1, 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      printfwrastatus("OK", 1);
    }
  }

  /* Send a software trigger */
  if (trigger_mode == ADQ_SW_TRIGGER_MODE)
  {
    nof_records_max = 0;
    for (i = 0; i < nof_full_rate_channels; ++i)
    {
      if (nof_records[i] > nof_records_max)
        nof_records_max = nof_records[i];
    }

    for (i = 0; i < nof_records_max; ++i)
    {
      ADQ_SWTrig(adq_cu, adq_num);
    }
    printf("Issuing software trigger.. %u/%u\n", nof_records_max, nof_records_max);
  }

  /* Readout loop */
  done = 0;
  while (!done)
  {
    if (_kbhit())
    {
      printfw("User abort detected...");
      printfwrastatus("ABRT", 2);
      goto stop_streaming_cleanup_exit;
    }

    channel = ADQ_ANY_CHANNEL;
    bytes_received = ADQ_WaitForRecordBuffer(adq_cu, adq_num, &channel, (void **)&record, 1000,
                                             &status);
    if (bytes_received == 0)
    {
      printf("Status event, flags 0x%08X.\n", status.flags);
      continue;
    }
    else if (bytes_received < 0)
    {
      if (bytes_received == ADQ_EAGAIN)
      {
        printf("Timeout, initiating a flush.\n");
        ADQ_FlushDMA(adq_cu, adq_num);
        continue;
      }
      printf("Waiting for a record buffer failed, code '%" PRIi64 "'.\n", bytes_received);
      goto stop_streaming_cleanup_exit;
    }

    /* Process the data */
    printf("Got record %u from channel %d.\n", record->header->RecordNumber, channel);

    /* Handle the received data */
    if (save_data_to_file)
    {
      if (data_files[channel])
      {
        printfw("Writing %ld bytes to file...", bytes_received);
        fwrite((void *)record->data, 1, (size_t)bytes_received, data_files[channel]);
        printfwrastatus("OK", 1);
      }
      if (header_files[channel])
      {
        fwrite((void *)record->header, sizeof(struct ADQRecordHeader), 1, header_files[channel]);
        printfwrastatus("OK", 1);
      }
    }

    result = ADQ_ReturnRecordBuffer(adq_cu, adq_num, channel, record);
    if (result != ADQ_EOK)
    {
      printf("Failed to return a record buffer for channel %d, code '%d'.\n", channel, result);
      goto stop_streaming_cleanup_exit;
    }

    ++records_completed[channel];
    done = !break_collection_on_key_press_only && !infinite_streaming;
    for (ch = 0; ch < NOF_PROCC_MAX; ++ch)
    {
      if (!(channel_mask & (1u << ch)))
        continue;

      source_channel = ch;
      if (ch >= nof_full_rate_channels)
        source_channel = ch - nof_full_rate_channels;
      if (records_completed[ch] != nof_records[source_channel])
        done = 0;
    }
  }

stop_streaming_cleanup_exit:
  printf("\n-----------------\n");
  printfw("Halt streaming...");
  switch (ADQ_StopDataAcquisition(adq_cu, adq_num))
  {
  case ADQ_EOK:
  case ADQ_EINTERRUPTED:
    printfwrastatus("OK", 1);
    break;
  default:
    printfwrastatus("FAILED", 2);
    break;
  }

  if (show_dram_status)
  {
    ADQ_GetWriteCount(adq_cu, adq_num, &dram_fill);
    ADQ_GetWriteCountMax(adq_cu, adq_num, &dram_peak_fill);

    printf("DRAM fill information:\n");
    printf("Current: %.3f MB\n", dram_fill * 128.0f / (1024.0f * 1024.0f));
    printf("Peak:    %.3f MB\n", dram_peak_fill * 128.0f / (1024.0f * 1024.0f));
  }

cleanup_exit:
  /* Exit gracefully. */
  if (use_synthetic_data)
  {
    for (ch = 0; ch < nof_full_rate_channels; ++ch)
    {
      printfw("Disarming pulse generator on ch %c...", "ABCD"[ch]);
      if (!ADQ_EnableTestPatternPulseGenerator(adq_cu, adq_num, ch + 1, 0))
      {
        printfwrastatus("FAILED", 2);
      }
      printfwrastatus("OK", 1);
    }
  }

  printf("%u padding records received\n", nof_pad_records);

  if (trigger_blocking_en)
  {
    printfw("Disarm trigger blocking...");
    if (!ADQ_DisarmTriggerBlocking(adq_cu, adq_num))
      printfwrastatus("FAILED", 2);
    printfwrastatus("OK", 1);
  }

  if (timestamp_sync_en)
  {
    printfw("Disarm timestamp synchronization...");
    if (!ADQ_DisarmTimestampSync(adq_cu, adq_num))
      printfwrastatus("FAILED", 2);
    printfwrastatus("OK", 1);
  }

  printfw("Disabling frame sync generator...");
  if (!ADQ_EnableFrameSync(adq_cu, adq_num, 0))
  {
    printfwrastatus("FAILED", 2);
  }
  printfwrastatus("OK", 1);

  printfw("Disabling internal trigger...");
  if (!ADQ_SetConfigurationTrig(adq_cu, adq_num, 0, 0, 0))
  {
    printfwrastatus("FAILED", 2);
  }
  printfwrastatus("OK", 1);

close_files_exit:
  printfw("Closing files...");
  for (ch = 0; ch < NOF_PROCC_MAX; ch++)
  {
    if (!((1 << ch) & channel_mask))
      continue;
    if (data_files[ch] != NULL)
      fclose(data_files[ch]);
    if (header_files[ch] != NULL)
      fclose(header_files[ch]);
  }
  printfwrastatus("OK", 1);

  /* Wait for user input */
  printf("Press ENTER to exit...\n");
  getchar();
}
