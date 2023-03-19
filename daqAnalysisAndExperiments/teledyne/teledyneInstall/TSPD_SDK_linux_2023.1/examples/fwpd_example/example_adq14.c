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
static int _kbhit()
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

void adq14(void *adq_cu, unsigned int adq_num)
{
  /*
   * Settings
   */
  const unsigned int break_collection_on_key_press_only = 0;

  const unsigned int nof_records[4] = {1, 1, 1, 1};
  const unsigned int record_length[4] = {1024, 1024, 1024, 1024};

  const unsigned int detection_window_length[4] = {1024, 1024, 1024, 1024};
  const unsigned int padding_grid_offset = 1024;

  const unsigned int nof_pretrig_samples[4] = {0 * 8, 0 * 8, 0 * 8, 0 * 8};
  const unsigned int nof_triggerdelay_samples[4] = {0};
  const unsigned int variable_len[4] = {0, 0, 0, 0};
  const char channel_mask = 0x1;

  const unsigned int reduction_factor[4] = {1, 1, 1, 1};
  const unsigned int minimum_frame_length[4] = {1, 1, 1, 1};

  /* Data collection modes:
   * 0: Default mode. Pulse data.
   * 1: Metadata mode with padding
   * 2: Every Nth record
   * 3: Pulse Data with padding (with detection window)
   * 4: Pulse Data with padding (without detection window)
   */
  const unsigned int collection_mode[4] = {0, 0, 0, 0};

  /* Fixed-length padding causes the otherwise dynamic padding behavior (where
   * the output data is always at least equal to 'minimum_frame_length') to
   * instead _always_ output 'minimum_frame_length' additional samples.
   *  0: Disabled
   *  1: Enabled
   */
  const unsigned int fixed_length_padding = 0;

  const unsigned int channel_mux[4] = {0, 1, 2, 3};

  /* Histogram options */
  const unsigned int histogram_tot_scale[4] = {1024, 1024, 1024, 1024};
  const unsigned int histogram_extr_scale[4] = {1024, 1024, 1024, 1024};

  const unsigned int histogram_tot_offset[4] = {0, 0, 0, 0};
  const unsigned int histogram_extr_offset[4] = {0, 0, 0, 0};

  /* Set to 1 to read the histogram bins into data arrays */
  const unsigned int histogram_read_data[4] = {0, 0, 0, 0};

  /*
   * Trigger setting
   * Available modes are:
   * ADQ_LEVEL_TRIGGER_MODE ADQ_EXT_TRIGGER_MODE ADQ_SW_TRIGGER_MODE;
   */
  const unsigned int trigger_mode = ADQ_SW_TRIGGER_MODE;
  const unsigned int padding_trigger_mode = ADQ_SW_TRIGGER_MODE;

  const double ext_trig_threshold = 0.2f; // External trigger threshold in V

  /* Level trigger specific options */
  const int trigger_level[4] = {2000, 2000, 2000, 2000};
  const int reset_hysteresis[4] = {50, 50, 50, 50};
  const int trigger_arm_hysteresis[4] = {50, 50, 50, 50};
  const int reset_arm_hysteresis[4] = {50, 50, 50, 50};

  /* Polarity: 1 - Rising, 0 - Falling */
  const unsigned int trigger_polarity[4] = {1, 1, 1, 1};
  const unsigned int reset_polarity[4] = {0, 0, 0, 0};

  /* Moving average options */
  const unsigned int mavg_bypass = 1;
  const int mavg_constant_level = 0;
  const unsigned int nof_mavg_samples[4] = {0, 0, 0, 0};
  const unsigned int mavg_delay[4] = {0, 0, 0, 0};

  /* Coincidence settings */
  const unsigned int coin_win_len_par = 350;
  const unsigned int coin_window_len[4] = {coin_win_len_par * 4, coin_win_len_par * 4,
                                           coin_win_len_par * 4, coin_win_len_par * 4};
  const unsigned int coin_enable = 0;

  // Bias ADC codes
  const int adjustable_bias = 0;

  // DBS settings
  const int dbs_bypass = 1;
  const int dbs_dc_target = adjustable_bias;
  const int dbs_lower_saturation_level = 0;
  const int dbs_upper_saturation_level = 0;

  /* Program options */
  const int use_synthetic_data = 1;
  const int save_data_to_file = 1;

  /* Synthetic pulse generator settings */
  const int pgen_baseline[4] = {200, 200, 200, 200};
  const int pgen_amplitude[4] = {12000, 12000, 12000, 12000};
  const unsigned int pgen_pulse_period[4] = {64, 64, 64, 64};
  const unsigned int pgen_pulse_width[4] = {16, 16, 16, 16};
  const unsigned int pgen_nof_pulses_in_burst[4] = {20, 20, 20, 20};
  const unsigned int pgen_nof_bursts[4] = {1, 1, 1, 1};
  const unsigned int pgen_burst_period[4] = {10000, 10000, 10000, 10000};

  /* Internal pulse generator modes
   * 0: Bypass
   * 1: Regular pulse output (no PRBS)
   * 2: PRBS width
   * 3: PRBS amplitude
   * 4: PRBS width & amplitude
   */
  unsigned int pgen_mode = 1;

  /* Pulse genrator trig mode should not be used with level trigger */
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

  /* Trigger blocking, supports external trigger input and synchronization input
     Modes:
       - 0: Once
       - 1: Window
       - 2: Gate
       - 3: Inverse window */
  const int trigger_blocking_en = 0;
  const int trigger_blocking_mode = 0;
  const int trigger_blocking_source = ADQ_EXT_TRIGGER_MODE;
  const int trigger_blocking_window_length = 1024;

  /* Timestamp synchronization
     Modes:
      - 0: First event
      - 1: Every event */
  const int timestamp_sync_en = 0;
  const int timestamp_sync_mode = 0;
  const int timestamp_sync_source = ADQ_EXT_TRIGGER_MODE;

  /*
   * Variables and data arrays
   */

  /* Iterators and status variables */
  unsigned int success = 1;

  unsigned int ch = 0;
  unsigned int nof_channels = 0;
  unsigned int nof_adc_cores = 0;
  unsigned int i = 0;

  /* DBS variables */
  unsigned char dbs_inst = 0;
  unsigned int dbs_nof_inst = 0;

  /* Collection status */
  unsigned int records_completed[4] = {0};
  unsigned int samples_added[4] = {0};
  unsigned int headers_added[4] = {0};
  unsigned int header_status[4] = {0};
  unsigned int nof_records_sum = 0;

  // DRAM analysis
  unsigned int show_dram_status = 0;
  unsigned int dram_fill;
  unsigned int dram_peak_fill;

  /* Histogram data arrays */
  unsigned int *histogram_extr_data[4] = {NULL, NULL, NULL, NULL};
  unsigned int *histogram_tot_data[4] = {NULL, NULL, NULL, NULL};
  unsigned int histogram_extr_overflow_bin[4] = {0, 0, 0, 0};
  unsigned int histogram_tot_overflow_bin[4] = {0, 0, 0, 0};
  unsigned int histogram_extr_underflow_bin[4] = {0, 0, 0, 0};
  unsigned int histogram_tot_underflow_bin[4] = {0, 0, 0, 0};
  unsigned int histogram_extr_count[4] = {0, 0, 0, 0};
  unsigned int histogram_tot_count[4] = {0, 0, 0, 0};
  unsigned int histogram_extr_status[4] = {0, 0, 0, 0};
  unsigned int histogram_tot_status[4] = {0, 0, 0, 0};

  // Transfer settings
  unsigned int nof_transfer_buffers = 8;
  unsigned int transfer_buffer_size = 512 * 1024;
  unsigned int transfer_timeout_ms = 60000;

  // Output file
  FILE *data_files[4] = {NULL, NULL, NULL, NULL};
  FILE *header_files[4] = {NULL, NULL, NULL, NULL};
  char file_name[256];

  unsigned int nof_pad_records = 0;
  unsigned int nof_records_max;
  unsigned int fwpd_generation = 0;
  unsigned char *expr_array = NULL;

  /* Data readout */
  int result;
  int done;
  int channel;
  int64_t bytes_received;
  struct ADQDataReadoutParameters readout;
  struct ADQDataReadoutStatus status;
  struct ADQRecord *record;

  /* This example only supports generation 2 of FWPD. */
  printfw("Verifying FWPD generation 2..");
  if (!ADQ_PDGetGeneration(adq_cu, adq_num, &fwpd_generation))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  if (fwpd_generation != 2)
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  /* Get the total number of records, not including padding records */
  for (ch = 0; ch < 4; ++ch)
  {
    if (!((1 << ch) & channel_mask))
      continue;
    nof_records_sum += nof_records[ch];
  }

  // Some variable verification
  if (use_synthetic_data && pgen_trig_mode_en && trigger_mode == ADQ_LEVEL_TRIGGER_MODE)
  {
    printfw("Can not use pulse generator trig mode and level trigger.");
    printfwrastatus("ERR", 2);
    goto close_files_exit;
  }

  // Get number of channels
  nof_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  printfw("Reading the number of ADC cores...\n");
  if (!ADQ_GetNofAdcCores(adq_cu, adq_num, &nof_adc_cores))
  {
    printfwrastatus("FAILED", 2);
    goto close_files_exit;
  }
  printfwrastatus("OK", 1);

  // Truncate output files.
  if (save_data_to_file)
  {
    printfw("Creating output files...");
    for (ch = 0; ch < nof_channels; ch++)
    {
      if (!((1 << ch) & channel_mask))
        continue;
      if (collection_mode[ch] == 1)
        sprintf(file_name, "metadata_ch_%c_adq14.bin", "ABCD"[ch]);
      else
        sprintf(file_name, "data_ch_%c.bin", "ABCD"[ch]);
      data_files[ch] = fopen(file_name, "wb");
      sprintf(file_name, "headers_ch_%c.bin", "ABCD"[ch]);
      header_files[ch] = fopen(file_name, "wb");
    }
    printfwrastatus("OK", 1);
  }

  /* Allocate histogram data arrays */
  printfw("Allocating histogram memory...");
  for (ch = 0; ch < nof_channels; ++ch)
  {
    if (histogram_read_data[ch])
    {
      histogram_extr_data[ch] = (unsigned int *)malloc(HISTOGRAM_EXTR_NOF_BINS
                                                       * sizeof(unsigned int));
      histogram_tot_data[ch] = (unsigned int *)malloc(HISTOGRAM_TOT_NOF_BINS
                                                      * sizeof(unsigned int));

      if (histogram_extr_data[ch] == NULL || histogram_tot_data[ch] == NULL)
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
    }
  }
  printfwrastatus("OK", 1);

  /*
   * Digitizer configuration
   */

  /* Enable User ID header from UL2 */
  ADQ_EnableUseOfUserHeaders(adq_cu, adq_num, 0, 0);

  // Set Clock Source
  success = ADQ_SetClockSource(adq_cu, adq_num, ADQ_CLOCK_INT_INTREF);
  if (success == 0)
  {
    printf("Clock setup failed, aborting...\n");
    goto cleanup_exit;
  }

  if (trigger_mode == ADQ_EXT_TRIGGER_MODE || padding_trigger_mode == ADQ_EXT_TRIGGER_MODE)
  {
    printfw("Configuring ext. trigger threshold %.3f...", ext_trig_threshold);
    if (!ADQ_SetExtTrigThreshold(adq_cu, adq_num, 1, ext_trig_threshold))
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

  // Set channel mux
  for (ch = 0; ch < nof_channels; ++ch)
  {
    printfw("Setting channel mux  %c -> %c ...", "ABCD"[channel_mux[ch]], "ABCD"[ch]);
    if (!ADQ_PDSetDataMux(adq_cu, adq_num, channel_mux[ch] + 1, ch + 1))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  // Setup histogram
  for (ch = 0; ch < nof_channels; ++ch)
  {
    if (histogram_read_data[ch])
    {
      printfw("Setting up histograms for channel %c...", "ABCD"[ch]);
      /* Setup TOT histogram */
      if (!ADQ_PDSetupHistogram(adq_cu, adq_num, histogram_tot_offset[ch], histogram_tot_scale[ch],
                                0, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }

      /* Setup extreme value histogram */
      if (!ADQ_PDSetupHistogram(adq_cu, adq_num, histogram_extr_offset[ch],
                                histogram_extr_scale[ch], 1, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      printfwrastatus("OK", 1);
    }
  }

  // Enable internal test pattern generation if use_synthetic_data was set
  if (use_synthetic_data)
  {
    printfw("Bypassing gain & offset compensation...");
    for (ch = 0; ch < nof_adc_cores; ch++)
    {
      ADQ_SetGainAndOffset(adq_cu, adq_num, 128 + ch + 1, 1024, 0);
    }
    printfwrastatus("OK", 1);

    // Combine settings into the mode parameter
    pgen_mode |= (pgen_trig_mode_en << 3) | (pgen_noise_en << 4);

    for (ch = 0; ch < nof_channels; ++ch)
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

  // Set up adjustable bias
  if (ADQ_HasAdjustableBias(adq_cu, adq_num))
  {
    for (ch = 0; ch < nof_channels; ch++)
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

  // Set up DBS
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
  Sleep(1000);

  // Set up trigger blocking
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
  else
  {
    printfw("Disabling up trigger blocking...");
    if (!ADQ_SetupTriggerBlocking(adq_cu, adq_num, 4, 0, 0, 0))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  // Set up timestamp synchronization
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

  for (ch = 0; ch < nof_channels; ch++)
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
    if (!ADQ_PDSetupTiming(adq_cu, adq_num, ch + 1, nof_pretrig_samples[ch], nof_mavg_samples[ch],
                           mavg_delay[ch], record_length[ch], nof_records[ch], variable_len[ch]))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);

    printfw("Configuring coincidence on channel %c...", "ABCD"[ch]);
    // Use-case: Open a window starting on an event on channel B. If an event on
    //           channel A and B (already fulfilled) occurs within this window,
    //           events on the channel using this core will be accepted.
    expr_array = calloc((2 << (nof_channels - 1)), sizeof(unsigned char));

    /* DCBA */
    expr_array[3] = 1; /* 0011 */

    success = success
              && ADQ_PDSetupTriggerCoincidenceCore(adq_cu, adq_num, ch, coin_window_len[ch],
                                                   expr_array, 0x2u);
    free(expr_array);

    if (coin_enable)
      success = success && ADQ_PDSetupTriggerCoincidence2(adq_cu, adq_num, ch + 1, ch, ch == 0);
    else
      success = success && ADQ_PDSetupTriggerCoincidence2(adq_cu, adq_num, ch + 1, ch, 0);

    if (!success)
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }
    printfwrastatus("OK", 1);
  }

  printfw("Restarting coincidence...");
  if (!ADQ_PDResetTriggerCoincidence(adq_cu, adq_num))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  for (ch = 0; ch < nof_channels; ++ch)
  {
    printfw("Configuring characterization on channel %c...", "ABCD"[ch]);
    if (!ADQ_PDSetupCharacterization(
          adq_cu, adq_num, ch + 1, (collection_mode[ch] | (fixed_length_padding << 8)),
          reduction_factor[ch], detection_window_length[ch], record_length[ch], padding_grid_offset,
          minimum_frame_length[ch], trigger_polarity[ch], trigger_mode, padding_trigger_mode))
    {
      printfwrastatus("FAILED", 2);
      goto cleanup_exit;
    }

    printfwrastatus("OK", 1);
  }

  if (mavg_bypass)
  {
    printfw("Bypassing moving average filter...");
    if (!ADQ_PDSetupMovingAverageBypass(adq_cu, adq_num, mavg_bypass, mavg_constant_level))
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

  // Initialize streaming

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

  for (ch = 0; ch < nof_channels; ++ch)
  {
    if (histogram_read_data[ch] && nof_records[ch] > 0)
    {

      printfw("Clearing histograms on channel %c ...", 'A' + ch);

      if (!ADQ_PDClearHistogram(adq_cu, adq_num, 0, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      if (!ADQ_PDClearHistogram(adq_cu, adq_num, 1, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }

      printfwrastatus("OK", 1);
    }
  }

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
    for (ch = 0; ch < nof_channels; ++ch)
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

  // Send a software trigger
  if (trigger_mode == ADQ_SW_TRIGGER_MODE)
  {
    nof_records_max = 0;
    for (i = 0; i < nof_channels; ++i)
    {
      if (nof_records[i] > nof_records_max)
        nof_records_max = nof_records[i];
    }

    for (i = 0; i < nof_records_max; ++i)
    {
      printf("Issuing software trigger.. %u/%u\n", i + 1, nof_records[0]);
      ADQ_SWTrig(adq_cu, adq_num);
    }
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
    done = !break_collection_on_key_press_only;
    for (ch = 0; ch < nof_channels; ++ch)
    {
      if (!(channel_mask & (1u << ch)))
        continue;

      if (records_completed[ch] != nof_records[ch])
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

  /* Read histogram data */
  for (ch = 0; ch < nof_channels; ++ch)
  {
    if (histogram_read_data[ch] && nof_records[ch] > 0)
    {
      printfw("Reading histogram data on channel %c...", "ABCD"[ch]);

      if (!ADQ_PDReadHistogram(adq_cu, adq_num, histogram_tot_data[ch], 0, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }

      if (!ADQ_PDReadHistogram(adq_cu, adq_num, histogram_extr_data[ch], 1, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }

      if (!ADQ_PDGetHistogramStatus(adq_cu, adq_num, &(histogram_tot_overflow_bin[ch]),
                                    &(histogram_tot_underflow_bin[ch]), &(histogram_tot_count[ch]),
                                    &(histogram_tot_status[ch]), 0, ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }

      if (!ADQ_PDGetHistogramStatus(adq_cu, adq_num, &(histogram_extr_overflow_bin[ch]),
                                    &(histogram_extr_underflow_bin[ch]),
                                    &(histogram_extr_count[ch]), &(histogram_extr_status[ch]), 1,
                                    ch + 1))
      {
        printfwrastatus("FAILED", 2);
        goto cleanup_exit;
      }
      printfwrastatus("OK", 1);

      printf("Channel [%c]: ToT Count:   %u Overflow: %u Underflow: %u\n", 'A' + ch,
             histogram_tot_count[ch], histogram_tot_underflow_bin[ch],
             histogram_tot_overflow_bin[ch]);
      printf("Channel [%c]: Extr. Count: %u Overflow: %u Underflow: %u\n", 'A' + ch,
             histogram_extr_count[ch], histogram_extr_underflow_bin[ch],
             histogram_extr_overflow_bin[ch]);

      for (i = 0; i < HISTOGRAM_EXTR_NOF_BINS; ++i)
      {
        if (histogram_extr_data[ch][i] != 0)
        {
          printf("Extr. Bin[%u] == %u\n", i, histogram_extr_data[ch][i]);
        }
      }

      for (i = 0; i < HISTOGRAM_TOT_NOF_BINS; ++i)
      {
        if (histogram_tot_data[ch][i] != 0)
        {
          printf("Tot. Bin[%u] == %u\n", i, histogram_tot_data[ch][i]);
        }
      }
    }
  }

cleanup_exit:
  // Exit gracefully.
  if (use_synthetic_data)
  {
    for (ch = 0; ch < nof_channels; ++ch)
    {
      printfw("Disarming pulse generator on ch %c...", "ABCD"[ch]);
      if (!ADQ_EnableTestPatternPulseGenerator(adq_cu, adq_num, ch + 1, 0))
      {
        printfwrastatus("FAILED", 2);
      }
      printfwrastatus("OK", 1);
    }
  }

  for (ch = 0; ch < 4; ++ch)
  {
    if (histogram_extr_data[ch])
      free(histogram_extr_data[ch]);

    if (histogram_tot_data[ch])
      free(histogram_tot_data[ch]);
  }

  printf("%u padding records recieved\n", nof_pad_records);

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
  for (ch = 0; ch < 4; ch++)
  {
    if (!((1 << ch) & channel_mask))
      continue;
    if (data_files[ch] != NULL)
      fclose(data_files[ch]);
    if (header_files[ch] != NULL)
      fclose(header_files[ch]);
  }
  printfwrastatus("OK", 1);

  // Wait for user input
  printf("Press ENTER to exit...\n");
  getchar();
}
