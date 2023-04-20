/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 *
 * This example demonstrates using ADQ14 or ADQ7 digitizers with firmware options
 * FW2DDC or FW4DDC installed.
 */

#define _CRT_SECURE_NO_WARNINGS // This define removes warnings for printf

#include "ADQAPI.h"
#include "os.h"
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#ifdef LINUX
#include <stdlib.h>
#endif

/* Number of DDCs enabled */
const unsigned int nof_ddcs = 2;

/* Each DDC produces two output data streams, I and Q */
const unsigned int nof_data_channels = 4;

/* Enable data collection for all enabled DDCs */
/* Note: Alternatively, if for example you are only interested in the I output
   of each DDC, set individual bits in channel_mask */
const unsigned int channel_mask = 0xF;

/* Decimation factor (2^decimation_factor) per DDC */
const unsigned int decimation_factor[4] = {8, 8, 0, 0};

/* DDC input data selection per DDC */
/* Note: On ADQ14-FW4DDC, there is no crosspoint switch, each DDC always takes
   its corresponding input channel as real-valued input data */
const unsigned int crosspoint_mode[4] = {0, 0, 0, 0};

/* Mixer frequency per DDC */
/* In units of Hz. An input signal f_in will show up at f_in + f_lo in the baseband spectrum */
const double frequency_lo[4] = {-100.0E6, -200.0E6, 0.0E6, 0.0E6};

/* Equalizer setup - defaults to bypassed */
const unsigned int eqmode = 0; // 0 = bypass equalizer, 1 = real-valued equalizer, 2 = complex valued equalizer
float eqcoeffs1[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float eqcoeffs2[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/* Set to 2 for 16-bit data, 4 for 32-bit data */
const unsigned int target_bytes_per_sample = 2;

const unsigned int trigger_mode = ADQ_SW_TRIGGER_MODE;

/* ADQ_WaitForRecordBuffer() will return a number of bytes for each enabled
   channel. It is up to the user to interpret this as a record or a batch and
   handle it accordingly. */
const int number_of_file_batches = 5;

/* Use ADQ_INFINITE_RECORD_LENGTH for continuous streaming */
/* If discrete record length is used instead of ADQ_INFINITE_RECORD_LENGTH, make
   sure to set the parameter record_buffer_size_max at least the same size as the
   record length in bytes. */
const int record_length = ADQ_INFINITE_RECORD_LENGTH;

/* Adjust the timeout accordingly depending on the decimation factor, trigger
   mode and transfer-buffer size. For finite record length, large decimation factor
   and large transfer-buffer size in combination with software trigger mode, use a
   small value for the timeout to get better latency */
const unsigned int timeout_ms = 1000;

const int write_data_to_file = 1; // 1 = write streaming data to file, 0 = don't

int main()
{
  /* Declaration of variables for later user */
  unsigned int success = 1;
  unsigned int nof_adq = 0;
  unsigned int adq_num = 1;
  unsigned int nof_failed_adq;
  char *serial_number;
  char *product_name;
  uint32_t *fw_rev;
  double samplerate = 0;
  double sampleratedec = 0;
  unsigned int ch, ddc;
  int exit = 0;
  int result = 0;
  unsigned int available_ddcs = 0;

  /* Set transfer buffers, buffer size must be a multiple of 1024 */
  unsigned int nof_buffers = 8;
  unsigned int buffer_size = 512*1024;

  char file_name[256];
  FILE* outfile[8] = {NULL};
  int batches_written[8] = {0};
  uint64_t samples_received[8] = { 0 };

  /* Data readout */
  int done;
  struct ADQDataAcquisitionParameters acquisition;
  struct ADQDataTransferParameters transfer;
  struct ADQDataReadoutParameters readout;
  struct ADQDataReadoutStatus status;
  struct ADQRecord *record;

  /* Create a control unit */
  void* adq_cu = CreateADQControlUnit();

  /* Enable Logging */
  // Errors, Warnings and Info messages will be logged
  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");

  /* Find Devices */
  /* We will only connect to the first device in this example */
  nof_adq = ADQControlUnit_FindDevices(adq_cu);
  printf("Number of ADQ devices found: %u\n", nof_adq);
  nof_failed_adq = ADQControlUnit_GetFailedDeviceCount(adq_cu);
  printf("Number of failed ADQ devices: %u\n", nof_failed_adq);

  if (nof_adq == 0)
  {
    printf("\nNo ADQ device found, aborting..\n");
  }
  else
  {
    /* Print product name, serial number and API revision */
    fw_rev = ADQ_GetRevision(adq_cu, adq_num);
    serial_number = ADQ_GetBoardSerialNumber(adq_cu, adq_num);
    product_name = ADQ_GetBoardProductName(adq_cu, adq_num);
    printf("\nAPI revision:        %08x\n", ADQAPI_GetRevision());
    printf("Firmware revision:   %u\n", fw_rev[0]);
    printf("Board serial number: %s\n", serial_number);
    printf("Board product name:  %s\n", product_name);

    if(ADQ_HasFeature(adq_cu, adq_num, "FW2DDC") > 0)
    {
      available_ddcs = 2;
    }

    if(ADQ_HasFeature(adq_cu, adq_num, "FW4DDC") > 0)
    {
      available_ddcs = 4;
    }

    if (available_ddcs == 0)
    {
      printf("\nThe board either does not have a FWxDDC license, or is not running FWxDDC firmware. Aborting.\n");
      DeleteADQControlUnit(adq_cu);
      return 1;
    }
    else if (available_ddcs < nof_ddcs)
    {
      printf("\nThe current firmware supports a maximum of %u DDCs, but %u are enabled. Aborting.\n",
              available_ddcs, nof_ddcs);
      DeleteADQControlUnit(adq_cu);
      return 1;
    }

    /* ======== Acquisition must be set up prior to FWDDC setup ============ */
    /* Initialize data acquisition parameters. */
    success = success && (ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_ACQUISITION, &acquisition) == sizeof(acquisition));

    /* Initialize data transfer parameters. */
    success = success && (ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_TRANSFER, &transfer) == sizeof(transfer));

    /* Initialize data readout parameters. */
    success = success && (ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_READOUT, &readout) == sizeof(readout));

    for (ch = 0; ch < nof_data_channels; ++ch)
    {
      acquisition.channel[ch].horizontal_offset = 0;
      acquisition.channel[ch].trigger_edge = ADQ_EDGE_RISING;
      acquisition.channel[ch].trigger_source = trigger_mode;
      acquisition.channel[ch].record_length = record_length;
      acquisition.channel[ch].nof_records = number_of_file_batches;
      acquisition.channel[ch].bytes_per_sample = target_bytes_per_sample;

      transfer.channel[ch].nof_buffers = nof_buffers;
      transfer.channel[ch].record_buffer_size = buffer_size;

      readout.channel[ch].nof_record_buffers_in_array = 0;
      readout.channel[ch].nof_record_buffers_max = 256;

      /* Start with a small record buffer and let it grow as needed */
      readout.channel[ch].record_buffer_size_increment = 1024;

      if (acquisition.channel[ch].record_length == ADQ_INFINITE_RECORD_LENGTH)
      {
        /* When using infinite record length, set the number of bytes per
           transfer here. For a value of nof_data_channels, keeping the buffer
           size lower than 1/nof_data_channels will yield lower latency */

        readout.channel[ch].incomplete_records_enabled = 1;
        readout.channel[ch].record_buffer_size_max = buffer_size / (2 * nof_data_channels);
      }
      else
      {
        readout.channel[ch].record_buffer_size_max = record_length * target_bytes_per_sample;
      }

    }

    success = success && (ADQ_SetParameters(adq_cu, adq_num, &acquisition) == sizeof(acquisition));
    success = success && (ADQ_SetParameters(adq_cu, adq_num, &transfer) == sizeof(transfer));
    success = success && (ADQ_SetParameters(adq_cu, adq_num, &readout) == sizeof(readout));
    /* =========== End of acquisition Setup ============ */

    /* =========== Setup FWDDC ============ */
    for(ddc = 1; ddc <= nof_ddcs; ddc++)
    {
      /* Set crosspoint mode for each DDC (except on ADQ14 which has no crosspoint switch) */
      if(strcmp(product_name, "ADQ14AC") != 0 && strcmp(product_name, "ADQ14DC") != 0) {
          success = success && ADQ_SetCrosspointSDR(adq_cu, adq_num, ddc, crosspoint_mode[ddc-1]);
      }

      /* Set mixer frequency */
      success = success && ADQ_SetMixerFrequency(adq_cu, adq_num, ddc, frequency_lo[ddc-1]);

      /* Set decimation factor */
      success = success && ADQ_SetChannelDecimation(adq_cu, adq_num, ddc, decimation_factor[ddc-1]);

      /* Setup equalizer */
      success = success && ADQ_SetEqualizerSDR(adq_cu, adq_num, ddc, eqcoeffs1, eqcoeffs2, eqmode);
    }

    /* Set 16b/32b data format. */
    /* NOTE: This must be called after setting the readout parameters in order to
       get the correct data format for FWxDDC */
    // cppcheck-suppress knownConditionTrueFalse
    success = success && ADQ_SetDataFormat(adq_cu, adq_num, (target_bytes_per_sample == 2) ? 0 : 3);

    /* Reset the NCOs and decimation stages of all channels in order to synchronize their phase */
    success = success && ADQ_ForceResynchronizationSDR(adq_cu, adq_num);

    success = success && ADQ_GetSampleRate(adq_cu, adq_num, 0, &samplerate);
    printf("Base sample rate:      %.2f Hz\n", samplerate);
    success = success && ADQ_GetSampleRate(adq_cu, adq_num, 1, &sampleratedec);
    printf("Decimated sample rate: %.2f Hz\n", sampleratedec);

    /*
    Alternatively, if synchronization of multiple FW4DDC digitizers is desired, use
    SetupTimestampSync instead of ForceResynchronizationSDR, along with an external trigger pulse.

    SetupTimestampSync(adq_cu, adq_num, 0, 12);
    DisarmTimestampSync(adq_cu, adq_num);
    ArmTimestampSync(adq_cu, adq_num);
    (send trigger pulse here, after all boards have been armed)

    This will reset the NCOs and decimation stages of all the boards simultaneously
    */

    /* =========== End of FWDDC Setup ============== */


    for (ch = 0; ch < nof_data_channels; ch++)
    {
      if (channel_mask & (1 << ch))
      {
        sprintf(file_name, "./channel_%u.bin", ch + 1);
        outfile[ch] = fopen(file_name, "wb");
        if (outfile[ch] == NULL)
        {
          printf("Failed to open output file '%s'.\n", file_name);
          success = 0;
          break;
        }
      }
    }


    int sw_trigger_counter = 0;

    /* Start streaming. After this, any trigger event matching the selected
       trigger mode will start the streaming */

    printf("\nArming acquisition logic\n");
    success = success && (ADQ_StartDataAcquisition(adq_cu, adq_num) == ADQ_EOK);

    /* Trigger once if ADQ_INFINITE_RECORD_LENGTH is used */
    if (trigger_mode == ADQ_EVENT_SOURCE_SOFTWARE && record_length == ADQ_INFINITE_RECORD_LENGTH)
    {
        ADQ_SWTrig(adq_cu, adq_num);
        sw_trigger_counter++;
    }

    done = 0;
    while (!done && success)
    {
        if(trigger_mode == ADQ_EVENT_SOURCE_SOFTWARE && record_length != ADQ_INFINITE_RECORD_LENGTH && sw_trigger_counter < number_of_file_batches)
        {
            ADQ_SWTrig(adq_cu, adq_num);
            sw_trigger_counter++;
        }

        int channel = ADQ_ANY_CHANNEL;
        int64_t bytes_received = ADQ_WaitForRecordBuffer(adq_cu, adq_num, &channel, (void**)&record, timeout_ms, &status);

        if (bytes_received == 0)
        {
        printf("\nStatus event, flags 0x%08X.\n", status.flags);
        continue;
        }
        else if (bytes_received < 0)
        {
        if (bytes_received == ADQ_EAGAIN)
        {
            if (record_length != ADQ_INFINITE_RECORD_LENGTH && number_of_file_batches != ADQ_INFINITE_NOF_RECORDS)
            {
                printf("Record collected: %d\n", ADQ_GetAcquiredRecords(adq_cu, adq_num));
                if (ADQ_GetAcquiredRecords(adq_cu, adq_num) == (unsigned int)number_of_file_batches)
                {
                    printf("All records acquired, initiating FlushDMA...\n");
                    ADQ_FlushDMA(adq_cu, adq_num);
                }
            }

            continue;
        }
        printf("Waiting for a record buffer failed, code '%" PRIi64 "'.\n", bytes_received);
        success = 0;
        break;
      }

      samples_received[channel] += (bytes_received/ target_bytes_per_sample);

      /* Process the data. */

      /* When infinite record length is used, the data is split into segments of the length returned by ADQ_WaitForRecordBuffer.
         Infinite record length = Seamless transition between each segment in the saved data file.
         Finite record length = Distinct sectioned segments of records can be seen in the data file */

      if (write_data_to_file)
        fwrite(record->data, 1, (size_t)bytes_received, outfile[channel]);

      /* Return the buffer to the API, ONLY AFTER data has been processed or copied. */
      result = ADQ_ReturnRecordBuffer(adq_cu, adq_num, channel, record);
      if (result != ADQ_EOK)
      {
          printf("Failed to return a record buffer, code %d.\n", result);
          break;
      }

      if (record_length == ADQ_INFINITE_RECORD_LENGTH)
      {
          batches_written[channel]++;
          printf("Received %" PRIi64 " samples (Total %" PRIi64 ") for channel %u (%d / %d batches).\n",
                 (bytes_received / target_bytes_per_sample), samples_received[channel],
                 channel + 1, batches_written[channel], number_of_file_batches);
      }
      else
      {
          batches_written[channel] = (unsigned int)(samples_received[channel] / record_length);
          printf("Received %" PRIi64 " samples (Total %" PRIi64 ") for channel %u (%d/%d records).\n",
                 (bytes_received / target_bytes_per_sample), samples_received[channel],
                 channel + 1, batches_written[channel], number_of_file_batches);
      }

      /* Check completion criteria. */
      done = 1;
      for (ch = 0; ch < nof_data_channels; ++ch)
      {
        if (batches_written[ch] < number_of_file_batches)
          done = 0;
      }
    }

    if (done)
    {
      printf("All enabled channels have written all file batches, shutting down stream.\n");
    }

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Warning: Overflow detected in the digitizer datapath.\n");
    }

    switch (ADQ_StopDataAcquisition(adq_cu, adq_num))
    {
    case ADQ_EOK:
    case ADQ_EINTERRUPTED:
      break;
    default:
      success = 0;
      break;
    }
  }

  /** Exit gracefully **/
  DeleteADQControlUnit(adq_cu);

  // Free buffers, close files
  for (ch = 0; ch < nof_data_channels; ch++)
  {
    if (channel_mask & (1 << ch))
    {
      if (outfile[ch] != NULL)
        fclose(outfile[ch]);
    }
  }

  if (success == 0)
    printf("\nAn error occurred, please view the trace logs for more information\n");

  printf("\nType 0 and ENTER to exit.\n");
  scanf("%d", &exit);
  return 0;
}
