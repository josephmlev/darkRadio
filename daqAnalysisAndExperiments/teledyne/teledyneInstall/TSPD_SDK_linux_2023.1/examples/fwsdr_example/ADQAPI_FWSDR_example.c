// 2019 (C)opyright Teledyne Signal Processing Devices Sweden AB
//
// NOTE: This example is only intended for ADQ14 boards with the firmware option FWSDR (software defined radio) installed
//

// File: ADQAPI_FWSDR_example.cpp
// Description: An example on how to use FWSDR

#define _CRT_SECURE_NO_WARNINGS // This define removes warnings for printf

#include "ADQAPI.h"
#include "os.h"
#include <stdio.h>
#include <inttypes.h>

#ifdef LINUX
#include <stdlib.h>
#endif

/** Parameters for the SDR setup **/
const double frequency_lo = -10.0E6; // Hz. An input frequency f_in will show up at f_in + f_lo

const unsigned int decimation_factor = 5; // Decimate by 2^decimation_factor

const unsigned int eqmode = 0; // 0 = bypass equalizer, 1 = real-valued equalizer, 2 = complex valued equalizer
float eqcoeffs1[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float eqcoeffs2[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/** Parameters for the data collection **/
const unsigned int target_bytes_per_sample = 2; // 2 for 16-bit data, 4 for 32-bit data

const unsigned int trigger_mode = ADQ_SW_TRIGGER_MODE; // ADQ_EXT_TRIGGER_MODE , choose trigger for starting the data stream

unsigned int channel_mask; // Bit mask, choose which channels to enable during streaming

const int write_data_to_file = 1; // 1 = write streaming data to file, 0 = don't
const unsigned int samples_per_file_batch = 1*1024*1024;
const unsigned int number_of_file_batches = 3; // Streaming will stop after (number_of_file_batches * samples_per_file) have been acquired per channel



int main()
{
  /** Declaration of variables for later user **/
  unsigned int success = 1;
  unsigned int nof_adq = 0;
  unsigned int adq_num = 1;
  unsigned int nof_failed_adq;
  char *serial_number;
  char *product_name;
  uint32_t *fw_rev;
  FILE* outfile[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  char file_name[256];
  unsigned int nof_inputchannels;
  unsigned int nof_ddcs;
  unsigned int nof_data_channels = 0;
  unsigned int ch, ddc;
  int exit = 0;
  double samplerate = 0;
  double sampleratedec = 0;

  unsigned int nof_buffers = 8;
  unsigned int buffer_size = 512*1024;

  unsigned int batches_written[8] = {0};

  /* Data readout */
  int done;
  struct ADQDataReadoutParameters readout;
  struct ADQDataReadoutStatus status;
  struct ADQRecord *record;

  /** Create a control unit **/
  void* adq_cu = CreateADQControlUnit();

  /** Enable Logging **/
  // Errors, Warnings and Info messages will be logged
  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");

  /** Find Devices **/
  // We will only connect to the first device in this example
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
    /** Print product name, serial number and API revision **/
    fw_rev = ADQ_GetRevision(adq_cu, adq_num);
    serial_number = ADQ_GetBoardSerialNumber(adq_cu, adq_num);
    product_name = ADQ_GetBoardProductName(adq_cu, adq_num);
    printf("\nAPI revision:        %08x\n", ADQAPI_GetRevision());
    printf("Firmware revision:   %u\n", fw_rev[0]);
    printf("Board serial number: %s\n", serial_number);
    printf("Board product name:  %s\n", product_name);

    if(!(ADQ_HasFeature(adq_cu, adq_num, "FWSDR") > 0)) {
      printf("\nThe board either does not have a FWSDR license, or is not running FWSDR firmware. Aborting.\n");
      DeleteADQControlUnit(adq_cu);
      return 1;
    }

    /** Set up FWSDR **/
    nof_inputchannels = ADQ_GetNofChannels(adq_cu, adq_num);

    // FWSDR for ADQ14 has one DDC per input channel pair
    nof_ddcs = nof_inputchannels / 2;

    // Each DDC produce two output data streams, I and Q
    nof_data_channels = 2 * nof_ddcs;
    // Enable acquisition for all enabled DDCs (or alternatively, if for example only interested in the I output of each DDC, set individual bits in channel_mask)
    channel_mask = (1u << nof_data_channels) - 1;

    for(ddc = 1; ddc <= nof_ddcs; ddc++) {
      success = success && ADQ_SetMixerFrequency(adq_cu, adq_num, ddc, frequency_lo);
      success = success && ADQ_SetEqualizerSDR(adq_cu, adq_num, ddc, eqcoeffs1, eqcoeffs2, eqmode);
    }

    success = success && ADQ_SetSampleDecimation(adq_cu, adq_num, decimation_factor);

    success = success && ADQ_SetDataFormat(adq_cu, adq_num, (target_bytes_per_sample == 2) ? 0 : 3);

    success = success && ADQ_ForceResynchronizationSDR(adq_cu, adq_num); // Reset the NCOs and decimation stages of all channels in order to synchronize them

    success = success && ADQ_GetSampleRate(adq_cu, adq_num, 0, &samplerate);
    printf("Base sample rate:      %.2f Hz\n", samplerate);
    success = success && ADQ_GetSampleRate(adq_cu, adq_num, 1, &sampleratedec);
    printf("Decimated sample rate: %.2f Hz\n", sampleratedec);

    /* Alternatively, if synchronization of multiple FWSDR digitizers is desired, use
       SetupTimestampSync instead of ForceResynchronizationSDR, along with an external trigger pulse.

       SetupTimestampSync(adq_cu, adq_num, 0, 12);
       DisarmTimestampSync(adq_cu, adq_num);
       ArmTimestampSync(adq_cu, adq_num);
       (send trigger pulse here, after all boards have been armed)

       This will reset the NCOs and decimation stages of all the boards simultaneously
    */

    /** Set Trigger **/
    success = success && ADQ_SetTriggerMode(adq_cu, adq_num, trigger_mode);

    /** Set up continuous streaming **/
    success = success && ADQ_ContinuousStreamingSetup(adq_cu, adq_num, channel_mask);
    success = success && ADQ_SetTransferBuffers(adq_cu, adq_num, nof_buffers, buffer_size);

    for (ch = 0; ch < nof_data_channels; ch++)
    {
      if (channel_mask & (1 << ch))
      {
        sprintf(file_name, "./channel_%u_batch0.bin", ch + 1);
        outfile[ch] = fopen(file_name, "wb");
        if (outfile[ch] == NULL)
        {
          printf("Failed to open output file '%s'.\n", file_name);
          success = 0;
          break;
        }
      }
    }

    // Configure data readout.
    success = success && (ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_READOUT, &readout) == sizeof(readout));

    for (ch = 0; ch < nof_data_channels; ++ch)
    {
      readout.channel[ch].incomplete_records_enabled = 1;
      readout.channel[ch].nof_record_buffers_max = 64;
      readout.channel[ch].record_buffer_size_increment = 512 * 1024;
      readout.channel[ch].record_buffer_size_max = 512 * 1024;
    }

    success = success && (ADQ_SetParameters(adq_cu, adq_num, &readout) == sizeof(readout));

    // Start streaming
    // After this, any trigger event matching the selected trigger mode will start the streaming
    printf("\nArming acquisition logic\n");
    success = success && (ADQ_StartDataAcquisition(adq_cu, adq_num) == ADQ_EOK);

    // If trigger mode is set to software trigger, send a single software trigger to start the data flow
    if (trigger_mode == 1)
      ADQ_SWTrig(adq_cu, adq_num);


    done = 0;
    while (!done && success)
    {
      int channel = ADQ_ANY_CHANNEL;
      int64_t result = ADQ_WaitForRecordBuffer(adq_cu, adq_num, &channel, (void**) &record, 1000, &status);

      if (result == 0)
      {
        printf("\nStatus event, flags 0x%08X.\n", status.flags);
        continue;
      }
      else if (result < 0)
      {
        if (result == ADQ_EAGAIN)
        {
          printf("Timeout, initiating a flush.\n");
          ADQ_FlushDMA(adq_cu, adq_num);
          continue;
        }
        printf("Waiting for a record buffer failed, code '%" PRIi64 "'.\n", result);
        success = 0;
        break;
      }

      // Process the data. The data is naturally split into segments of a length
      // equal to the maximum size of a record buffer, i.e.
      // 'readout.channel[ch].record_buffer_size_max'.
      printf("Finished batch %d for channel %d.\n", batches_written[channel]++, channel + 1);
      fwrite(record->data, 1, (size_t) result, outfile[channel]);

      // Check completion criteria.
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
