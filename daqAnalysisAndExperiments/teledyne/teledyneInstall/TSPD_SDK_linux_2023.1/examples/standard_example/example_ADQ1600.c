// File: example_ADQ1600.cpp
// Description: An example of how to use the ADQ-API.
// Will connect to a single ADQ device and collect a batch of data into
// "data.out" (single channel boards) or "data_A.out" and "data_B.out" (dual channel boards).
// ADQAPI.lib should be linked to this project and ADQAPI.dll should be in the same directory
// as the resulting executable.

#define _CRT_SECURE_NO_WARNINGS

#include "ADQAPI.h"
#include <assert.h>
#include <stdio.h>

#ifndef LINUX
#include <windows.h>
#else
#include <stdlib.h>
#include <string.h>
#endif

void adq1600_multirecord(void *adq_cu, int adq_num);
void adq1600_microtca_functionality_demo(void *adq_cu, int adq_num);
void adq1600_streaming(void *adq_cu, int adq_num);

// Special define
#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void adq1600(void *adq_cu, int adq_num)
{
    int mode = 1;
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    printf("\nConnected to ADQ1600 #1\n\n");

    //Print revision information

    printf("FPGA Revision: %d, ", revision[0]);
    if (revision[1])
        printf("Local copy");
    else
        printf("SVN Managed");
    printf(", ");
    if (revision[2])
        printf("Mixed Revision");
    else
        printf("SVN Updated");
    printf("\n\n");

    ADQ_SetDirectionGPIO(adq_cu, adq_num, 31, 0);
    ADQ_WriteGPIO(adq_cu, adq_num, 31, 0);

    printf("\nChoose collect mode.\n 1 = Multi-Record\n 2 = Streaming\n 3 = MicroTCA functionality demo\n\n\n");
    scanf("%d", &mode);

    switch (mode)
    {
        case 1:
          adq1600_multirecord(adq_cu, adq_num);
          break;
        case 2:
          adq1600_streaming(adq_cu, adq_num);
          break;
        case 3:
          adq1600_microtca_functionality_demo(adq_cu, adq_num);
          break;
        default:
            return;
            break;
    }

}

void adq1600_multirecord(void *adq_cu, int adq_num)
{
    //Setup ADQ
    int trig_mode;
    int trig_level;
    int trig_flank;
    unsigned int samples_per_record;
    unsigned int n_records_collect;
    unsigned int number_of_records;
    unsigned int buffersize;
    unsigned int channelsmask = 0xf;
    unsigned int write_to_file = 1;
    short* buf_a;
    unsigned int max_nof_samples = 0;
    int trigged;
    unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0) / 256;
    unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1) / 256;
    unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2) / 256;
    unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3) / 256;
    unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4) / 256;
    void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers
    unsigned int channel;
    FILE* outfile[4];
    int exit = 0;
    printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n", tlocal, tr1, tr2, tr3, tr4);

    outfile[0] = fopen("dataA.out", "w");
    if(outfile[0] == NULL)
    {
      printf("Error: Failed to open output files.\n");
      return;
    }

    trig_mode = ADQ_SW_TRIGGER_MODE;
    printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level Trigger Mode\n",
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
            printf("\nChoose trig level.\n 0 <= level <= 255\n");
            trig_level = 127;
            scanf("%d", &trig_level);
            CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
            printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
            trig_flank = 1;
            scanf("%d", &trig_flank);
            CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
            printf("\n");
            break;
        }
        default:
            return;
            break;
    }

    printf("\nChoose number of records.\n");
    number_of_records = 1;
    scanf("%u", &number_of_records);

    samples_per_record = 4800;
    printf("\nChoose number of samples per record.\n");

    scanf("%u", &samples_per_record);

    ADQ_GetMaxNofSamplesFromNofRecords(adq_cu, adq_num, number_of_records, &max_nof_samples);
    while((samples_per_record == 0) || (samples_per_record > max_nof_samples))
    {
      printf("\nError: Invalid number of samples.\n");
      printf("\nChoose number of samples per record.\n 1 <= samples <= %u.\n", max_nof_samples);
      scanf("%u", &samples_per_record);
    }

    // Use only multirecord mode for data collection.
    CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num,number_of_records,samples_per_record));
    if (trig_mode == ADQ_SW_TRIGGER_MODE)
    {
        printf("Automatically triggering your device to collect data.\n");
        CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
        CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
        do
        {
            CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
            trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);

        } while (trigged == 0);
    }
    else
    {
        printf("\nPlease trig your device to collect data.\n");
        CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
        CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
        do
        {
            trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
        } while (trigged == 0);
    }
    printf("\nDevice trigged\n");

    n_records_collect = 0;

    while (n_records_collect > number_of_records)
    {
        printf("\nError: The chosen number exceeds the number of available records in the memory\n");
        printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
        scanf("%u", &n_records_collect);
    }
    if (n_records_collect == 0)
        n_records_collect = number_of_records;

    // Get pointer to non-streamed unpacked data (Only one channel in ADQ1600)

    printf("Collecting data, please wait...\n");
      // Each data buffer must contain enough samples to store all the records consecutively

    buffersize = n_records_collect * samples_per_record;

    buf_a = (short*)calloc(buffersize,sizeof(short));
    if(buf_a == NULL)
      goto error;

    // Create a pointer array containing the data buffer pointers
    target_buffers[0] = (void*)buf_a;

    // Use the GetData function
    CHECKADQ(ADQ_GetData(adq_cu, adq_num,target_buffers,buffersize,sizeof(short),0,n_records_collect,channelsmask,0,samples_per_record,ADQ_TRANSFER_MODE_NORMAL));

    // Get pointer to non-streamed unpacked data
    if (write_to_file)
    {
      for( channel = 0; channel < 1; channel++)
      {
        unsigned int j;
        for (j=0; j<buffersize; j++)
        {
          if(channelsmask & (0x01 << channel))
            fprintf(outfile[channel], "%d\n", ((short*)target_buffers[channel])[j]);
        }
      }
    }

    // Only disarm trigger after data is collected
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

    printf("\n\n Done. Samples stored in data.out.\n");

    error:

    fclose(outfile[0]);
    if(buf_a != NULL)
      free(buf_a);

    printf("Press 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);

    return;
}

void adq1600_streaming(void *adq_cu, int adq_num)
{
    //Setup ADQ
    unsigned int n_samples_collect;
    unsigned int buffers_filled;
    int collect_result;
    unsigned int samples_to_collect;
    signed short* data_stream_target = NULL;
    unsigned int retry = 0;
    unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0) / 256;
    unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1) / 256;
    unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2) / 256;
    unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3) / 256;
    unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4) / 256;
    unsigned int LoopVar;
    unsigned int ik;
    unsigned int samples_in_buffer;
    FILE* outfile, *outfileBin;
    int exit = 0;

    printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n", tlocal, tr1, tr2, tr3, tr4);

    CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num, 2));
    // Unpscked 16-bit

    outfile = fopen("data.out", "w");
    outfileBin = fopen("data.bin", "wb");

    printf("Choose how many samples to collect.\n samples > 0.\n Multiples of 8 recommended.");
    scanf("%u", &n_samples_collect);
    while (n_samples_collect == 0)
    {
        printf("\nError: Invalid number of samples.\n");
        printf("Choose how many samples to collect.\n samples > 0.\n");
        scanf("%u", &n_samples_collect);
    }

    printf("\nSetting up streaming...");
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));
    printf("\nDone.");

    printf("Collecting data, please wait...\n");

    // Allocate temporary buffer for streaming data
    CHECKADQ(data_stream_target = (signed short*)malloc(n_samples_collect*sizeof(signed short)));

    // Start streaming by arming
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

    samples_to_collect = n_samples_collect;

    while (samples_to_collect > 0)
    {
        do
        {
            collect_result = ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled);
            printf("Filled: %2u\n", buffers_filled);
            retry += 1;
            printf("Retry: %2u\n", retry);

        } while ((buffers_filled == 0) && (collect_result) && (retry < 10));

        collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
        samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

        if (ADQ_GetStreamOverflow(adq_cu, adq_num))
        {
            printf("Warning: Streaming Overflow 1!\n");
            //collect_result = 0;
        }

        if (collect_result)
        {
            // Buffer all data in RAM before writing to disk, if streaming to disk is need a high performance
            // procedure could be implemented here.
            // Data format is set to 16 bits, so buffer size is Samples*2 bytes
            memcpy((void*) &data_stream_target[n_samples_collect - samples_to_collect], ADQ_GetPtrStream(adq_cu, adq_num),
                    samples_in_buffer * sizeof(signed short));
            samples_to_collect -= samples_in_buffer;
        }
        else
        {
            printf("Collect next data page failed!\n");
            samples_to_collect = 0;
        }
    }

    // Only disarm trigger after data is collected
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

    // Disable streaming bypass of DRAM
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,0));

    // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
    printf("Writing stream data in RAM to disk.\n");

    samples_to_collect = n_samples_collect;

    for (LoopVar = 0; LoopVar < samples_to_collect; LoopVar += 1)
    {
        for (ik = 0; ik < 1; ik++)
        {
            //  fprintf(outfile, "%04x", (unsigned short)data_stream_target[LoopVar+ik]);
            fprintf(outfile, "%d", (unsigned short) data_stream_target[LoopVar + ik]);

        }
        fprintf(outfile, "\n");
    }

    CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, 0x01, 0x00000000, 0x00000000, NULL)); // Clear user register 1

    //Restore mux
    printf("\n\n Done. Samples stored.\n");

    error:

    fclose(outfile);
    fclose(outfileBin);

    printf("Press 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);

    return;
}



void adq1600_microtca_functionality_demo(void *adq_cu, int adq_num)
{
  int exit=0;
  printf("\n\nThis example sets up the additional hardware present on MicroTCA ADQ boards.\n\n");
  printf("Running this example on ADQs with other interfaces (USB/PCIE/PXIE) will \ncause the functions to return errors.\n\n");

  printf("Setting MTCA board PLLs to: \n\n");

  printf("10G Ethernet clock:   156.25 MHz\n");
  printf("1G Ethernet clock:    125 MHz\n");
  printf("Point-to-point clock: 250 MHz\n\n");

  /* Simple PLL setup: */

  CHECKADQ(ADQ_SetEthernetPllFreq(adq_cu, adq_num, ETH10_FREQ_156_25MHZ, ETH1_FREQ_125MHZ));  // Set 10G ethernet clocks to 156.25 MHz and 1G clocks to 125 MHz
  CHECKADQ(ADQ_SetPointToPointPllFreq(adq_cu, adq_num, PP_FREQ_250MHZ));                     // Set point-to-point interface clocks to 250 MHz

  /* Advanced PLL setup: */

  // Useref2 = 0        => Internal 10 MHz reference used
  // Refdiv = 0         => 10 MHz reference
  // N = A + B*P        = 10 + 15*16 = 250
  // VCO freq           = N * 10 MHz = 2500 MHz
  // vcooutdiv = 2      => VCO output freq = 1250 MHz
  // pp_outdiv = 5      => point-to-point clock output = 1250/8 = 156.25 MHz
  // ppsync_outdiv = 5  => eth1g point-to-point synched clock output = 1250/10 = 125 MHz
  CHECKADQ(ADQ_SetEthernetPll(adq_cu, adq_num, 1, 0, 10, 15, 16, 2, 8, 10));

  // Useref2 = 0        => Internal 10 MHz reference used
  // Refdiv = 0         => 10 MHz reference
  // N = A + B*P        = 10 + 15*16 = 250
  // VCO freq           = N * 10 MHz = 2500 MHz
  // vcooutdiv = 2      => VCO output freq = 1250 MHz
  // pp_outdiv = 5      => point-to-point clock output = 1250/5 = 250 MHz
  // ppsync_outdiv = 5  => eth1g point-to-point synched clock output = 1250/5 = 250 MHz
  CHECKADQ(ADQ_SetPointToPointPll(adq_cu, adq_num, 1, 0, 10, 15, 16, 2, 5, 5));

  printf("Setting M-LVDS interface direction to all inputs.\n");
  CHECKADQ(ADQ_SetDirectionMLVDS(adq_cu, adq_num, 0x00)); // Set direction of the eight LVDS backplane signals

  printf("\n\n Done. All MTCA-specific API functionality has been initialized.\n");

error:

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}
