// File: example_SDR14.cpp
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

void sdr14_streaming(void *adq_cu, int adq_num);
void sdr14_awg_demo(void *adq_cu, int adq_num);
void sdr14_multirecordexample(void *adq_cu, int adq_num);

// Special define
#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


void sdr14(void *adq_cu, int adq_num)
{
  int mode;
  int* revision = ADQ_GetRevision(adq_cu, adq_num);
  printf("\nConnected to SDR14 #1\n\n");

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

  ADQ_SetDirectionGPIO(adq_cu, adq_num,31,0);
  ADQ_WriteGPIO(adq_cu, adq_num, 31,0);

    printf("\nChoose collect mode.\n 1 = Multi-Record\n 2 = Streaming\n 3 = AWG Demo\n\n\n");
  scanf("%d", &mode);

  switch (mode)
  {
  case 1:
    sdr14_multirecordexample(adq_cu, adq_num);
    break;
  case 2:
    sdr14_streaming(adq_cu, adq_num);
    break;
  case 3:
    sdr14_awg_demo(adq_cu, adq_num);
    break;
  default:
    return;
    break;
  }


}

void sdr14_multirecordexample(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  unsigned int samples_per_record;
  unsigned int number_of_records;
  unsigned int buffersize;
  unsigned char channelsmask;
  unsigned int maskinput;
  unsigned int trig_channel;
  unsigned int write_to_file = 1;
  unsigned int records_to_collect;
  unsigned int i;
  short* buf_a;
  short* buf_b;
  FILE* outfile[4];
  void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers
  unsigned int max_nof_samples = 0;
  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  int exit=0;

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

  outfile[0] = fopen("dataA.out", "w");
  outfile[1] = fopen("dataB.out", "w");
  for(i = 0; i < 2; i++) {
    if(outfile[i] == NULL) {
      printf("Error: Failed to open output files.\n");
      return;
    }
  }

  printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level Trigger Mode\n",
    ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE);
  scanf("%d", &trig_mode);

  switch (trig_mode)
  {
  case ADQ_SW_TRIGGER_MODE : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
    }
  case ADQ_EXT_TRIGGER_MODE : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
    }
  case ADQ_LEVEL_TRIGGER_MODE : {
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
  default :
    return;
    break;
  }

  printf("\nChoose number of records.\n");
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

  channelsmask = 0x00;
  printf("\nEnable channel A data collection? (0 or 1)\n");
  scanf("%u", &maskinput);
  if(maskinput > 0)
    channelsmask |= 0x01;

  printf("\nEnable channel B data collection? (0 or 1)\n");
  scanf("%u", &maskinput);
  if(maskinput > 0)
    channelsmask |= 0x02;

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
    int trigged;
    printf("Automatically triggering your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    }while (trigged == 0);
  }
  else
  {
    int trigged;
    printf("\nPlease trig your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
    }while (trigged == 0);
  }
  printf("\nDevice trigged\n");

  // collect data
  printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
  scanf("%u", &records_to_collect);

  while(records_to_collect > number_of_records)
  {
    printf("\nError: The chosen number exceeds the number of available records in the memory\n");
    printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
    scanf("%u", &records_to_collect);
  }
  if (records_to_collect == 0)
    records_to_collect = number_of_records;

  printf("Collecting data, please wait...\n");

  // Each data buffer must contain enough samples to store all the records consecutively
  buffersize = records_to_collect * samples_per_record;

  buf_a = (short*)calloc(buffersize,sizeof(short));
  buf_b = (short*)calloc(buffersize,sizeof(short));
  if(buf_a == NULL)
    goto error;
  if(buf_b == NULL)
    goto error;

  // Create a pointer array containing the data buffer pointers
  target_buffers[0] = (void*)buf_a;
  target_buffers[1] = (void*)buf_b;

  // Use the GetData function
  CHECKADQ(ADQ_GetData(adq_cu, adq_num,target_buffers,buffersize,sizeof(short),0,records_to_collect,channelsmask,0,samples_per_record,ADQ_TRANSFER_MODE_NORMAL));

  // Get pointer to non-streamed unpacked data
  if (write_to_file)
  {
    unsigned int channel;
    for(channel = 0; channel < 2; channel++)
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
  fclose(outfile[1]);

  if(buf_a != NULL)
    free(buf_a);
  if(buf_b != NULL)
    free(buf_b);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

void sdr14_streaming(void *adq_cu, int adq_num)
{
  //Setup ADQ
  unsigned int n_samples_collect;
  unsigned int buffers_filled;
  int collect_result;
  unsigned int samples_to_collect;
  signed short* data_stream_target;
  unsigned int LoopVar;
  FILE* outfile, *outfileBin;

  unsigned int tlocal =ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  int exit=0;

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

  CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num, 2)); // Unpscked 16-bit

  outfile = fopen("data.out", "w");
  outfileBin = fopen("data.bin", "wb");
  if(outfile == NULL || outfileBin == NULL) {
    printf("Error: Failed to open output files.\n");
    return;
  }


  printf("Choose how many samples to collect.\n samples > 0.\n Multiples of 8 recommended.");
  scanf("%u", &n_samples_collect);
  while(n_samples_collect == 0)
  {
    printf("\nError: Invalid number of samples.\n");
    printf("Choose how many samples to collect.\n samples > 0.\n");
    scanf("%u", &n_samples_collect);
  }

    printf("\nSetting up streaming...");
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));
    printf("\nDone.");

  printf("Collecting data, please wait...\n");

    // Created temporary target for streaming data
    data_stream_target = NULL;

  // Allocate temporary buffer for streaming data
  CHECKADQ(data_stream_target = (signed short*)malloc(n_samples_collect*sizeof(signed short)));

    // Start streaming by arming
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  samples_to_collect = n_samples_collect;

  while (samples_to_collect > 0)
  {
    unsigned int samples_in_buffer;
    do
    {
      collect_result = ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled);
      printf("Filled: %2u\n", buffers_filled);

    } while ((buffers_filled == 0) && (collect_result));

    collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
    samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Warning: Streaming Overflow 1!\n");
      collect_result = 0;
    }

    if (collect_result)
    {
      // Buffer all data in RAM before writing to disk, if streaming to disk is need a high performance
      // procedure could be implemented here.
      // Data format is set to 16 bits, so buffer size is Samples*2 bytes
      memcpy((void*)&data_stream_target[n_samples_collect-samples_to_collect],
        ADQ_GetPtrStream(adq_cu, adq_num),
        samples_in_buffer*sizeof(signed short));
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

  for (LoopVar = 0; LoopVar < samples_to_collect; LoopVar+=1)
  {
    int ik;
    for(ik =0; ik<1; ik++)
    {
      //  fprintf(outfile, "%04x", (unsigned short)data_stream_target[LoopVar+ik]);
      fprintf(outfile, "%d", (unsigned short)data_stream_target[LoopVar+ik]);

    }
    fprintf(outfile, "\n");
  }

  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, 0x01, 0x00000000, 0x00000000, NULL)); //Restore mux by clearing user reg 1
  printf("\n\n Done. Samples stored.\n");

error:

  fclose(outfile);
  fclose(outfileBin);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

void sdr14_awg_demo(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  unsigned int samples_per_record;
  unsigned int n_records_collect;
  unsigned int number_of_records;
  unsigned int i;
  unsigned int k;
  unsigned int dac;
  int dac_data[512];
  short* data_ptr_addr[4];
  unsigned int channel;
  FILE* outfile[4], *outfileBin[4];

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  int exit=0;
  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

    // Create triangle wave test-pattern data
    for (k=0;k<128;k++)
    {
        dac_data[k] = 16*k;
    }
    for (k=0;k<128;k++)
    {
        dac_data[128 + k] = 16*(128 - k);
    }
    for (k=0;k<128;k++)
    {
        dac_data[256 + k] = -16*((int) k);
    }
    for (k=0;k<128;k++)
    {
        dac_data[384 + k] = 16*(-128 + k);
    }

  // Two's complement
  for (k = 0; k < 512; k++) {
    if(dac_data[k] < 0) {
      dac_data[k] = dac_data[k] + 16384;
    }
  }

  // Set up AWG
  for (dac = 1; dac <= 2; dac++) {
    CHECKADQ(ADQ_AWGDisarm(adq_cu, adq_num,dac));              // Disarm AWG
    CHECKADQ(ADQ_AWGSegmentMalloc(adq_cu, adq_num,dac,1,32768,0));      // Allocate one segment
    CHECKADQ(ADQ_AWGWriteSegment(adq_cu, adq_num,dac,1,0,4,512,dac_data));  // Write segment, set to four laps
    CHECKADQ(ADQ_AWGEnableSegments(adq_cu, adq_num,dac,1));          // Enable one segment
    //CHECKADQ(ADQ_AWGContinuous(adq_cu, adq_num,dac,1));          // (optional) Enable continuous mode
    //CHECKADQ(ADQ_AWGAutoRearm(adq_cu, adq_num,dac,1));            // (optional) Enable auto-rearm
    //CHECKADQ(ADQ_AWGTrigMode(adq_cu, adq_num,dac,1));            // (optional) Enable trigger-per-segment-lap mode
    CHECKADQ(ADQ_AWGArm(adq_cu, adq_num,dac));
  }

  outfile[0] = fopen("dataA.out", "w");
  outfile[1] = fopen("dataB.out", "w");
  outfileBin[0] = fopen("dataA.bin", "wb");
  outfileBin[1] = fopen("dataB.bin", "wb");
  // Setup Multi record
  for(i = 0; i < 2; i++) {
    if(outfile[i] == NULL || outfileBin[i] == NULL) {
      printf("Error: Failed to open output files.\n");
      return;
    }
  }

   trig_mode = ADQ_SW_TRIGGER_MODE;
  CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
  number_of_records = 1;
  samples_per_record = 16384;
  // Use only multirecord mode for data collection.
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num,number_of_records,samples_per_record));

  if (trig_mode == ADQ_SW_TRIGGER_MODE)
  {
    int trig;
    printf("Automatically triggering your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    do
    {
      trig = ADQ_GetAcquiredAll(adq_cu, adq_num);
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    }while (trig == 0);
  }
  printf("\nDevice trigged\n");
  n_records_collect = 1;
  // Get pointer to non-streamed unpacked data
  for (channel=1; channel<=2; channel++)
    data_ptr_addr[channel-1] = (short*) ADQ_GetPtrData(adq_cu, adq_num, channel);

  printf("Collecting data, please wait...\n");
  for (i=0; i<n_records_collect; i++)
  {
    unsigned int samples_to_collect = samples_per_record;
    while (samples_to_collect > 0)
    {
      int collect_result = ADQ_CollectRecord(adq_cu, adq_num, i);
      unsigned int samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

      if (collect_result)
      {
        for (channel=0; channel<2; channel++) {
          unsigned int j;
          fwrite(data_ptr_addr[channel], sizeof(unsigned short), samples_in_buffer, outfileBin[channel]);
          for (j=0; j<samples_in_buffer; j++)
          {
              fprintf(outfile[channel], "%d\n", data_ptr_addr[channel][j]);
          }
        }
        samples_to_collect -= samples_in_buffer;
      }
      else
      {
        printf("Collect next data page failed!\n");
        samples_to_collect = 0;
        i = n_records_collect;
      }
    }
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  printf("\n\n Done. Samples stored in data.out.\n");

error:

  fclose(outfile[0]);
  fclose(outfile[1]);
  fclose(outfileBin[0]);
  fclose(outfileBin[1]);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}
