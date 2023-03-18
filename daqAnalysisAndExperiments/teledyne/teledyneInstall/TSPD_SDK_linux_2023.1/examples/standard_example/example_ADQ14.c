
// File: example_ADQ14.cpp
// Description: Examples of how to use the ADQAPI.
//
// The example shows how to use different acquisition modes:
//    Triggered Streaming
//    MultiRecord
//    Continuous Streaming
//    Raw Streaming
//    Streaming from FPGA DevKit
//
// Note: The handling of the initialization of the API is located
//       in the file example_ADQAPI.c
//
// Note: These examples are also valid for ADQ12
//

#define _CRT_SECURE_NO_WARNINGS // This define removes warnings for printf

#include "ADQAPI.h"
#include "os.h"
#include <stdio.h>
#include <time.h>

#define PRINT_RECORD_INFO

#ifdef LINUX
  #include <stdlib.h>
  #include <string.h>
  #include <unistd.h>
  extern int _kbhit();
  #define Sleep(interval) usleep(1000*interval)
#else
  #include <conio.h> // For kbhit
#endif

void adq14_multirecordexample(void *adq_cu, int adq_num);
void adq14_continuous_streaming(void *adq_cu, int adq_num);
void adq14_raw_streaming(void *adq_cu, int adq_num);
void adq14_devkit_streaming(void *adq_cu, int adq_num);

#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}

#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void adq14(void *adq_cu, int adq_num)
{
  int mode;
  int* revision = ADQ_GetRevision(adq_cu, adq_num);
  printf("\nConnected to ADQ14 #1\n\n");

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

  // Checking for in-compatible firmware
  if (ADQ_HasFeature(adq_cu, adq_num, "FWATD") == 1)
  {
    printf("ERROR: This device is loaded with FWATD firmware and cannot be used for this example. Please see FWATD examples.\n");
    return;
  }
  if (ADQ_HasFeature(adq_cu, adq_num, "FWPD") == 1)
  {
    printf("ERROR: This device is loaded with FWPD firmware and cannot be used for this example. Please see FWPD examples.\n");
    return;
  }

  ADQ_SetDirectionGPIO(adq_cu, adq_num,31,0);
  ADQ_WriteGPIO(adq_cu, adq_num, 31,0);
  while(1) {
    printf("\nChoose collect mode.\n 1 = Multirecord\n 2 = Triggered streaming\n 3 = Continuous streaming\n 4 = Raw streaming\n 5 = Raw streaming with devkit example\n\n"); //\n 2 = Streaming\n 3 = MicroTCA functionality demo\n 4 = TriggeredStreaming

    scanf("%d", &mode);

    switch (mode)
    {
    case 1:
      adq14_multirecordexample(adq_cu, adq_num);
      break;
    case 2:
      printf("This example is deprecated for the triggered streaming acquisition mode. Refer to "
             "the example 'gen3_streaming' for an updated reference implementation and to the user "
             "guide (20-2465) for its documentation.\n");
      break;
    case 3:
      printf("This example is deprecated for the continuous streaming acquisition mode. Refer to "
             "the example 'gen3_streaming' for an updated reference implementation and to the user "
             "guide (20-2465) for its documentation.\n");
      break;
    case 4:
      adq14_raw_streaming(adq_cu, adq_num);
      break;
    case 5:
      adq14_devkit_streaming(adq_cu, adq_num);
      break;
    default:
      return;
      break;
    }
  }
}

void adq14_multirecordexample(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  unsigned int samples_per_record;
  unsigned int number_of_records;
  unsigned int buffersize;
  unsigned int channel;
  unsigned char channelsmask;
  unsigned int maskinput;
  unsigned int trig_channel;
  unsigned int write_to_file = 1;
  unsigned int records_to_collect;
  unsigned int max_nof_samples = 0;
  // Bias level
  int adjustable_bias = 0; // Codes
  unsigned int success = 1;

  // DBS settings
  unsigned int dbs_nof_inst = 0;
  unsigned char dbs_inst = 0;
  int dbs_bypass = 0;
  int dbs_dc_target = adjustable_bias;
  int dbs_lower_saturation_level = 0;
  int dbs_upper_saturation_level = 0;

  short* buf_a = NULL;
  short* buf_b = NULL;
  short* buf_c = NULL;
  short* buf_d = NULL;
  void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  FILE* outfile[4] = {NULL, NULL, NULL, NULL};
  char *serialnumber;
  int exit=0;

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
         tlocal, tr1, tr2, tr3, tr4);

  serialnumber = ADQ_GetBoardSerialNumber(adq_cu, adq_num);

  printf("Device Serial Number: %s\n",serialnumber);

  printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level Trigger Mode\n",
         ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE);
  scanf("%d", &trig_mode);

  // Setup adjustable bias
  if (ADQ_HasAdjustableBias(adq_cu, adq_num)) {
    for (channel=0; channel<2; channel++) {
      success = ADQ_SetAdjustableBias(adq_cu, adq_num, channel+1, adjustable_bias);
      if (success == 0)
        printf("Failed setting adjustable bias for channel %c.\n","ABCD"[channel]);
      else
        printf("Adjustable bias for channel %c set to %d codes.\n", "ABCD"[channel], adjustable_bias);
    }

    printf("Waiting for bias settling...\n");
    Sleep(1000);
  }

  // Setup DBS
  ADQ_GetNofDBSInstances(adq_cu, adq_num, &dbs_nof_inst);
  for (dbs_inst = 0; dbs_inst < dbs_nof_inst; ++dbs_inst) {
    printf("Setting up DBS instance %u ...\n", dbs_inst);
    success = ADQ_SetupDBS(adq_cu, adq_num,
                           dbs_inst,
                           dbs_bypass,
                           dbs_dc_target,
                           dbs_lower_saturation_level,
                           dbs_upper_saturation_level);
    if (success == 0)
      printf("Failed setting up DBS instance %d.", dbs_inst);
  }
  Sleep(1000);

  switch (trig_mode)
  {
  case ADQ_SW_TRIGGER_MODE : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu,adq_num, trig_mode));
    break;
  }
  case ADQ_EXT_TRIGGER_MODE : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu,adq_num, trig_mode));
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

  printf("\nEnable channel C data collection? (0 or 1)\n");
  scanf("%u", &maskinput);
  if(maskinput > 0)
    channelsmask |= 0x04;

  printf("\nEnable channel D data collection? (0 or 1)\n");
  scanf("%u", &maskinput);
  if(maskinput > 0)
    channelsmask |= 0x08;

  ADQ_GetMaxNofSamplesFromNofRecords(adq_cu, adq_num, number_of_records, &max_nof_samples);
  while((samples_per_record == 0) || (samples_per_record > max_nof_samples))
  {
    printf("\nError: Invalid number of samples.\n");
    printf("\nChoose number of samples per record.\n 1 <= samples <= %u.\n", max_nof_samples);
    scanf("%u", &samples_per_record);
  }

  // Use only multirecord mode for data collection.
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu,adq_num,number_of_records,samples_per_record));

  if (trig_mode == ADQ_SW_TRIGGER_MODE)
  {
    int trigged;
    printf("Issuing software trigger(s).\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu,adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu,adq_num));
    CHECKADQ(ADQ_SWTrig(adq_cu,adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu,adq_num);
      CHECKADQ(ADQ_SWTrig(adq_cu,adq_num));
    }while (trigged == 0);
  }
  else
  {
    int trigged;
    printf("\nPlease trigger your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu,adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu,adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu,adq_num);
    }while (trigged == 0);
  }
  printf("\nAll records triggered\n");

  if (ADQ_GetStreamOverflow(adq_cu,adq_num))     //This part is needed to prevent a lock-up in case of overflow, which can happen very rarely in normal use
  {
    printf("\nData FIFO indicated overflow!\n");
    ADQ_ResetDevice(adq_cu, adq_num,4);
  }

  // collect data
  printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
  scanf("%u", &records_to_collect);

  printf("\nSave collected data to file?\n 0 = Do not save\n 1 = Save in ascii format (slow)\n 2 = Save in binary format (fast)\n");
  scanf("%u", &write_to_file);

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
  buf_c = (short*)calloc(buffersize,sizeof(short));
  buf_d = (short*)calloc(buffersize,sizeof(short));

  // Create a pointer array containing the data buffer pointers
  target_buffers[0] = (void*)buf_a;
  target_buffers[1] = (void*)buf_b;
  target_buffers[2] = (void*)buf_c;
  target_buffers[3] = (void*)buf_d;
  if(buf_a == NULL)
    goto error;
  if(buf_b == NULL)
    goto error;
  if(buf_c == NULL)
    goto error;
  if(buf_d == NULL)
    goto error;


  // Use the GetData function
  CHECKADQ(ADQ_GetData(adq_cu,adq_num,target_buffers,buffersize,sizeof(short),0,records_to_collect,channelsmask,0,samples_per_record,ADQ_TRANSFER_MODE_NORMAL));

  switch (write_to_file)
  {
  case 0 :  //Do not save
    break;
  case 1 :  //Save as ascii
    outfile[0] = fopen("dataA.out", "w");
    outfile[1] = fopen("dataB.out", "w");
    outfile[2] = fopen("dataC.out", "w");
    outfile[3] = fopen("dataD.out", "w");
    for(channel = 0; channel < 4; channel++)
    {
      if (outfile[channel] != NULL)
      {
        unsigned int  j;
        for (j=0; j<buffersize; j++)
        {
          if(channelsmask & (0x01 << channel))
            fprintf(outfile[channel], "%d\n", ((short*)target_buffers[channel])[j]);
        }
      }
      else
      {
        printf("Error: Failed to open output files.\n");
      }
    }
    break;
  case 2 :  //Save as binary
    outfile[0] = fopen("dataA.out", "wb");
    outfile[1] = fopen("dataB.out", "wb");
    outfile[2] = fopen("dataC.out", "wb");
    outfile[3] = fopen("dataD.out", "wb");
    for(channel = 0; channel < 4; channel++)
    {
      if (outfile[channel] != NULL)
      {
        if(channelsmask & (0x01 << channel))
          fwrite((short*)target_buffers[channel], sizeof(short), buffersize, outfile[channel]);
      }
      else
        printf("Error: Failed to open output files.\n");
    }
    break;
  default :
    printf("Error: Unknown format!\n");
    break;
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu,adq_num));

  if (write_to_file > 0)
    printf("\n\nDone. Samples stored in data.out.\n");


 error:

  for(channel = 0; channel < 4; channel++)
  {
    if (outfile[channel] != NULL)
      fclose(outfile[channel]);
  }

  if(buf_a != NULL)
    free(buf_a);
  if(buf_b != NULL)
    free(buf_b);
  if(buf_c != NULL)
    free(buf_c);
  if(buf_d != NULL)
    free(buf_d);


  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

void adq14_raw_streaming(void *adq_cu, int adq_num)
{
  unsigned int n_samples_collect  = 64*1024;
  unsigned int buffers_filled;
  int collect_result;
  unsigned int samples_to_collect;
  signed short* data_stream_target;
  unsigned int LoopVar;
  FILE* outfile = NULL, *outfileBin = NULL;

  unsigned int en_A = 1;
  unsigned int en_B = 0;
  unsigned int en_C = 0;
  unsigned int en_D = 0;

  unsigned int tlocal =ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  int exit=0;
  unsigned int test_pattern = 0; //0 or 2

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
         tlocal, tr1, tr2, tr3, tr4);

  outfile = fopen("data.out", "w");
  outfileBin = fopen("data.bin", "wb");
  if(outfile == NULL || outfileBin == NULL) {
    printf("Error: Failed to open output files.\n");
    return;
  }
  printf("\nSetting up streaming...");
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, 8, 1024));

  //Enable streaming
  CHECKADQ(ADQ_SetSampleSkip(adq_cu, adq_num, 8));
  CHECKADQ(ADQ_SetTestPatternMode(adq_cu,adq_num, test_pattern));
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 2, 1)); //RAW mode
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 3,  1*en_A + 2*en_B + 4*en_C+8*en_D)); //mask
  CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, ADQ_SW_TRIGGER_MODE));


  printf("Collecting data, please wait...\n");

  // Created temporary target for streaming data
  data_stream_target = NULL;

  // Allocate temporary buffer for streaming data
  CHECKADQ(data_stream_target = (signed short*)malloc(n_samples_collect*sizeof(signed short)));

  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));
  CHECKADQ(ADQ_StartStreaming(adq_cu, adq_num));

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

  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));

  // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
  printf("Writing stream data in RAM to disk.\n");

  samples_to_collect = n_samples_collect;

  for (LoopVar = 0; LoopVar < samples_to_collect; LoopVar+=1)
  {
    //  fprintf(outfile, "%04hx\n", data_stream_target[LoopVar]);
    fprintf(outfile, "%d\n", data_stream_target[LoopVar]);
  }

  printf("\n\n Done. Samples stored.\n");

 error:

  if(NULL != outfile)
    fclose(outfile);
  if(NULL != outfileBin)
    fclose(outfileBin);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

void adq14_devkit_streaming(void *adq_cu, int adq_num)
{
  unsigned int n_samples_collect  = 64*1024;
  unsigned int buffers_filled;
  int collect_result;
  unsigned int samples_to_collect;
  signed short* data_stream_target;
  unsigned int LoopVar;
  FILE* outfile = NULL, *outfileBin = NULL;

  //Channel selection. The collected data will be in set of 1024 bytes from each channel.
  unsigned int en_A = 1;
  unsigned int en_B = 0;
  //unsigned int en_C = 0; // Not supported in this example
  //unsigned int en_D = 0; // Not supported in this example

  unsigned int tlocal =ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  int exit=0;

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
         tlocal, tr1, tr2, tr3, tr4);

  outfile = fopen("data.out", "w");
  outfileBin = fopen("data.bin", "wb");
  if(outfile == NULL || outfileBin == NULL) {
    printf("Error: Failed to open output files.\n");
    return;
  }
  printf("\nSetting up streaming...");
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, 8, 1024));

  //Make sure data flow is off in UL 2
  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x11, 0, 0, NULL); // Data valid low
  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x10, ~2, 2, NULL); // Use counter

  //Reset counter in UL 2
  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x10, ~1, 1, NULL);
  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x10, ~1, 0, NULL);

  //Set active channels
  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x10, ~6, 2*en_A+4*en_B, NULL);

  // Set data rate with sample skip. (data valid toggel rate)
  CHECKADQ(ADQ_SetSampleSkip(adq_cu, adq_num, 1600));

  //Enable streaming
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));  // Enable streaming mode
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 1, 0)); // Enable DRAM FIFO
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 2, 1)); // Enable raw streaming withou packet headers
  CHECKADQ(ADQ_SetStreamConfig(adq_cu, adq_num, 3,  1*en_A+2*en_B)); // Set channel mask

  // Created temporary target for streaming data
  data_stream_target = NULL;

  // Allocate temporary buffer for streaming data
  CHECKADQ(data_stream_target = (signed short*)malloc(n_samples_collect*sizeof(signed short)));

  printf("Collecting data, please wait...\n");

  //Abort old transfers and start new
  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));
  CHECKADQ(ADQ_StartStreaming(adq_cu, adq_num));

  //Enable data flow in UL2
  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x11, ~1, 1, NULL);

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
      printf("Warning: Streaming Overflow!\n");
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

  ADQ_WriteUserRegister(adq_cu, adq_num, 2, 0x10, ~2, 0, NULL); // Normal flow

  // Disable streaming
  CHECKADQ(ADQ_StopStreaming(adq_cu, adq_num));
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 0));


  // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
  printf("Writing stream data in RAM to disk.\n");

  samples_to_collect = n_samples_collect;

  for (LoopVar = 0; LoopVar < samples_to_collect; LoopVar+=1)
  {
    //fprintf(outfile, "%04hx\n", data_stream_target[LoopVar]);
    fprintf(outfile, "%d\n", data_stream_target[LoopVar]);
  }

  printf("\n\n Done. Samples stored.\n");

 error:

  if(NULL != outfile)
    fclose(outfile);
  if(NULL != outfileBin)
    fclose(outfileBin);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}
