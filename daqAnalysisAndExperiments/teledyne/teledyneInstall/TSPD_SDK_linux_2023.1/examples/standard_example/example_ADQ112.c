// File: example_ADQ112.cpp
// Description: An example of how to use the ADQ-API.
// Will connect to a single ADQ device and collect a batch of data into
// "data.out" (single channel boards) or "data_A.out" and "data_B.out" (dual channel boards).
// ADQAPI.lib should be linked to this project and ADQAPI.dll should be in the same directory
// as the resulting executable.

#define _CRT_SECURE_NO_WARNINGS

#include "ADQAPI.h"
#include <stdio.h>

#ifndef LINUX
#include <windows.h>
#else
#include <stdlib.h>
#include <string.h>
#endif

void adq112_multi_record(void *adq_cu, int adq_num);
void adq112_streaming(void *adq_cu, int adq_num);
//void adq112_time_stamp(void *adq_cu, int adq_num);

// Special define
#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void adq112(void *adq_cu, int adq_num)
{
  int* revision = ADQ_GetRevision(adq_cu, adq_num);
  int mode;
  printf("\nConnected to ADQ112 #1\n\n");

  //Print revision information

  printf("FPGA 1 Revision: %d, ", revision[3]);

  if (revision[4])
    printf("Local copy");
  else
    printf("SVN Managed");

  printf(", ");

  if (revision[5])
    printf("Mixed Revision");
  else
    printf("SVN Updated");

  printf("\n");

  printf("FPGA 2 Revision: %d, ", revision[0]);

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

  //Select mode

  printf("\nChoose collect mode.\n 0 = Continuous DRAM\n 1 = Multi-Record\n 2 = Streaming\n\n");
  scanf("%d", &mode);

  switch (mode)
  {
  case 1:
    adq112_multi_record(adq_cu, adq_num);
    break;
  case 2:
    adq112_streaming(adq_cu, adq_num);
    break;
  default:
    return;
    break;
  }
}

void adq112_multi_record(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  int clock_source;
  int pll_divider;
  unsigned int number_of_records;
  unsigned int samples_per_record;
  unsigned int buffersize;
  unsigned int n_records_collect;
  int trigged;
  int overflow;
  unsigned int sample;
  void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers
  int exit = 0;

  short* buf_a;

  //file for output storage
  FILE* outfile;
  outfile = fopen("data.out", "w");

  //trigger mode
  printf("\nChoose trig mode.\n 1 = SW Trigger Mode\n 2 = External Trigger Mode\n 3 = Level Trigger Mode\n");
  scanf("%d", &trig_mode);
  switch (trig_mode)
  {
  case 1 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
           }
  case 2 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
           }
  case 3 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n -2048 <= level <= 2047\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    printf("\n");
    break;
           }
  default :
    return;
    break;
  }
  //set clock source
  printf("\nChoose clock source.\n 0 = Internal clock, internal reference\n 1 = Internal clock, external reference\n 2 = External clock\n");
  scanf("%d", &clock_source);
  if ((clock_source == 0) || (clock_source == 1))
  {
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));
    printf("\nChoose PLL frequency divider.\n 2 <= divider <= 20, f_clk = 800MHz/divider\n");
    scanf("%d", &pll_divider);
    CHECKADQ(ADQ_SetPllFreqDivider(adq_cu, adq_num, pll_divider));
  }
  else
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));

  //setup multi records
  printf("\nChoose number of records.\n");
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num, number_of_records, samples_per_record));

  if (trig_mode == 1) //SW Trigger Mode
  {
    int trig;
    printf("Automatically triggering your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num)); //Trig multiple times until all records are triggered
      trig = ADQ_GetAcquiredAll(adq_cu, adq_num);
    }while (trig == 0);
  }
  else // External & Level
  {
    printf("\nPlease trig your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
    }while (trigged == 0);
  }
  printf("\nDevice trigged\n");

  printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
  scanf("%u", &n_records_collect);

  while(n_records_collect > number_of_records)
  {
    printf("\nError: The chosen number exceeds the number of available records in the memory\n");
    printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
    scanf("%u", &n_records_collect);
  }

  if (n_records_collect == 0)
    n_records_collect = number_of_records;

  // Each data buffer must contain enough samples to store all the records consecutively
  //buffersize = number_of_records * samples_per_record;          //This row can be good to have for debugging purpose but the row below is better
  buffersize = n_records_collect * samples_per_record;            //Only create a buffer with a size that is enough for the number of records you want to collect and NOT for ALL the availible records
  buf_a = (short*)calloc(buffersize,sizeof(short));
  if(buf_a == NULL)
    goto error;

  // Create a pointer array containing the data buffer pointers
  target_buffers[0] = (void*)buf_a;

  CHECKADQ(ADQ_GetData(adq_cu, adq_num,target_buffers,buffersize,sizeof(short),0,n_records_collect, 1,0,samples_per_record,ADQ_TRANSFER_MODE_NORMAL));

  for (sample=0; sample<buffersize; sample++)
  {
    fprintf(outfile, "%d\n", ((short*)target_buffers[0])[sample]);
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  // Return ADQ to single-trig mode
  CHECKADQ(ADQ_MultiRecordClose(adq_cu, adq_num));

  printf("\n\n Done. %u samples stored in data.out.\n", n_records_collect*samples_per_record);
  if ((trig_mode == 2) || (trig_mode == 3))
  {
    int trig_point = ADQ_GetTrigPoint(adq_cu, adq_num);
    printf("Trig point: %d \n",trig_point);
  }
  overflow = ADQ_GetOverflow(adq_cu, adq_num);
  if (overflow ==1)
    printf("Sample overflow in batch.\n");

error:

  fclose(outfile);
  if(buf_a != NULL)
    free(buf_a);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  scanf("%d", &exit);
  if (exit == 1)
  {
    trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",level);
    printf("Trig flank:        %d \n",flank);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }
  return;
}

void adq112_streaming(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int clock_source;
  int pll_divider;
  unsigned int n_samples_collect;
  unsigned int samples_to_collect;
  signed short sign_extend;
  signed short* data_stream_target = NULL;
  int i;
  int exit;

  FILE* outfile;
  outfile = fopen("data.out", "w");

  // Unpacked 12 bit data
  CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num,ADQ112_DATA_FORMAT_UNPACKED_12BIT));

  printf("\nChoose clock source.\n 0 = Internal clock, internal reference\n 1 = Internal clock, external reference\n 2 = External clock\n");
  scanf("%d", &clock_source);
  if ((clock_source == 0) || (clock_source == 1))
  {
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));
    printf("\nChoose PLL frequency divider.\n 2 <= divider <= 20, f_clk = 800MHz/divider\n");
    scanf("%d", &pll_divider);
    CHECKADQ(ADQ_SetPllFreqDivider(adq_cu, adq_num, pll_divider));
  }
  else
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));

  printf("Choose how many samples to collect.\n samples > 0.\n Multiples of 8 recommended.\n");
  scanf("%u", &n_samples_collect);
  while(n_samples_collect == 0)
  {
    printf("\nError: Invalid number of samples.\n");
    printf("Choose how many samples to collect.\n samples > 0.\n");
    scanf("%u", &n_samples_collect);
  }

  // Allocate temporary buffer for streaming data
  CHECKADQ(data_stream_target = (signed short*)malloc(n_samples_collect*sizeof(signed short)));

  //Enable streaming bypass of DRAM
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,ADQ112_STREAM_ENABLED));
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  printf("Collecting data, please wait...\n");

  samples_to_collect = n_samples_collect;

  while (samples_to_collect > 0)
  {
    int collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
    unsigned int samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

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

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  // Disable streaming bypass of DRAM
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,0));

  // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
  printf("Writing stream data in RAM to disk.\n");

  // Sign extend data if using unpacked data
  switch(ADQ_GetDataFormat(adq_cu, adq_num))
  {
  case ADQ112_DATA_FORMAT_UNPACKED_12BIT:
    sign_extend = 32-12;
    break;
  default:
    sign_extend = 0;
    break;
  }

  // ADC channel data is interleaved in 8 bytes chunks (4 samples with unpacked 12-bit or 16-bit modes)
  // These samples must be interleaved before writing to file, in order B0 A0 B1 A1.
  // Sample order in raw stream: A0 A1 A2 A3 B0 B1 B2 B3 A4 ...
  samples_to_collect = n_samples_collect;
  while (samples_to_collect>0)
  {
    if (samples_to_collect >= 8)
    {
      for (i=0; i<4; i++)
      {
        fprintf(outfile, "%d\n", (int)data_stream_target[n_samples_collect-samples_to_collect+i] << sign_extend >> sign_extend);
        fprintf(outfile, "%d\n", (int)data_stream_target[n_samples_collect-samples_to_collect+4+i] << sign_extend >> sign_extend);
      }
      samples_to_collect -= 8;
    }
    else
    {
      printf("Not enough samples to generate more interleaved data, try setting multiples of 8 samples when collecting.\n");
      samples_to_collect = 0;
    }
  }

error:

  fclose(outfile);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  scanf("%d", &exit);
  if (exit == 1)
  {
    int  trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  trig_level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",trig_level);
    printf("Trig flank:        %d \n",flank);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }
  return;
}

// This section regards function Time Stamp. De-comment if wanted.
/*void adq112_time_stamp(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  int clock_source;
  int pll_divider;

  unsigned int number_of_records;
  unsigned int samples_per_record;
  unsigned int n_records_collect;

  int trig_time_mode;
  int restart_mode;
  int synctrig;

  vector<double> od_timest;
  vector<double> od_timest_cal;

  FILE* outfile;
  outfile = fopen("data.out", "w");

  printf("\nChoose trig mode.\n 1 = SW Trigger Mode\n 2 = External Trigger Mode\n 3 = Level Trigger Mode\n");
  scanf("%d", &trig_mode);
  switch (trig_mode)
  {
  case 1 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
    }
  case 2 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
    }
  case 3 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n -2048 <= level <= 2047\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    printf("\n");
    break;
    }
  default :
    return;
    break;
  }

  printf("\nChoose clock source.\n 0 = Internal clock, internal reference\n 1 = Internal clock, external reference\n 2 = External clock\n");
  scanf("%d", &clock_source);
  if ((clock_source == 0) || (clock_source == 1))
  {
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));
    printf("\nChoose PLL frequency divider.\n 2 <= divider <= 20, f_clk = 800MHz/divider\n");
    scanf("%d", &pll_divider);
    CHECKADQ(ADQ_SetPllFreqDivider(adq_cu, adq_num, pll_divider));
  }
  else
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));

  //set time stamp mode
  printf("\nChoose timestamp mode.\n 0 = Sync off\n 1 = Sync on\n");
  scanf("%d", &trig_time_mode);

  //Set restart mode
  printf("\nSet reset mode. \n 0 = Pulse restart\n 1 = Immediate restart\n");
  scanf("%d", &restart_mode);

  printf("\nChoose number of records.\n");
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

  CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1)); // immediate start at power up
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num, number_of_records, samples_per_record));

  //sync mode set
  if (trig_mode == 1) //SW Trigger Mode
  {
    int trig;
    printf("Automatically triggering your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num)); //Trig multiple times until all records are triggered
      trig = ADQ_GetAcquiredAll(adq_cu, adq_num);
      switch (trig_time_mode)
      {
      case 0 :
        {
          CHECKADQ(ADQ_ReadGPIO(adq_cu, adq_num));
          synctrig = ADQ_ReadGPIO(adq_cu, adq_num);

          CHECKADQ(ADQ_SetTrigTimeMode(adq_cu, adq_num,0));//continious count
          CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
          CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
          CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));

          do{
            CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));
            od_timest.push_back(ADQ_GetTrigTime(adq_cu, adq_num));
          }while (!synctrig);

          do{
            if (restart_mode == 1)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1));
            }
            else if (restart_mode == 0)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,0));
            }
          }while (synctrig);

          break;}

      case 1 :
        {
          CHECKADQ(ADQ_ReadGPIO(adq_cu, adq_num));
          synctrig = ADQ_ReadGPIO(adq_cu, adq_num);

          CHECKADQ(ADQ_SetTrigTimeMode(adq_cu, adq_num,1));//activate sync mode
          CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
          CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

          do{
            CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
            CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
            CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));
            //CHECKADQ(ADQ_GetTrigTimeCycles(adq_cu, adq_num));
            CHECKADQ(ADQ_GetTrigTimeSyncs(adq_cu, adq_num));

            //resetTrigTime
            if (restart_mode == 1)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1));
            }
            else if (restart_mode == 0)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,0));
            }
          }while(synctrig);

          break;}

      default :
        return;
        break;}

    }while (trig == 0);
  }//if-slut

  else // level & external
  {
    printf("\nPlease trig your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    int trigged;
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      switch (trig_time_mode)
      {
      case 0 :
        { //sync off
          CHECKADQ(ADQ_ReadGPIO(adq_cu, adq_num));
          synctrig = ADQ_ReadGPIO(adq_cu, adq_num);
          //synctrig = 1;//test setting

          CHECKADQ(ADQ_SetTrigTimeMode(adq_cu, adq_num,0));//continious count
          CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));

          do{
            CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));
            od_timest.push_back(ADQ_GetTrigTime(adq_cu, adq_num));
          }while (!synctrig);

          //for testing. Else decomment section below.
          //if (synctrig){
          //CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1));
          //}
          do{
            if (restart_mode == 1)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1));
            }
            else if (restart_mode == 0)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,0));
            }
          }while (synctrig);

          break;
        }

        case 1 :
          { //sync on
          CHECKADQ(ADQ_ReadGPIO(adq_cu, adq_num));
          synctrig = ADQ_ReadGPIO(adq_cu, adq_num);
          //synctrig = 1;//test setting

          CHECKADQ(ADQ_SetTrigTimeMode(adq_cu, adq_num,1));//activate sync mode
          CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
          CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

          do{
            CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
            CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
            CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));
            od_timest.push_back(ADQ_GetTrigTime(adq_cu, adq_num));
            //CHECKADQ(ADQ_GetTrigTimeCycles(adq_cu, adq_num));
            CHECKADQ(ADQ_GetTrigTimeSyncs(adq_cu, adq_num));

            //resetTrigTime
            if (restart_mode == 1)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1));
            }
            else if (restart_mode == 0)
            {
            CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,0));
            }
          }while(synctrig);

          break;
          }

      default :
        return;
        break;
      }
    }while (trigged == 0);
  }

  printf("\nDevice trigged\n");

  printf("Choose how many records to collect.\n 1 <= records <= %d, 0 == All in ADQ buffer.\n", number_of_records);
  scanf("%u", &n_records_collect);

  while(n_records_collect > number_of_records)
  {
    printf("\nError: The chosen number exceeds the number of available records in the memory\n");
    printf("Choose how many records to collect.\n 1 <= records <= %d, 0 == All in ADQ buffer.\n", number_of_records);
    scanf("%u", &n_records_collect);
  }
  if (n_records_collect == 0)
    n_records_collect = number_of_records;

  // Get pointer to non-streamed unpacked data
  int* data_ptr_addr0 = ADQ_GetPtrData(adq_cu, adq_num);

  printf("Collecting data, please wait...\n");
  for (unsigned int i=0; i<n_records_collect; i++)
  {
    unsigned int samples_to_collect = samples_per_record;
    while (samples_to_collect > 0)
    {
      int collect_result = ADQ_CollectRecord(adq_cu, adq_num, i);
      unsigned int samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

      if (collect_result)
      {
        for (unsigned int j=0; j<samples_in_buffer; j++)
        {
          vector<double>::const_iterator cii;
          vector<double>::const_iterator cii_cal;
          for(cii=od_timest.begin(); cii!=od_timest.end(); cii++)
          {
            for(cii_cal=od_timest_cal.begin(); cii_cal!=od_timest_cal.end(); cii_cal++)
            {
              //cout << *cii << endl; //print test
              //cout << ((*cii+1 - *cii)/2)<< endl; //print test
              od_timest_cal.push_back((*cii+1 - *cii)/2);
            }
          }
          int data = *(data_ptr_addr0 + j);
          fprintf(outfile, "%d\n", data);
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

  // Return ADQ to single-trig mode
  CHECKADQ(ADQ_MultiRecordClose(adq_cu, adq_num));

  printf("\n\n Done. %u samples stored in data.out.\n", n_records_collect*samples_per_record);
  if ((trig_mode == 2) || (trig_mode == 3))
  {
    int trig_point = ADQ_GetTrigPoint(adq_cu, adq_num);
    printf("Trig point: %d \n",trig_point);
  }
  int overflow = ADQ_GetOverflow(adq_cu, adq_num);
  if (overflow ==1)
    printf("Sample overflow in batch.\n");

error:

  fclose(outfile);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  int exit;
  scanf("%d", &exit);
  if (exit == 1)
  {
    int  trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  trig_level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",trig_level);
    printf("Trig flank:        %d \n",flank);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }
  return;
}*/
