// File: example_ADQ214.cpp
// Description: An example of how to use the ADQ-API.
// Will connect to a single ADQ device and collect a batch of data into
// "data.out" (single channel boards) or "data_A.out" and "data_B.out" (dual channel boards).
// ADQAPI.lib should be linked to this project and ADQAPI.dll should be in the same directory
// as the resulting executable.

#define _CRT_SECURE_NO_WARNINGS

#include "ADQAPI.h"
#include <stdio.h>
#include <time.h>

#ifndef LINUX
#include <windows.h>
#else
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#define Sleep(interval) usleep(1000*interval)
#endif

//#ifdef _DEBUG
//  #define scanf(a, b) noscanf()
//#endif
//
//void noscanf()
//{
//    //Dummy function
//}

//using namespace std;

void adq214_multi_record(void *adq_cu, int adq_num);
void adq214_streaming(void *adq_cu, int adq_num);
void wfa_manual_rearm(void *adq_cu, int adq_num);
void wfa_auto_rearm(void *adq_cu, int adq_num);
//void adq214_time_stamp(void *adq_cu, int adq_num);

// Special define
#define CHECKADQ(f) {if(!(f)){printf("Error in " #f "\n"); goto error;}}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define WFAVG_FLAG_COMPENSATE_EXT_TRIG      0x0001
#define WFAVG_FLAG_COMPENSATE_LEVEL_TRIG    0x0002
#define WFAVG_FLAG_READOUT_FAST             0x0004
#define WFAVG_FLAG_READOUT_MEDIUM           0x0008
#define WFAVG_FLAG_READOUT_SLOW             0x0010
#define WFAVG_FLAG_ENABLE_LEVEL_TRIGGER     0x0020
#define WFAVG_FLAG_ENABLE_WAVEFORM_GET      0x0040
#define WFAVG_FLAG_ENABLE_AUTOARMNREAD      0x0080
#define WFAVG_FLAG_READOUT_A_ONLY           0x0100u
#define WFAVG_FLAG_READOUT_B_ONLY           0x0200u
#define WFAVG_FLAG_IMMEDIATE_READOUT        0x0400u

//debug counter output mode, bit [2:0]
#define COUNTER_BYPASS                       0x0000u
#define COUNTER_ALL_CHAN                     0x0001u
#define COUNTER_INTERLEAVED                  0x0002u
#define COUNTER_A_CHAN                       0x0003u
#define COUNTER_B_CHAN                       0x0004u
#define COUNTER_C_CHAN                       0x0005u
#define COUNTER_D_CHAN                       0x0006u

//debug counter count direction, bit [4:3]
#define COUNT_UP_ONLY                        0u << 3         //2'b00;
#define COUNT_DOWN_ONLY                      3u << 3         //2'b11;
#define COUNT_UP_AND_DOWN                    1u << 3         //2'b01;

//debug counter mode, bit [7:5]
#define FREE_RUNNING                         0u << 5
#define TRIGGER_COUNTER_UP                   1u << 5
#define TRIGGER_COUNTER_DOWN                 2u << 5
#define COUNT_UP_FROM_CONSTANT               3u << 5
#define COUNT_DOWN_FROM_CONSTANT             4u << 5
#define TRIGGER_GENERATOR_POS                5u << 5
#define TRIGGER_GENERATOR_NEG                6u << 5

void adq214(void *adq_cu, int adq_num)
{
  int mode;
  unsigned int enable_debug_counter = 0;
  int* revision = ADQ_GetRevision(adq_cu, adq_num);
  printf("\nConnected to ADQ214 #1\n\n");

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

  printf("\nUse debug data?\n0 = No\n1 = Yes\n");
  scanf("%u", &enable_debug_counter);

  if (enable_debug_counter)
  {
    /*Output Mode*/
    //COUNTER_BYPASS
    //COUNTER_ALL_CHAN
    //COUNTER_INTERLEAVED
    //COUNTER_A_CHAN
    //COUNTER_B_CHAN
    //COUNTER_C_CHAN
    //COUNTER_D_CHAN

    /*Count Direction*/
    //COUNT_UP_ONLY
    //COUNT_DOWN_ONLY
    //COUNT_UP_AND_DOWN

    /*Counter Mode*/
    //FREE_RUNNING
    //TRIGGER_COUNTER_UP
    //TRIGGER_COUNTER_DOWN
    //COUNT_UP_FROM_CONSTANT
    //COUNT_DOWN_FROM_CONSTANT
    //TRIGGER_GENERATOR_POS
    //TRIGGER_GENERATOR_NEG

    //ConfigureDebugCounter(Direction, Output_Mode, Counter_Mode, Constant)
    ADQ_ConfigureDebugCounter(adq_cu, adq_num, COUNT_UP_AND_DOWN, COUNTER_ALL_CHAN, COUNT_UP_FROM_CONSTANT, (signed int) 13);
  }

  //Select mode
  mode = 1;
  printf("\nChoose collect mode.\n 1 = Multi-Record\n 2 = Streaming\n 3 = Waveform Averaging manual rearm\n 4 = Waveform Averaging auto rearm\n\n");
  scanf("%d", &mode);

  switch (mode)
  {
  case 1:
    adq214_multi_record(adq_cu, adq_num);
    break;
  case 2:
    adq214_streaming(adq_cu, adq_num);
    break;
  case 3:
    wfa_manual_rearm(adq_cu, adq_num);
    break;
  case 4:
    wfa_auto_rearm(adq_cu, adq_num);
    break;
    //case 5:
    //ContinuousStreamingExample(adq_cu, adq_num);
    //    break;
  default:
    return;
    break;
  }

}
void adq214_multi_record(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  int trig_channel;
  int clock_source;
  int pll_divider;
  unsigned int number_of_records;
  unsigned int samples_per_record;
  unsigned int pretrigger_samples;
  unsigned int triggerdelay_samples;
  unsigned int buffersize;
  unsigned char channelsmask;
  unsigned int maskinput;
  unsigned int n_records_collect;
  unsigned int n_sample_skip;
  unsigned int sample_decimation;
  int write_to_file = 1;
  int overflow;
  short* buf_a = NULL;
  short* buf_b = NULL;
  void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers
  // Files for output storage
  FILE* outfile[2];
  int exit = 0;

  outfile[0] = fopen("data_A.out", "w");
  outfile[1] = fopen("data_B.out", "w");

//======================== This section is for the lazy developer who does not want to enter every value one by one=======

  //In DEBUG MODE, No user input promt will be executed. This is controlled by the #ifdef _DEBUG define.

  n_sample_skip = 1;
  sample_decimation = 0;
  trig_mode = 1;
  trig_level = 0;
  trig_flank = 1;
  trig_channel = 3;
  clock_source = 0;
  pll_divider = 2;
  number_of_records = 1;
  samples_per_record = 1024;
  pretrigger_samples = 0;
  triggerdelay_samples = 0;
  channelsmask = 3;
  n_records_collect = 0;
  maskinput = 1;

//========================================================================================================================
  // Sample skip
  printf("\nChoose sample skip.\n 1 <= skip <= 131070.\n");
  scanf("%u", &n_sample_skip);
  while((n_sample_skip == 0) || (n_sample_skip > 131070))
  {
    printf("\nError: Invalid sample skip.\n");
    printf("\nChoose sample skip.\n 1 <= skip <= 131070.\n");
    scanf("%u", &n_sample_skip);
  }

  CHECKADQ(ADQ_SetSampleSkip(adq_cu, adq_num, n_sample_skip));

  if (n_sample_skip == 1)
  {
    printf("\nChoose sample decimation.\n 2^0 <= 2^decimation <= 2^34.\n");
    scanf("%u", &sample_decimation);
    while(sample_decimation > 131070)
    {
      printf("\nError: Invalid sample decimation.\n");
      printf("\nChoose sample decimation.\n 2^0 <= 2^decimation <= 2^34.\n");
      scanf("%u", &sample_decimation);
    }

    CHECKADQ(ADQ_SetSampleDecimation(adq_cu, adq_num, sample_decimation));
    if (sample_decimation > 0)
      CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num, ADQ214_DATA_FORMAT_UNPACKED_32BIT)); //Decimation always outputs 32 bit samples
  }

  // Trigger mode
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
    printf("\nChoose trig level.\n -8192 <= level <= 8191\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    printf("\nChoose trig channel.\n 1 = Channel A\n 2 = Channel B\n 3 = Any channel\n");
    scanf("%d", &trig_channel);
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    printf("\n");
    break;
           }
  default :
    return;
    break;
  }

  // Clock source
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

  // Set up multi reckord
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

  CHECKADQ(ADQ_SetPreTrigSamples(adq_cu, adq_num, pretrigger_samples));
  CHECKADQ(ADQ_SetTriggerDelay(adq_cu, adq_num, triggerdelay_samples));
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num, number_of_records, samples_per_record));

  if (trig_mode == 1) //SW Trigger Mode
  {
    int trigged;
    printf("Automatically triggering your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num)); //Trig multiple times until all records are triggered
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
    }while (trigged == 0);
  }
  else // External & Level
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

  if (ADQ_GetStreamOverflow(adq_cu, adq_num))     //This part is needed to prevent a lock-up in case of overflow, which can happen very rarely in normal use
  {
    printf("\nOVERFLOW!!!\n");
    printf("\nOVERFLOW!!!\n");
    printf("\nOVERFLOW!!!\n");
    ADQ_ResetDevice(adq_cu, adq_num,4);
  }

  // collect data
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

  printf("Collecting data, please wait...\n");


  // Each data buffer must contain enough samples to store all the records consecutively
  //buffersize = number_of_records * samples_per_record;          //This row can be good to have for debugging purpose but the row below is better
  buffersize = n_records_collect * samples_per_record;            //Only create a buffer with a size that is enough for the number of records you want to collect and NOT for ALL the availible records
  buf_a = (short*)calloc(buffersize,sizeof(short));
  buf_b = (short*)calloc(buffersize,sizeof(short));
  if(buf_a == NULL)
    goto error;
  if(buf_b == NULL)
    goto error;

  // Create a pointer array containing the data buffer pointers
  target_buffers[0] = (void*)buf_a;
  target_buffers[1] = (void*)buf_b;

  CHECKADQ(ADQ_GetData(adq_cu, adq_num,target_buffers,buffersize,sizeof(short),0,n_records_collect,channelsmask,0,samples_per_record,ADQ_TRANSFER_MODE_NORMAL));

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

  // Return ADQ to single-trig mode
  CHECKADQ(ADQ_MultiRecordClose(adq_cu, adq_num));

  printf("\n\n Done. %u samples stored in data_A.out and data_B.out respectively.\n", n_records_collect*samples_per_record);
  if ((trig_mode == 2) || (trig_mode == 3))
  {
    int trig_point = ADQ_GetTrigPoint(adq_cu, adq_num);
    int trigged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);
    printf("Trig point: %d \n",trig_point);
    printf("Trig channel: %d \n", trigged_ch);
  }
  overflow = ADQ_GetOverflow(adq_cu, adq_num);
  if (overflow ==1)
    printf("Sample overflow in batch.\n");

error:

  fclose(outfile[0]);
  fclose(outfile[1]);

  if(buf_a != NULL)
    free(buf_a);
  if(buf_b != NULL)
    free(buf_b);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  scanf("%d", &exit);
  if (exit == 1)
  {
    int	trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int	level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int	flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int trg_ch = ADQ_GetLvlTrigChannel(adq_cu, adq_num);
    int	frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int	clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int	trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);
    int trged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",level);
    printf("Trig flank:        %d \n",flank);
    printf("Trig channel:      %d \n",trg_ch);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:	   %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Trigged Ch:        %d \n", trged_ch);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }
  return;
}

void adq214_streaming(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  int trig_channel;
  int stream_ch;
  int clock_source;
  int pll_divider;
  int sample_size;
  unsigned int n_samples_collect;
  unsigned int n_sample_skip;
  unsigned int sample_decimation;
  unsigned int buffers_filled;
  int collect_result;
  unsigned int samples_to_collect;
  int n_buffers; //Number of buffers
  int s_buffer = 131072; //Size of each buffer
  signed short* data_stream_target_s = NULL;
  signed int* data_stream_target_i = NULL;
  unsigned int interleave_channels_factor = 4;
  unsigned int tmp;
  FILE* outfileA = NULL;
  FILE* outfileB = NULL;
  int exit;
  unsigned int i;
  int sample;
  char fstr[10];

  outfileA = fopen("data_A.out", "w");
  outfileB = fopen("data_B.out", "w");

  //========================================================
  //FOR DEBUGGING PURPOSE ONLY!!!
  /*
  tmp = 0;
  n_sample_skip = 1;
  n_buffers = 8;
  s_buffer = 512*4;
  sample_decimation = 0;
  stream_ch = 0;
  trig_mode = 0;
  n_samples_collect = 4096;
  trig_level = 200;
  trig_flank = 0;
  trig_channel = 1;
  clock_source = 0;
  pll_divider = 2;
  */
  //========================================================
  printf("\nSetup transfer buffers.\n 0 = DEFAULT, 1 = OVERRIDE\n");
  scanf("%u", &tmp);
  if (tmp != 0)
  {
    printf("\nChoose number of buffers, 1 <= buffers <= 65536.\n");
    scanf("%d", &n_buffers);
    printf("\nChoose bytes per buffer (512 byte increments), 1024 <= bytesbuffers <= 32*(1024*1024).\n");
    scanf("%d", &s_buffer);

    CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, n_buffers, s_buffer));
  }

  // Unpacked 14 bit data.
  CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num,ADQ214_DATA_FORMAT_UNPACKED_14BIT));

  //CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num,ADQ214_DATA_FORMAT_UNPACKED_16BIT));


  printf("\nChoose sample skip.\n 1 <= skip <= 131070.\n");
  scanf("%u", &n_sample_skip);
  while((n_sample_skip == 0) || (n_sample_skip > 131070))
  {
    printf("\nError: Invalid sample skip.\n");
    printf("\nChoose sample skip.\n 1 <= skip <= 131070.\n");
    scanf("%u", &n_sample_skip);
  }

  CHECKADQ(ADQ_SetSampleSkip(adq_cu, adq_num, n_sample_skip));

  if (n_sample_skip == 1)
  {
    printf("\nChoose sample decimation.\n 2^0 <= 2^decimation <= 2^34.\n");
    scanf("%u", &sample_decimation);
    while(sample_decimation > 131070)
    {
      printf("\nError: Invalid sample decimation.\n");
      printf("\nChoose sample decimation.\n 2^0 <= 2^decimation <= 2^34.\n");
      scanf("%u", &sample_decimation);
    }
    CHECKADQ(ADQ_SetSampleDecimation(adq_cu, adq_num, sample_decimation));

    if (sample_decimation > 0)
      CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num, ADQ214_DATA_FORMAT_UNPACKED_32BIT)) //Decimation always outputs 32 bit samples
  }

  printf("\nChoose stream channels.\n 0 = Both\n 1 = A\n 2 = B\n");
  scanf("%d", &stream_ch);
  switch(stream_ch)
  {
  case 0:
    stream_ch = ADQ214_STREAM_ENABLED_BOTH;
    break;
  case 1:
    stream_ch = ADQ214_STREAM_ENABLED_A;
    break;
  case 2:
    stream_ch = ADQ214_STREAM_ENABLED_B;
    break;
  default:
    return;
    break;
  }

  printf("\nChoose trig mode.\n 0 = No trigger, start directly on arm\n 1 = SW Trigger Mode\n 2 = External Trigger Mode\n 3 = Level Trigger Mode\n");
  scanf("%d", &trig_mode);
  switch (trig_mode)
  {
  case 0 : {
    stream_ch &= 0x7;
    break;
           }
  case 1 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    stream_ch |= 0x8;
    break;
           }
  case 2 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    stream_ch |= 0x8;
    break;
           }
  case 3 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n -8192 <= level <= 8191\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    printf("\nChoose trig channel.\n 1 = Channel A\n 2 = Channel B\n 3 = Any channel\n");
    scanf("%d", &trig_channel);
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    printf("\n");
    stream_ch |= 0x8;
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

  printf("Choose how many samples to collect.\n samples > 0.\n Multiples of 8 recommended.");
  scanf("%u", &n_samples_collect);
  while(n_samples_collect == 0)
  {
    printf("\nError: Invalid number of samples.\n");
    printf("Choose how many samples to collect.\n samples > 0.\n");
    scanf("%u", &n_samples_collect);
  }

  // Allocate temporary buffer for streaming data
  if (sample_decimation > 0)
  {
    sample_size = sizeof(signed int);
    CHECKADQ(data_stream_target_i = (signed int*)calloc(n_samples_collect, sizeof(signed int)));
    memset(data_stream_target_i, 0, n_samples_collect*sizeof(signed int));
    interleave_channels_factor = 2;
  }
  else
  {
    sample_size = sizeof(signed short);
    CHECKADQ(data_stream_target_s = (signed short*)calloc(n_samples_collect, sizeof(signed short)));
    memset(data_stream_target_s, 0, n_samples_collect*sizeof(signed short));
    interleave_channels_factor = 4;
  }

  //Enable streaming bypass of DRAM
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,stream_ch));
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  printf("Collecting data, please wait...\n");

  samples_to_collect = n_samples_collect;

  while (samples_to_collect > 0)
  {
    unsigned int samples_in_buffer;
    if (trig_mode == 1)	//If trigger mode is sofware
    {
      ADQ_SWTrig(adq_cu, adq_num);
    }
    do
    {
      collect_result = ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled);
      printf("Filled: %2u\n", buffers_filled);

    } while ((buffers_filled == 0) && (collect_result));

    collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);

    samples_in_buffer = (sample_decimation > 0) ? (s_buffer / 4) : (s_buffer / 2);
    samples_in_buffer = MIN(samples_in_buffer, samples_to_collect);

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Warning: Streaming Overflow!\n");
      collect_result = 0;
    }

    if (collect_result)
    {
      // Buffer all data in RAM before writing to disk, if streaming to disk is need a high performance
      // procedure could be implemented here.
      // Data format is set to 16 bits or 32 bits, so buffer size is Samples*2bytes or Samples*4bytes
      if (sample_decimation > 0)
      {
        memcpy((void*)&data_stream_target_i[n_samples_collect-samples_to_collect], ADQ_GetPtrStream(adq_cu, adq_num), samples_in_buffer*sample_size);
      }
      else
      {
        memcpy((void*)&data_stream_target_s[n_samples_collect-samples_to_collect], ADQ_GetPtrStream(adq_cu, adq_num), samples_in_buffer*sample_size);
      }

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

  if (sample_decimation > 0)
    sprintf(fstr, "%s", "%i\n");
  else
    sprintf(fstr, "%s", "%hi\n");

  switch(stream_ch)
  {

  case ADQ214_STREAM_ENABLED_BOTH:
    {
      //Channel data is interleaved in 8 bytes chunks (4 samples with unpacked 14-bit or 16-bit modes)
      // Sample order in raw stream: A0 A1 A2 A3 B0 B1 B2 B3 A4 ...
      samples_to_collect = n_samples_collect;
      while (samples_to_collect>0)
      {
        for (i=0; (i<interleave_channels_factor) && (samples_to_collect>0); i++)
        {
          if (sample_decimation > 0)
          sample = data_stream_target_i[n_samples_collect-samples_to_collect];

          else
            sample = (int)data_stream_target_s[n_samples_collect-samples_to_collect];
          fprintf(outfileA, fstr, sample);
          samples_to_collect--;
        }
        for (i=0; (i<interleave_channels_factor) && (samples_to_collect>0); i++)
        {
          if (sample_decimation > 0)
            sample = data_stream_target_i[n_samples_collect-samples_to_collect];
          else
            sample = (int)data_stream_target_s[n_samples_collect-samples_to_collect];
          fprintf(outfileB, fstr, sample);
          samples_to_collect--;
        }
      }
    }
    break;
  case ADQ214_STREAM_ENABLED_A:
    for (i=0; i<n_samples_collect; i++)
    {
      if (sample_decimation > 0)
        sample = data_stream_target_i[i];
      else
        sample = (int)data_stream_target_s[i];
      fprintf(outfileA, fstr, sample);
    }
    break;
  case ADQ214_STREAM_ENABLED_B:
    for (i=0; i<n_samples_collect; i++)
    {
      if (sample_decimation > 0)
        sample = data_stream_target_i[i];
      else
        sample = (int)data_stream_target_s[i];
      fprintf(outfileB, fstr, sample);
    }
    break;
  default:
    break;
  }

error:

  if (outfileA != NULL) fclose(outfileA);
  if (outfileB != NULL) fclose(outfileB);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  scanf("%d", &exit);
  if (exit == 1)
  {
    int  trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int trg_ch = ADQ_GetLvlTrigChannel(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);
    int trged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",level);
    printf("Trig flank:        %d \n",flank);
    printf("Trig channel:      %d \n",trg_ch);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Trigged Ch:        %d \n", trged_ch);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }
  return;
}


void wfa_manual_rearm(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_edge;
  int clock_source;
  int pll_divider;
  int WriteToFile;
  int Int_Trig_Freq;
  unsigned int record_counter;
  unsigned int number_of_averaged_records;
  unsigned int number_of_waveforms;
  unsigned int samples_per_waveform;

  FILE* outfile = NULL;
  char fname[256];

  int readmode = 3; //Means both channels, but is needed to fool WFA
  unsigned int flags = 0x0000;

  unsigned int collect_result;
  int* data_stream;
  int* data_target[2];

  unsigned int retrycounter;
  unsigned char averageready = 0;
  unsigned int nofrecordscompleted = 0;
  unsigned char in_idle = 0;

  unsigned int LoopVar;
  int exit;

  //======================== This section is for the lazy developer who does not want to enter every value one by one=======

  //When you use this section remember to comment the lines that starts with "scanf_s"
  /*
  WriteToFile = 1;
  trig_mode = 1;
  trig_edge = 1;
  trig_level = 0;
  Int_Trig_Freq = 2000;
  clock_source = 0;
  pll_divider = 2;

  //use_debug_counter = 1;
  number_of_waveforms =3;                  //How many waveforms to add together
  number_of_averaged_records = 2;          //This is how many AVERAGED records you want to save to file.
  samples_per_waveform = 1024;
  */
  //========================================================================================================================

  printf("\nWrite to file?.\n 0 = No\n 1 = Yes\n 2 = Yes (and suppress console output)\n");
  scanf("%d", &WriteToFile);

  printf("\nChoose trig mode.\n 1 = SW Trigger Mode\n 2 = External Trigger Mode\n 3 = Level Trigger Mode\n 4 = Internal Trigger Mode\n");
  scanf("%d", &trig_mode);

  switch (trig_mode)
  {
  case 1 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
           }
  case 2 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_edge);
    CHECKADQ(ADQ_SetExternTrigEdge(adq_cu, adq_num, trig_edge));
    break;
           }
  case 3 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n -8192 <= level <= 8191\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_edge);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_edge));
    printf("\n");
    break;
           }
  case 4: {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose Internal Trigger Frequency (Hz)\n");
    scanf("%d", &Int_Trig_Freq);
    CHECKADQ(ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, 20000));    //Just set a default value in case...
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

  printf("\nChoose number of averaged records to collect.\n");
  scanf("%u", &number_of_averaged_records);

  printf("\nChoose number of waveforms for every averaged record.\n");
  scanf("%u", &number_of_waveforms);

  printf("\nChoose number of samples per waveform to average (steps of 128 samples).\n");
  scanf("%u", &samples_per_waveform);

  switch(readmode)
  {
  case 1:
    flags |= WFAVG_FLAG_READOUT_A_ONLY;
    break;
  case 2:
    flags |= WFAVG_FLAG_READOUT_B_ONLY;
    break;
  default:
    break;
  }

  if (trig_mode == 2)   //for trig_mode = 1 (software) see further below
  {
    flags |= WFAVG_FLAG_COMPENSATE_EXT_TRIG;
  }
  else if (trig_mode == 3)
  {
    flags |= WFAVG_FLAG_COMPENSATE_LEVEL_TRIG;
    flags |= WFAVG_FLAG_ENABLE_LEVEL_TRIGGER;
  }

  if (ADQ_IsPCIeDevice(adq_cu, adq_num))
  {
    flags |= WFAVG_FLAG_READOUT_FAST; // Fast readout on PCIE
  }
  else
    flags |= WFAVG_FLAG_READOUT_SLOW; // Slow readout on USB

  flags |= WFAVG_FLAG_ENABLE_WAVEFORM_GET; // Enable the special get waveform function


  // Allocate temporary buffer for streamed data
  CHECKADQ(data_stream = (int*)malloc(samples_per_waveform*sizeof(signed int)*2));      //multiplied by 2 because 2 channels

  // Allocate channels
  for(unsigned int ch = 0; ch < 2; ch++)
    data_target[ch] = (int*)malloc(samples_per_waveform*sizeof(signed int));

  CHECKADQ(ADQ_SetTransferTimeout(adq_cu, adq_num,1000));

  //If using the debug counter mode 2 to verify if the samples are added up correctly, trigger delay should be set to 68 samples to align the counter pattern
  CHECKADQ(ADQ_WaveformAveragingSetup(adq_cu, adq_num, number_of_waveforms, samples_per_waveform, 0, 0, flags));


  record_counter = 0;

  CHECKADQ(ADQ_WaveformAveragingArm(adq_cu, adq_num));

  //Run below according to number of loops
  for (LoopVar = 0; LoopVar < number_of_averaged_records; LoopVar++)
  {
    CHECKADQ(ADQ_WaveformAveragingArm(adq_cu, adq_num));	//Arm the WFA block
    retrycounter = 0;	//Reset retrycounter

    do
    {
      // Get the data available signal
      Sleep(1);	//Sleep for one milisecond
      CHECKADQ(ADQ_WaveformAveragingGetStatus(adq_cu, adq_num, &averageready, &nofrecordscompleted, &in_idle));	//Get the current status and how many waveforms has been accumulated
      retrycounter++;
      printf("Averaged records: %5u\n", nofrecordscompleted);
      if (trig_mode == 1)	//If trigger mode is sofware
      {
        ADQ_SWTrig(adq_cu, adq_num);
      }
    } while (!averageready && retrycounter < (2*number_of_waveforms + 20));

    if (retrycounter == (2*number_of_waveforms + 20))
    {
      printf("\nFailed to get ready signal!!!!\n");
    }
    CHECKADQ(collect_result = ADQ_WaveformAveragingGetWaveform(adq_cu, adq_num, data_stream));

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Warning: Streaming Overflow!\n");
      // collect_result = 0;

    }

    if (collect_result)
    {

      CHECKADQ(ADQ_WaveformAveragingParseDataStream(adq_cu, adq_num, samples_per_waveform, data_stream, data_target));
      // Resort vector according to documentation

      if (WriteToFile != 2)
      {
        unsigned int sample = 0;
        printf("First 10 Samples:\n");
        for (sample = 0; sample < 10; sample++)
        {
          printf("[%4u]: %d\n", sample, data_stream[sample]);
        }
      }

      //All previous generated data files will be deleted and new file will be created. Please check Properties->Build Events>Post-Build Event to edit this action.
      if (WriteToFile > 0)
      {
        for(unsigned int ch = 0; ch < 2; ch++)
        {
          sprintf(fname, "averaged_waveform_%05u_ch%u.out", record_counter, ch);
          outfile = fopen(fname, "wb");
          fwrite((void*)data_target[ch], sizeof(int), samples_per_waveform, outfile);
          fclose(outfile);
          outfile = NULL;
          if (ch == 0)
            printf("Wrote record %05u of channel (A) to file %s\n", record_counter, fname);
          else
            printf("Wrote record %05u of channel (B) to file %s\n", record_counter, fname);
        }

        record_counter++;
      }
    }
    else
    {
      printf("Get Waveform function failed!\n");
    }

    CHECKADQ(ADQ_WaveformAveragingDisarm(adq_cu, adq_num));
  }

error:

  if (ADQ_GetLastError(adq_cu, adq_num) != 0xCDCDCDCD)        //CDCDCDCD is the value of error code when "m_last_error" has not been initialized, E.I there has not been any errors
  {
    printf("Last error reported by device is HEX: %8X.\n", ADQ_GetLastError(adq_cu, adq_num));
  }

  free(data_stream);
  for(unsigned int ch = 0; ch < 2; ch++)
    free(data_target[ch]);

  if (outfile != NULL)
    fclose(outfile);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  scanf("%d", &exit);
  if (exit == 1)
  {
    int  trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int trg_ch = ADQ_GetLvlTrigChannel(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);
    int trged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",level);
    printf("Trig flank:        %d \n",flank);
    printf("Trig channel:      %d \n",trg_ch);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Trigged Ch:        %d \n", trged_ch);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }

  return;
}


void wfa_auto_rearm(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_edge;
  int clock_source;
  int pll_divider;
  int WriteToFile;
  int Int_Trig_Freq;
  unsigned int number_of_waveforms;
  unsigned int samples_per_waveform;
  unsigned int number_of_averaged_records;

  FILE* outfile = NULL;
  char fname[256];

  int readmode = 3;
  unsigned int collect_result;
  unsigned int flags = 0x0000;
  int* data_stream;
  int* data_target[2];
  unsigned int retrycounter;
  unsigned char in_idle = 0;
  unsigned int record_counter = 0;
  unsigned int ShutdownStatus = 0;
  int exit;

  //======================== This section is for the lazy developer who does not want to enter every value one by one=======

  //When you use this section remember to comment the lines that starts with "scanf_s"
  /*
  WriteToFile = 1;
  trig_mode = 2;
  trig_edge = 1;
  //trig_level = 200;
  Int_Trig_Freq = 200;
  clock_source = 0;
  pll_divider = 2;

  number_of_waveforms = 2;               //This is how many waveforms TO ADD TOGETHER
  number_of_averaged_records = 5;     //This is how many AVERAGED records you want to save to file. 0 = INFINITE STREAMING
  samples_per_waveform = 1024;
  */
  //========================================================================================================================

  printf("\nWrite to file?.\n 0 = No\n 1 = Yes\n 2 = Yes (and suppress console output)\n");
  scanf("%d", &WriteToFile);

  printf("\nChoose trig mode.\n 1 = SW Trigger Mode\n 2 = External Trigger Mode\n 4 = Internal trigger (Level Trigger Mode is not allowed in auto-mode)\n");
  scanf("%d", &trig_mode);
  switch (trig_mode)
  {
  case 1 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num,trig_mode));
    break;
           }
  case 2 : {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num,trig_mode));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_edge);
    CHECKADQ(ADQ_SetExternTrigEdge(adq_cu, adq_num,trig_edge));
    break;
           }
  case 4:
    {
      CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num,trig_mode));
      printf("\nChoose Internal Trigger Frequency (Hz)\n");
      scanf("%d", &Int_Trig_Freq);
      CHECKADQ(ADQ_SetInternalTriggerFrequency(adq_cu, adq_num,Int_Trig_Freq));
      //CHECKADQ(ADQ_SetInternalTriggerPeriod(20000));    //Just set a default value in case...
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
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num,clock_source));
    printf("\nChoose PLL frequency divider.\n 2 <= divider <= 20, f_clk = 800MHz/divider\n");
    scanf("%d", &pll_divider);
    CHECKADQ(ADQ_SetPllFreqDivider(adq_cu, adq_num,pll_divider));
  }
  else
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num,clock_source));

  printf("\nChoose number of averaged records to collect.\n");
  scanf("%u", &number_of_averaged_records);

  printf("\nChoose number of waveforms for every averaged record.\n");
  scanf("%u", &number_of_waveforms);

  printf("\nChoose number of samples per waveform to average (steps of 128 samples).\n");
  scanf("%u", &samples_per_waveform);

  /*
  printf("\nPress any key to START streaming!\nPress any key again to STOP streaming!");
  unsigned char key;
  while (_kbhit())
  {
  key = _getch();
  }
  key = 'o';
  do
  {
  if (_kbhit())
  {
  key = _getch();
  key = 's';
  }
  Sleep(1);
  } while (key != 's');	//exit loop and start streaming
  */

  switch(readmode)
  {
  case 1:
    flags |= WFAVG_FLAG_READOUT_A_ONLY;
    break;
  case 2:
    flags |= WFAVG_FLAG_READOUT_B_ONLY;
    break;
  default:
    break;
  }

  if (trig_mode == 2)
  {
    flags |= WFAVG_FLAG_COMPENSATE_EXT_TRIG;
  }

  flags |= WFAVG_FLAG_ENABLE_AUTOARMNREAD; // Enable the auto armnread feature
  //flags |= WFAVG_FLAG_IMMEDIATE_READOUT; // Enable the readout buffer for fast re-arm. Please respect max record size for this function.

  if (ADQ_IsPCIeDevice(adq_cu, adq_num))
  {
    flags |= WFAVG_FLAG_READOUT_FAST; // Fast readout on PCIE
  }
  else
    flags |= WFAVG_FLAG_READOUT_SLOW; // Slow readout on USB

  //flags |= WFAVG_FLAG_IMMEDIATE_READOUT; // Continous readout


  // Allocate temporary buffer for streaming data
  CHECKADQ(data_stream = (int*)malloc(samples_per_waveform*2*sizeof(signed int)));

  // Allocate buffers for parsed data (one buffer per channel)
  for(unsigned int ch = 0; ch < 2; ch++)
    CHECKADQ(data_target[ch] = (int*)malloc(samples_per_waveform*sizeof(signed int)));

  CHECKADQ(ADQ_SetTransferTimeout(adq_cu, adq_num,5000));

  // Allocate DMA buffers. The buffers need to be the same size as the WFA output records, due to the fact
  // that the WFA module does not have any knowledge of the DMA buffer size, and will just keep outputting data.
  // Using non record-sized buffers will therefore cause the data to be misaligned.
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, 16, 2*samples_per_waveform*sizeof(signed int)));

  //If using the debug counter mode 2 to verify if the samples are added up correctly, trigger delay should be set to 68 samples to align the counter pattern
  CHECKADQ(ADQ_WaveformAveragingSetup(adq_cu, adq_num, number_of_waveforms, samples_per_waveform, 0, 0, flags));

  ADQ_IsPCIeDevice(adq_cu, adq_num);

  CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num,3));
  CHECKADQ(ADQ_WaveformAveragingArm(adq_cu, adq_num));
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,0x7));
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  retrycounter = 0;
  collect_result = 0;


  do
  {

    if (trig_mode == 1)    //If trigger mode is software trigger ....
    {
      unsigned int t;
      for(t = 0; t < number_of_waveforms; t++) // ...issue enough software triggers to produce a single averaged record
      {
        CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
      }
    }

    collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
    printf("collect_result: %01u \n",collect_result);
    if (collect_result)
    {
      retrycounter = 0;
      memcpy((void*)data_stream, ADQ_GetPtrStream(adq_cu, adq_num), 2*samples_per_waveform*sizeof(int));
      CHECKADQ(ADQ_WaveformAveragingParseDataStream(adq_cu, adq_num, samples_per_waveform, data_stream, data_target));

      if ((WriteToFile != 2) & (ShutdownStatus == 0))     //Do not print to file if shutdown sequence is started
      {
        unsigned int sample;
        printf("First 12 Samples:\n");
        for (sample = 0; sample < 12; sample++)
        {
          printf("[%4u]: %d\n", sample, data_stream[sample]);
        }
      }

      //All previous generated data files will be deleted and new file will be created. Please check Properties->Build Events>Post-Build Event to edit this action.
      if (WriteToFile > 0)
      {
        for(unsigned int ch = 0; ch < 2; ch++)
        {
          sprintf(fname, "averaged_waveform_%05u_ch%u.out", record_counter, ch);
          outfile = fopen(fname, "wb");
          fwrite((void*)data_target[ch], sizeof(int), samples_per_waveform, outfile);
          fclose(outfile);
          outfile = NULL;
          if (ch == 0)
            printf("Wrote record %05u of channel (A) to file %s\n", record_counter, fname);
          else
            printf("Wrote record %05u of channel (B) to file %s\n", record_counter, fname);
        }

      }
      else
      {
        printf("  - parsed record %05u.\n", record_counter);
      }
      record_counter++;
    }
    else
    {
      printf("Get stream failed!\n");
      retrycounter++;
      printf("retrycounter: %01u \n",retrycounter);
    }

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("WARNING: STREAMING OVERFLOW!\n");
      ShutdownStatus = 1;
    }

    // Check if to initiate shutdown
    if (((record_counter == number_of_averaged_records) && (number_of_averaged_records !=0))) //if number_of_averaged_records is zero the streaming will not stop
    {
      printf("Initiated shutdown.\n");
      ShutdownStatus = 1;
    }

  } while ((retrycounter < number_of_waveforms*number_of_averaged_records) && (ShutdownStatus < 1));

  CHECKADQ(ADQ_WaveformAveragingDisarm(adq_cu, adq_num));
  do
  {
    Sleep(5);
    CHECKADQ(ADQ_WaveformAveragingGetStatus(adq_cu, adq_num, NULL, NULL, &in_idle));
  } while (in_idle != 2);

  if (in_idle)
  {
    printf("Finalized shutdown.\n");
  }

  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));           //Note !!! DisarmTrigger must be executed BEFORE SetStreamStatus to shut down streaming interface properly
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,0x00));

error:

  if (ADQ_GetLastError(adq_cu, adq_num) != 0xCDCDCDCD)        //CDCDCDCD is the value of error code when "m_last_error" has not been initialized, E.I there has not been any errors
  {
    printf("Last error reported by device is HEX: %8X.\n", ADQ_GetLastError(adq_cu, adq_num));
  }

  free(data_stream);
  for(unsigned int ch = 0; ch < 2; ch++)
    free(data_target[ch]);

  if (outfile != NULL) fclose(outfile);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  scanf("%d", &exit);
  if (exit == 1)
  {
    int  trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  trig_level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int trg_ch = ADQ_GetLvlTrigChannel(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);
    int trged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",trig_level);
    printf("Trig flank:        %d \n",flank);
    printf("Trig channel:      %d \n",trg_ch);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Trigged Ch:        %d \n", trged_ch);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }

  return;
}

// This section regards function Time Stamp. De-comment if wanted.
/*
void adq214_time_stamp(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  int trig_channel;
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

//.................................
  // Files for output storage
  FILE* outfileA;
  FILE* outfileB;
  outfileA = fopen("data_A.out", "w");
  outfileB = fopen("data_B.out", "w");
//..................................

//Set trigger mode
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
    printf("\nChoose trig level.\n -8192 <= level <= 8191\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    printf("\nChoose trig channel.\n 1 = Channel A\n 2 = Channel B\n 3 = Any channel\n");
    scanf("%u", &trig_channel);
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    printf("\n");
    break;
    }
  default :
    return;
    break;
  }

//For testing purposes: Activate one of these, comment the section above.
  //Set trigger mode: Level trigger mode
  //CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num,3));

  //Set trigger mode: SW trigger mode
  //CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, 1)); //SW trigger mode

//.......................................
  //Set clock source
  printf("\nChoose clock source.\n 0 = Internal clock, internal reference\n 1 = Internal clock, external reference\n 2 = External clock\n");
  scanf("%d", &clock_source);
  if ((clock_source == 0) || (clock_source == 1))
  {
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));
    printf("\nChoose PLL frequency divider.\n 2 <= divider <= 20, f_clk = 800MHz/divider\n");
    scanf("%d", &pll_divider);
    CHECKADQ(ADQ_SetPllFreqDivider(adq_cu, adq_num, pll_divider));//Sampling frequency
  }
  else
    CHECKADQ(ADQ_SetClockSource(adq_cu, adq_num, clock_source));

//.......................................
  //set number of records
  printf("\nChoose number of records.\n");
  scanf("%u", &number_of_records);

  //set number of samples
  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

//......................................
  //set time stamp mode
  printf("\nChoose timestamp mode.\n 0 = Sync off\n 1 = Sync on\n");
  scanf("%d", &trig_time_mode);

  //Set restart mode
  printf("\nSet reset mode. \n 0 = Pulse restart\n 1 = Immediate restart\n");
  scanf("%d", &restart_mode);

  CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,restart_mode));

//-----------------------------------
  //Setup multi records
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num, number_of_records, samples_per_record));*/
/*
    //printf("\nPlease trig your device to collect data.\n");

    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

    int trigged;
    do
    {
      printf("\ntrigged.\n");
      trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
    }while (trigged == 0);

  printf("\nDevice trigged\n");*/
/*//.............?

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
  }

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
          //synctrig = 1;//tests�ttning

          CHECKADQ(ADQ_SetTrigTimeMode(adq_cu, adq_num,0));//continious count
          CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));

          do{
            CHECKADQ(ADQ_GetTrigTime(adq_cu, adq_num));
            od_timest.push_back(ADQ_GetTrigTime(adq_cu, adq_num));
          }while (!synctrig);

          //for testing. Else decomment section below.
          /*if (synctrig){
          CHECKADQ(ADQ_ResetTrigTimer(adq_cu, adq_num,1));
          }*/
          /*do{
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
          //synctrig = 1;//tests�ttning

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
  int* data_a_ptr_addr0 = ADQ_GetPtrDataChA(adq_cu, adq_num);
  int* data_b_ptr_addr0 = ADQ_GetPtrDataChB(adq_cu, adq_num);

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
              cout << *cii << endl;
              cout << ((*cii+1 - *cii)/2)<< endl;
              od_timest_cal.push_back((*cii+1 - *cii)/2);
            }
          }


          int dataa = *(data_a_ptr_addr0 + j);
          int datab = *(data_b_ptr_addr0 + j);
          fprintf(outfileA, "%d\n", dataa);
          fprintf(outfileB, "%d\n", datab);
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

  printf("\n\n Done. %u samples stored in data_A.out and data_B.out respectively.\n", n_records_collect*samples_per_record);
  if ((trig_mode == 2) || (trig_mode == 3))
  {
    int trig_point = ADQ_GetTrigPoint(adq_cu, adq_num);
    int trigged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);
    printf("Trig point: %d \n",trig_point);
    printf("Trig channel: %d \n", trigged_ch);
  }
  int overflow = ADQ_GetOverflow(adq_cu, adq_num);
  if (overflow ==1)
    printf("Sample overflow in batch.\n");

error:

  fclose(outfileA);
  fclose(outfileB);

  printf("Press 0 followed by ENTER to exit, 1 followed by ENTER to see all settings of the device\n");
  int exit;
  scanf("%d", &exit);
  if (exit == 1)
  {
    int  trigged = ADQ_GetAcquired(adq_cu, adq_num);
    unsigned int page_count = ADQ_GetPageCount(adq_cu, adq_num);
    int  trig_level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    int  flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);
    int trg_ch = ADQ_GetLvlTrigChannel(adq_cu, adq_num);
    int  frq_div = ADQ_GetPllFreqDivider(adq_cu, adq_num);
    int  clk_src = ADQ_GetClockSource(adq_cu, adq_num);
    int  trg_md = ADQ_GetTriggerMode(adq_cu, adq_num);
    unsigned int usb_addr = ADQ_GetUSBAddress(adq_cu, adq_num);
    int* revision = ADQ_GetRevision(adq_cu, adq_num);
    int trig_pt = ADQ_GetTrigPoint(adq_cu, adq_num);
    int ovrflow = ADQ_GetOverflow(adq_cu, adq_num);
    int trged_ch = ADQ_GetTriggedCh(adq_cu, adq_num);

    printf("\nPage count:        %u \n",page_count);
    printf("Trigged:           %d \n",trigged);
    printf("Trig level:        %d \n",trig_level);
    printf("Trig flank:        %d \n",flank);
    printf("Trig channel:      %d \n",trg_ch);
    printf("Freqency divider:  %d \n",frq_div);
    printf("Clock Source:      %d \n",clk_src);
    printf("Trigger Mode:     %d \n",trg_md);
    printf("USB address:       %u \n", usb_addr);
    printf("Revision[0]:       %d \n",revision[0]);
    printf("Revision[1]:       %d \n",revision[1]);
    printf("Revision[2]:       %d \n",revision[2]);
    printf("Trig Point:        %d \n", trig_pt);
    printf("Trigged Ch:        %d \n", trged_ch);
    printf("Overflow:          %d \n", ovrflow);

    printf("\nPress 0 followed by ENTER to exit.\n");
    scanf("%d", &exit);
  }
  return;
}*/
