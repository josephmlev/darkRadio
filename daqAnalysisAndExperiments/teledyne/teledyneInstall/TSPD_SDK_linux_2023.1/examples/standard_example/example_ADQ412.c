// File: example_ADQ412.cpp
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
  #include <conio.h>
  #include <excpt.h>
#else
  #include <unistd.h>
  #include <sys/time.h>
  #include <termios.h>
  #define Sleep(interval) usleep(1000*interval)
  #define _kbhit() kbhit_linux()
  #include <stdlib.h>
  #include <string.h>
#endif

int kbhit_linux (void);
void adq412_streaming(void *adq_cu, int adq_num);
void adq412_microtca_functionality_demo(void *adq_cu, int adq_num);
void adq412_multirecordexample(void *adq_cu, int adq_num);
void adq412_triggeredstreamingexample(void *adq_cu, int adq_num);

// Special define
#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#ifdef LINUX

void changemode(int dir)
{
  static struct termios oldt, newt;

  if ( dir == 1 )
  {
    tcgetattr( STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newt);
  }
  else
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
}

int kbhit_linux (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO(&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET(STDIN_FILENO, &rdfs);
}

#endif

void adq412(void *adq_cu, int adq_num)
{
  int mode;
  int* revision = ADQ_GetRevision(adq_cu, adq_num);
  printf("\nConnected to ADQ412 #1\n\n");

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

START:

  printf("\nChoose collect mode.\n 0 = Exit\n 1 = Multi-Record\n 2 = Streaming\n 3 = MicroTCA functionality demo\n 4 = TriggeredStreaming \n\n\n");

  scanf("%d", &mode);

  switch (mode)
  {
  case 0:
    goto EXIT;
    break;
  case 1:
    adq412_multirecordexample(adq_cu, adq_num);
    break;
  case 2:
    adq412_streaming(adq_cu, adq_num);
    break;
  case 3:
    adq412_microtca_functionality_demo(adq_cu, adq_num);
    break;
  case 4:
    adq412_triggeredstreamingexample(adq_cu, adq_num);
    break;
  default:
    return;
    break;
  }

  goto START;

EXIT:

  return;
}

//=========Uncomment this block to skip scanf() in debug mode. Just make sure all paramters already have default values===========

//#ifdef _DEBUG
//  #define scanf(a, b) noscanf()
//#endif
//
//void noscanf()
//{
//    //Dummy function
//}

//================================================================================================================================

void adq412_multirecordexample(void *adq_cu, int adq_num)
{
  int get_trig_level;
  int get_trig_flank;
  //Setup ADQ
  int trig_mode;
  int trig_level;
  int trig_flank;
  unsigned int samples_per_record;
  unsigned int number_of_records;
  unsigned int buffersize;
  unsigned int interleaving_mode = 0;
  unsigned char channelsmask;
  unsigned int maskinput;
  unsigned int trig_channel;
  unsigned int write_to_file = 1;
  unsigned int records_to_collect;
  unsigned int max_nof_samples = 0;

  short* buf_a;
  short* buf_b;
  short* buf_c;
  short* buf_d;
  void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  FILE* outfile[4] = {NULL, NULL, NULL, NULL};
  char *serialnumber;
  unsigned int i;
//   int exit=0;

//======================== This section is for the lazy developer who does not want to enter every value one by one=======

  trig_mode = 2;
  number_of_records = 10;
  samples_per_record = 4096;
  interleaving_mode = 0;
  maskinput = 1;
  records_to_collect = 0;
  trig_level = 0;
  trig_flank = 1;
  trig_channel = 1;
//========================================================================================================================


  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

  serialnumber = ADQ_GetBoardSerialNumber(adq_cu, adq_num);

  printf("Device Serial Number: %s\n",serialnumber);


  outfile[0] = fopen("dataA.out", "w");
  outfile[1] = fopen("dataB.out", "w");
  outfile[2] = fopen("dataC.out", "w");
  outfile[3] = fopen("dataD.out", "w");
  for(i = 0; i < 4; i++) {
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
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    scanf("%u", &trig_channel);
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, trig_channel));
    printf("\n");

    get_trig_level = ADQ_GetLvlTrigLevel(adq_cu, adq_num);
    get_trig_flank = ADQ_GetLvlTrigEdge(adq_cu, adq_num);

    printf("Trig level set to %d", get_trig_level);
    printf("Trig flank set to %d", get_trig_flank);

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

  printf("\nRun collection in interleaved mode?\n(Please be aware of the reverted channel order marking between USB and P*Ie)\n0 = Non-interleave\n1 = Active ports A, C\n2 = Active ports B, D\n4 = Active ports A, B, C, D\n");
  scanf("%u", &interleaving_mode);

  CHECKADQ(ADQ_SetInterleavingMode(adq_cu, adq_num, interleaving_mode));

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
    printf("Automatically triggering your device to collect data.\n");
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
    printf("\nPlease trig your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu,adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu,adq_num));
    do
    {
      trigged = ADQ_GetAcquiredAll(adq_cu,adq_num);
    }while (trigged == 0);
  }
  printf("\nDevice trigged\n");

  if (ADQ_GetStreamOverflow(adq_cu,adq_num))     //This part is needed to prevent a lock-up in case of overflow, which can happen very rarely in normal use
  {
    printf("\nOVERFLOW!!!\n");
    printf("\nOVERFLOW!!!\n");
    printf("\nOVERFLOW!!!\n");
    ADQ_ResetDevice(adq_cu, adq_num,4);
  }

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

  // Get pointer to non-streamed unpacked data
  if (write_to_file)
  {
    unsigned int channel;
    for(channel = 0; channel < 4; channel++)
    {
      unsigned int  j;
      for (j=0; j<buffersize; j++)
      {
        if(channelsmask & (0x01 << channel))
          fprintf(outfile[channel], "%d\n", ((short*)target_buffers[channel])[j]);
      }
    }
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu,adq_num));

  printf("\n\nDone. Samples stored in data*.out!\n");


error:

  if (outfile[0] != NULL) fclose(outfile[0]);
  if (outfile[1] != NULL) fclose(outfile[1]);
  if (outfile[2] != NULL) fclose(outfile[2]);
  if (outfile[3] != NULL) fclose(outfile[3]);

  if(buf_a != NULL)
    free(buf_a);
  if(buf_b != NULL)
    free(buf_b);
  if(buf_c != NULL)
    free(buf_c);
  if(buf_d != NULL)
    free(buf_d);

  //printf("Press 0 followed by ENTER to exit.\n");
  //scanf("%d", &exit);

  return;
}

void adq412_streaming(void *adq_cu, int adq_num)
{
  unsigned int buffers_filled = 0;
  unsigned int buffers_filled_old = 0;
  unsigned int buffers_filled_max = 0;
  unsigned int buffers_filled_max_old = 0;
  unsigned int buffer_retries = 0;
  int collect_result;
  unsigned int data_counter = 0; //to check for lost data
  unsigned int data_counter_start = 0; //to check for lost data
  unsigned int collect_result_retries = 0;
  unsigned int test_pattern = 0;
  unsigned int test_pattern_verify = 0;
  unsigned int nofbuffers= 0;
  unsigned int samples_in_buffer;
  unsigned int data_sets_to_collect;
  signed short* data_stream_target = NULL;
  int WriteToFile;
  unsigned int stop = 0;
  unsigned int trig_mode;
  unsigned int Int_Trig_Freq;
  unsigned int interleaving_mode = 0;

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;

  FILE* outfile[4], *outfileBin;
  outfile[0] = NULL;
  outfile[1] = NULL;
  outfile[2] = NULL;
  outfile[3] = NULL;
  outfileBin = NULL;

//======================== This section is for the lazy developer who does not want to enter every value one by one=======

  test_pattern = 1;
  trig_mode = 2;
  data_sets_to_collect = 0;
  nofbuffers = 80;
  WriteToFile = 1;
  test_pattern_verify = 1;
  interleaving_mode = 0;
  Int_Trig_Freq = 500;
//========================================================================================================================

  printf("\n=======================================================================\n");
  printf("This streaming example requires that the running firmware contains the\ncorrect user_logic module with the appropriate test pattern and data\nrate control logic.\n");
  printf("=======================================================================\n");

  //Setup ADQ
  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

  CHECKADQ(ADQ_SetDataFormat(adq_cu, adq_num, 2)); // Unpacked 16-bit
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, 0x01, 0x00000000, 0x00000000, NULL)); //Restore mux by clearing user reg 1

  CHECKADQ(ADQ_SetTransferTimeout(adq_cu, adq_num, 1000));

  printf("\nChoose test pattern mode.\n 0 = Raw ADC data\n 1 = Test pattern\n");
  scanf("%u", &test_pattern);
  if (test_pattern)
  {
    outfile[0] = fopen("data.out", "w");
    if(outfile[0] == NULL)
    {
      printf("Error: Failed to open output files.\n");
      return;
    }

    printf("\nVerify test pattern? (0 = no verification, 1 = verify)\n");
    scanf("%u", &test_pattern_verify);
  }
  else
  {
    outfile[0] = fopen("dataA.out", "w");
    outfile[1] = fopen("dataB.out", "w");
    outfile[2] = fopen("dataC.out", "w");
    outfile[3] = fopen("dataD.out", "w");
    if(outfile[0] == NULL || outfile[1] == NULL || outfile[2] == NULL || outfile[3] == NULL)
    {
      printf("Error: Failed to open output files.\n");
      return;
    }

    //The following counter pattern canbe used to identify the channel data and sample order in the buffer.
    //Only counting up is used to make it easier to identify the sanmple order.

    ////Output counter value on channel A
    //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12332, 0xFFFFFF00, 4));
    ////Output counter value on channel B
    //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12332, 0xFFFFFF00, 5));
    ////Output counter value on channel C
    //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12332, 0xFFFFFF00, 6));
    ////Output counter value on channel D
    //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12332, 0xFFFFFF00, 7));
  }
  outfileBin = fopen("data.bin", "wb");
  if(outfileBin == NULL)
  {
    printf("Error: Failed to open output files.\n");
    return;
  }

  //User reg 1 bit 1 high. Enables mux in user logic to send 16 bit values
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, 0x01, 0xFFFFFFC0, 0x00000020, NULL)); // Set data format to 16 bit and turn on trigger as data_valid gate in user_logic
  if (test_pattern)
  {
    CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, 0x01, 0xFFFFFFFD, 0x00000002, NULL)); //Turn on test pattern streaming in user_logic
  }

  printf("\nChoose DATA RATE CONTROL MODE.\n 2 = External Trigger Mode\n 4 = Internal Trigger Mode\n");
  scanf("%u", &trig_mode);

  switch (trig_mode)
  {
  case ADQ_EXT_TRIGGER_MODE :
    {
      CHECKADQ(ADQ_SetConfigurationTrig(adq_cu, adq_num, 0x00, 0, 0));    //Disable internal trigger and trigout port
      CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));          //enable the external trigger so that it can propagate to the user logic_module
      CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, 0x01, 0xFFFFFFF7, 0x00000008, NULL)); //Turn on external trigger as gate in user_logic
      break;
    }
  case ADQ_INTERNAL_TRIGGER_MODE:
    {
      CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, 1));      //Turn OFF external trigger mode in case it might be ON.

      printf("\nChoose Internal Trigger Frequency (Hz)\n");
      scanf("%u", &Int_Trig_Freq);
      CHECKADQ(ADQ_SetInternalTriggerFrequency(adq_cu, adq_num,(unsigned int) (Int_Trig_Freq)));
      CHECKADQ(ADQ_SetConfigurationTrig(adq_cu, adq_num, 0x45, 960, 0));             // enable 960 ns pulse length of internal trigger to trigout port, this parameter should be multible of 32 ns
      break;
      }
  default:
    goto error;
    break;
  }

  printf("\nWrite to file?.\n 0 = No\n 1 = Yes\n");
  scanf("%d", &WriteToFile);

  printf("\nRun collection in interleaved mode?\n(Please be aware of the reverted channel order marking between USB and P*Ie)\n0 = Non-interleave\n1 = Active ports A, C\n2 = Active ports B, D\n4 = Active ports A, B, C, D\n");
  scanf("%u", &interleaving_mode);

  CHECKADQ(ADQ_SetInterleavingMode(adq_cu, adq_num, interleaving_mode));

  printf("\nSet a limit of data sets to retreave before terminating (ins steps of 2048):\n0 = Infinite streaming\n");
  scanf("%u", &data_sets_to_collect);

  //131072 bytes is maximum buffer size. But number of buffers can also be used to increase performance.
  //Default number of buffers is 8, if not set by user.
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, nofbuffers, 131072));

  samples_in_buffer = 131072/2;     //2 bytes per sample

  // Writing data to disk inside the acquisition loop will decrease the streaming performance. If you have enough ram and you want to
  // improve the performace of the acquisition, make this buffer really big and store all your collected data in this buffer first.
  // Once all data has been collected and the acquisition loop has exited, you can write it to file.
  CHECKADQ(data_stream_target = (signed short*)malloc(samples_in_buffer*sizeof(signed short)));    //65536 samples @ 2 bytes = 131072 bytes per buffer

#ifndef LINUX
  printf("\nPress any key to START streaming!\nPress any key again to STOP streaming!");

  while (!_kbhit())
  {
    Sleep(2);
  }
  _getch();      //reset _kbhit

#else
  printf("\nPress ENTER to START streaming!\nPress ENTER again to STOP streaming!\n");
  changemode(1);
  while (!_kbhit())
  {
    Sleep(2);
  }
  changemode(0);
#endif

  //CHECKADQ(ADQ_SetTransferTimeout(adq_cu, adq_num, 1000));
  printf("\nSetting up streaming...");
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));
  printf("\nDone.");

  printf("Collecting data, please wait...\n");

  // Start streaming by arming
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  do
  {
    CHECKADQ(ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled));
    if(buffers_filled_max < buffers_filled)
      buffers_filled_max = buffers_filled;

    //This if case is just for updating the buffer status if the status has changed.
    //This is to prevent filling the screen with unnecessary printout.
    if ((buffers_filled != buffers_filled_old) || (buffers_filled_max != buffers_filled_max_old))
    {
      printf("Buffers currently filled: %3u (peak %3u).\n", buffers_filled, buffers_filled_max);
      buffers_filled_old = buffers_filled;
      buffers_filled_max_old = buffers_filled_max;
    }

    if(buffers_filled  > 0)
    {
      buffer_retries = 0;       //reset retry counter

      collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
      if (collect_result)
      {
        collect_result_retries = 0;       //reset retry counter

        //printf("Samples per buffer: %u (%u data sets per buffer) \n", samples_in_buffer, samples_in_buffer/32);      //Each data set is 32 samples per clock cycle

        // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
        memcpy((void*)&data_stream_target[0], ADQ_GetPtrStream(adq_cu, adq_num), samples_in_buffer*sizeof(signed short));

        if (test_pattern)
        {
          unsigned int LoopVar;
          for (LoopVar = 0; LoopVar < samples_in_buffer; LoopVar+=32)
          {
            if (test_pattern_verify)
            {
              //Set initial value to start the comparision. This is a better way to check for data lost because depending on implementation the initial value might differ
              //but it does not mean that data has been lost.
              if (data_counter == 0)
              {
                data_counter_start = (((unsigned short)data_stream_target[1] << 16) | (unsigned short)data_stream_target[0]);
                data_counter = data_counter_start;
              }

              if (data_counter != (unsigned int)(((unsigned short)data_stream_target[LoopVar + 1] << 16) | (unsigned short)data_stream_target[LoopVar])) // check for lost data. This must not happen!
              {
                fprintf(outfile[0], "DATA LOST DETECTED!!!");
                printf("DATA LOST DETECTED!!! STREAMING TERMINATE...\n");
                data_counter = (((unsigned short)data_stream_target[LoopVar + 1] << 16) | (unsigned short)data_stream_target[LoopVar]);
                stop = 1;
              }

              if (WriteToFile == 1)
              {
                int ik;
                fprintf(outfile[0], "%10u\t", (unsigned int) ((unsigned short)data_stream_target[LoopVar+1] << 16) | ((unsigned short)data_stream_target[LoopVar]));  // data block counter. Each is 32 samples large.
                fprintf(outfile[0], "%10u\t", (unsigned int) ((unsigned short)data_stream_target[LoopVar+3] << 16) | (unsigned short)data_stream_target[LoopVar+2]); // data block time in untis of 8 ns

                // comment the for-block below if you don't want to write all other samples to file

                for(ik =4; ik<32; ik++)
                {
                  fprintf(outfile[0], "%04x ", (unsigned short)data_stream_target[LoopVar+ik]);
                }
                fprintf(outfile[0], "\n");
              }

            }
            else    //if test pattern is used but no verification is needed
            {
              if (WriteToFile == 1)
              {
                int ik;
                 for(ik =0; ik<32; ik++)
                {
                  fprintf(outfile[0], "%04x ", (unsigned short)data_stream_target[LoopVar+ik]);
                }
                fprintf(outfile[0], "\n");
              }
            }

            data_counter++;
          }

          //Print onscreen the current samples count that has been collected
          printf("Data sets retreaved: %u\n", data_counter-data_counter_start);
        }
        else // Raw ADC data
        {
          if (WriteToFile == 1)
          {
            unsigned int LoopVar;
            short *data_ptr = data_stream_target;
            fwrite(data_stream_target, sizeof(short), samples_in_buffer, outfileBin);

            // Data is stored interleaved as 8 samples for channel A, 8 for B, 8 for C and 8 for D
            for (LoopVar = 0; LoopVar < samples_in_buffer; LoopVar+=32)
            {
              unsigned int LoopVarCh;
              for (LoopVarCh = 0; LoopVarCh < 4; LoopVarCh++)
              {
                unsigned int LoopVar2;
                for (LoopVar2 = 0; LoopVar2 < 8; LoopVar2++)
                {
                  fprintf(outfile[LoopVarCh], "%d\n", *data_ptr++);
                }
              }
            }
          }

          data_counter = data_counter + (samples_in_buffer/32);
          printf("Data sets retreaved: %u\n", data_counter-data_counter_start);
        }
      }
      else  //If collect_result failed
      {
        collect_result_retries++;
        printf("WARN: Failed to collect data from DMA buffer, retries = %u.\n", collect_result_retries);

      }
    }
    else    //No buffer filled
    {
      buffer_retries++;
      if ( (buffer_retries % 1000) == 0 )
        printf("WARN: No buffer filled! Incorrect setup or no triggers? buffer_retries = %u\n", buffer_retries);
    }

    if (ADQ_GetStreamOverflow(adq_cu, adq_num))
    {
      printf("Warning: Streaming Overflow 1!\n");
      stop = 1;
      collect_result = 0;
    }

    if ( _kbhit() || ( (data_counter - data_counter_start >= data_sets_to_collect) && (data_sets_to_collect > 0) ) )
    {
      printf("\nTerminate streaming! Please wait!\n");
      stop = 1;
    }

  } while (stop == 0);

  if (WriteToFile == 1)
    printf("\n%u data sets stored to file.\n\n", data_counter-data_counter_start);
  else
    printf("\n%u data sets retreaved.\n\n", data_counter-data_counter_start);

error:

  // Only disarm trigger after data is collected
  ADQ_DisarmTrigger(adq_cu, adq_num);

  // Disable streaming bypass of DRAM
  ADQ_SetStreamStatus(adq_cu, adq_num,0);

  //Restore dataformat
  ADQ_SetDataFormat(adq_cu, adq_num, 0);

  //This reset level will make it possible to switch back to multi-record mode again, assuming that Devkit users has emptied all buffers in user_logic.
  ADQ_ResetDevice(adq_cu, adq_num, 3);

  free(data_stream_target);

  if (outfile[0] != NULL) fclose(outfile[0]);
  if (outfile[1] != NULL) fclose(outfile[1]);
  if (outfile[2] != NULL) fclose(outfile[2]);
  if (outfile[3] != NULL) fclose(outfile[3]);
  if (outfileBin != NULL) fclose(outfileBin);

  return;
}

void adq412_microtca_functionality_demo(void *adq_cu, int adq_num)
{
  int exit=0;
  printf("\n\nThis example sets up the additional hardware present on MicroTCA ADQ boards.\n\n");
  printf("Running this example on ADQs with other interfaces (USB/PCIE/PXIE) will \ncause the functions to return errors.\n\n");

  printf("Setting MTCA board PLLs to: \n\n");

  printf("10G Ethernet clock:   156.25 MHz\n");
  printf("1G Ethernet clock:    125 MHz\n");
  printf("Point-to-point clock: 250 MHz\n\n");

  /* Simple PLL setup: */

  CHECKADQ(ADQ_SetEthernetPllFreq(adq_cu, adq_num,ETH10_FREQ_156_25MHZ,ETH1_FREQ_125MHZ));  // Set 10G ethernet clocks to 156.25 MHz and 1G clocks to 125 MHz
  CHECKADQ(ADQ_SetPointToPointPllFreq(adq_cu, adq_num,PP_FREQ_250MHZ));                     // Set point-to-point interface clocks to 250 MHz

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

void adq412_triggeredstreamingexample(void *adq_cu, int adq_num)
{
  unsigned int WriteToFile = 1;
  char *serialnumber;
  int trig_mode;
  unsigned int int_trig_freq;
  unsigned int interleaving_mode = 0;
  unsigned int channels_mask = 0x0;
  unsigned int channel_active = 0;
  unsigned int nof_records_per_channel = 1;
  unsigned int nof_records_total = 1;
  unsigned int nof_records_max;
  unsigned int nof_samples_per_record = 1024;
  unsigned int stream_option = 1;
  unsigned int sample_size_bytes = 2;
  unsigned int nof_buffers;
  unsigned int pretrig = 0;
  unsigned int triggerdelay = 0;
  unsigned int NofRecordsCompleted = 0;
  unsigned int trigs = 0;
  unsigned int nof_trigs_max;
  void* data_target[4];
  unsigned int nof_channels = 4;
  unsigned int ts_header_size;
  void* header_target = NULL;
  unsigned int NofRecordsRead = 0;
  unsigned int in_idle = 0;
  unsigned int trigger_skipped = 0;
  unsigned int overflow = 0;
  unsigned int nof_wait = 0;
  unsigned int nof_wait_max = 100000;

  unsigned int nof_dram_words;
  unsigned char* dump_buffer;
  unsigned int nof_bytes_read = 0;
  unsigned int data_offset[4] = {0, 0, 0, 0};

  unsigned long long Timestamp = 0;
  unsigned int Channel = 0;
  int RegisterValue = 0;
  unsigned int SerialNumber = 0;
  unsigned int RecordCounter = 0;
  unsigned int batch;
  int exit=0;

#ifndef LINUX
    LARGE_INTEGER frequency;
    LARGE_INTEGER start_val;
    LARGE_INTEGER end_val;
#else
    struct timeval start_val;
    struct timeval end_val;
#endif

  unsigned int individual_mode = 0;
  int trig_levels[4] = {0, 0, 0, 0};
  int trig_edges[4] = {1, 1, 1, 1};
  int trig_reset_levels[4] = {
    (int)ADQ_LEVEL_TRIGGER_USE_DEFAULT_RESETLEVEL,
    (int)ADQ_LEVEL_TRIGGER_USE_DEFAULT_RESETLEVEL,
    (int)ADQ_LEVEL_TRIGGER_USE_DEFAULT_RESETLEVEL,
    (int)ADQ_LEVEL_TRIGGER_USE_DEFAULT_RESETLEVEL};

  double start_time_micro;
  double end_time_micro;

  // Check if the unit has triggered streaming.
  CHECKADQ(ADQ_HasTriggeredStreamingFunctionality(adq_cu, adq_num));
  if (ADQ_HasTriggeredStreamingFunctionality(adq_cu, adq_num) == 0)
  {
    printf("This example is not supported by the firmware on the ADQ, exiting..\n");
    goto error;
  }

  serialnumber = ADQ_GetBoardSerialNumber(adq_cu, adq_num);

  printf("Device Serial Number: %s\n",serialnumber);

  trig_mode = ADQ_SW_TRIGGER_MODE;
  printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level Trigger Mode\n %d = Internal Trigger Mode\n", ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE, ADQ_INTERNAL_TRIGGER_MODE);
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
      printf("Do you wish to use individual setups for the different channels available? 1 = Yes, 0 = No\n");
      scanf("%u", &individual_mode);
      if (!individual_mode)
      {
        printf("\nChoose trig level. -2048 < trig level < 2047\n");
        scanf("%i", &trig_levels[0]);
        printf("\nChoose trig edge. 1 = Rising edge 0 = Falling edge\n");
        scanf("%i", &trig_edges[0]);
      }
      else
      {
        for (unsigned int ch = 0; ch < 4; ch++)
        {
          printf("\nChoose trig level for channel %c. -2048 < trig level < 2047\n", ch+65);
          scanf("%i", &trig_levels[ch]);
          printf("\nChoose trig edge for channel %c. 1 = Rising edge, 0 = Falling edge\n", ch+65);
          scanf("%i", &trig_edges[ch]);
        }
      }
      CHECKADQ(ADQ_SetupLevelTrigger(adq_cu, adq_num, trig_levels, trig_edges, trig_reset_levels, 0xf, individual_mode));
      break;
      }
    case ADQ_INTERNAL_TRIGGER_MODE: {
      CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));

      printf("\nChoose Internal Trigger Frequency (Hz)\n");
      scanf("%u", &int_trig_freq);
      CHECKADQ(ADQ_SetInternalTriggerFrequency(adq_cu, adq_num,int_trig_freq));
      break;
      }
    default :
      return;
      break;
  }

  printf("\nRun collection in interleaved mode? 1 = yes, 0 = no\n");
  scanf("%u", &interleaving_mode);

  CHECKADQ(ADQ_SetInterleavingMode(adq_cu, adq_num, interleaving_mode));

  if (interleaving_mode == 0)
  {
    printf("\nActivate Channel A collection? 1 = yes, 0 = no\n");
    scanf("%u", &channel_active);
    channels_mask = channels_mask | (channel_active != 0);
    printf("\nActivate Channel B collection?\n");
    scanf("%u", &channel_active);
    channels_mask = channels_mask | ((channel_active != 0) << 1);
    printf("\nActivate Channel C collection?\n");
    scanf("%u", &channel_active);
    channels_mask = channels_mask | ((channel_active != 0) << 2);
    printf("\nActivate Channel D collection?\n");
    scanf("%u", &channel_active);
    channels_mask = channels_mask | ((channel_active != 0) << 3);
  }
  else if (interleaving_mode == 1)
  {
    printf("\nActivate Channel A collection? 1 = yes, 0 = no\n");
    scanf("%u", &channel_active);
    channels_mask = channels_mask | (channel_active != 0);
    printf("\nActivate Channel C collection?\n");
    scanf("%u", &channel_active);
    channels_mask = channels_mask | ((channel_active != 0) << 2);
  }
  else
  {
    channels_mask = 0xf; // All channels
  }

  if(interleaving_mode == 0)
  {
    printf("\nSelect the number of samples for each record. The streaming header will overwrite 8 samples\n");
  }
  else
  {
    printf("\nSelect the number of samples for each record. The streaming header will overwrite 16 samples\n");
  }
  scanf("%u", &nof_samples_per_record);
  printf("\nSelect the maximum number of records to collect for each active channel\n");
  scanf("%u", &nof_records_per_channel);
  nof_records_max = nof_records_per_channel*(((channels_mask & 1) == 1) + ((channels_mask & 2) == 2) + ((channels_mask & 4) == 4) + ((channels_mask & 8) == 8));
  printf("\nSelect the total number of records to collect\n");
  scanf("%u", &nof_records_total);
  while (nof_records_total > nof_records_max)
  {
    printf("Too high number of records, must be at most %u. Please select new value\n", nof_records_max);
    scanf("%u", &nof_records_total);
  }



  printf("\nStream directly (1) or redirect data to DRAM (2)?\n");
  scanf("%u", &stream_option);

  if (stream_option == 1)
  {
    // Cap the number of DMA buffers to 8 in this example. This can be increased if enough memory can be allocated on the system.
    // If the number of buffers is set to the number of records to transfer, transfer of data will newer stop due to the host not being able to receive the data
    nof_buffers = MIN(nof_records_total, 8);
    CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, nof_buffers, nof_samples_per_record*sample_size_bytes));
  }

  CHECKADQ(ADQ_TriggeredStreamingSetup(adq_cu, adq_num, nof_records_per_channel, nof_samples_per_record, pretrig, triggerdelay, channels_mask));
  CHECKADQ(ADQ_SetTriggeredStreamingTotalNofRecords(adq_cu, adq_num, nof_records_total));

  if (stream_option == 1)
  {
    // Normal streaming

    // Check that the transfer size is OK for the interface
    if (ADQ_IsUSBDevice(adq_cu, adq_num))
    {
      if (nof_samples_per_record * sizeof(short) < 512)
      {
        printf("The minimum transfer size is 512 bytes, adjusting number of samples..\n");
        nof_samples_per_record = 256;
      }
    }
    else // if PCIe
    {
      if (nof_samples_per_record * sizeof(short) < 128)
      {
        printf("The minimum transfer size is 128 bytes, adjusting number of samples..\n");
        nof_samples_per_record = 64;
      }
    }
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 7));
  }
  else
  {
    // Dump data in DRAM and read later
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 9));
  }
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
  ///////////////////////////////////////////////////////
  // Start counter before arming triggered streaming

#ifndef LINUX
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start_val);
#else
    gettimeofday(&start_val, NULL);
#endif
  /////////////////////////////////////////////////////////
  CHECKADQ(ADQ_TriggeredStreamingArm(adq_cu, adq_num));

  printf("\nCollection started..\n");
  if (trig_mode == ADQ_SW_TRIGGER_MODE)
  {
    nof_trigs_max = nof_records_per_channel * 100;
    while ((NofRecordsCompleted < nof_records_total) && (trigs < nof_trigs_max))
    {
      // Perform a special Triggered-Streaming SW-trig
      CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12343, ~0x0010, 0x0010));
      CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12343, ~0x0010, 0x0000));
      CHECKADQ(ADQ_TriggeredStreamingGetNofRecordsCompleted(adq_cu, adq_num, channels_mask, &NofRecordsCompleted));
      trigs++;
    }
    if (trigs >= nof_trigs_max)
    {
      printf("Error: triggering seems to be stuck!");
      goto error;
    }
  }

  for(unsigned int ch = 0; ch < nof_channels; ch++)
  {
    data_target[ch] = (void*) malloc(nof_records_per_channel * nof_samples_per_record*sizeof(short));
  }

  ts_header_size = ADQ_GetTriggeredStreamingHeaderSizeBytes(adq_cu, adq_num);
  header_target = malloc(nof_records_total * ts_header_size);

  if (stream_option == 1)
  {
    // Start reading the records to host. In this example, assume that all records are completed or will be so soon.
    CHECKADQ(ADQ_GetTriggeredStreamingRecords(adq_cu, adq_num, nof_records_total, data_target, header_target, &NofRecordsRead));
    // Get status before disarm
    // Stop counter
#ifndef LINUX
    QueryPerformanceCounter(&end_val);
    start_time_micro = start_val.QuadPart * (1000000.0 / frequency.QuadPart);
    end_time_micro = end_val.QuadPart * (1000000.0 / frequency.QuadPart);
#else
    gettimeofday(&end_val, NULL);
    start_time_micro = (start_val.tv_sec * 1000000.0) + start_val.tv_usec;
    end_time_micro = (end_val.tv_sec * 1000000.0) + end_val.tv_usec;
#endif
    CHECKADQ(ADQ_TriggeredStreamingGetStatus(adq_cu, adq_num, &in_idle, &trigger_skipped, &overflow));
    CHECKADQ(ADQ_TriggeredStreamingDisarm(adq_cu, adq_num));
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 0));
  }
  else
  {
    CHECKADQ(ADQ_TriggeredStreamingGetNofRecordsCompleted(adq_cu, adq_num, channels_mask, &NofRecordsCompleted));
    while ((NofRecordsCompleted < nof_records_total) && (nof_wait < nof_wait_max))
    {
      // We must collect all records before disarming
      Sleep(10);
      CHECKADQ(ADQ_TriggeredStreamingGetNofRecordsCompleted(adq_cu, adq_num, channels_mask, &NofRecordsCompleted));
      nof_wait++;
    }
    if (nof_wait >= nof_wait_max)
    {
      printf("Collection seem to have stuck.. aborting \n");
      goto error;
    }
    // Get status before disarm
    CHECKADQ(ADQ_TriggeredStreamingGetStatus(adq_cu, adq_num, &in_idle, &trigger_skipped, &overflow));
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 0));

    nof_dram_words = nof_records_total * sample_size_bytes * nof_samples_per_record / (512 / 8); // 512 bits per DRAM address
    nof_dram_words = ((nof_dram_words + 31) / 32)*32; // MemoryDump reads in batches of 32 addresses
    dump_buffer = (unsigned char*) malloc(nof_dram_words * (512/8));

    CHECKADQ(ADQ_MemoryDump(adq_cu, adq_num, 0, nof_dram_words-1, dump_buffer, &nof_bytes_read, 0));
    CHECKADQ(ADQ_MemoryShadow(adq_cu, adq_num, dump_buffer, nof_bytes_read));
    CHECKADQ(ADQ_GetTriggeredStreamingRecords(adq_cu, adq_num, nof_records_total, data_target, header_target, &NofRecordsRead))
    CHECKADQ(ADQ_TriggeredStreamingDisarm(adq_cu, adq_num));
    free((void*) dump_buffer);
    // Stop counter
#ifndef LINUX
    QueryPerformanceCounter(&end_val);
    start_time_micro = start_val.QuadPart * (1000000.0 / frequency.QuadPart);
    end_time_micro = end_val.QuadPart * (1000000.0 / frequency.QuadPart);
#else
    gettimeofday(&end_val, NULL);
    start_time_micro = (start_val.tv_sec * 1000000.0) + start_val.tv_usec;
    end_time_micro = (end_val.tv_sec * 1000000.0) + end_val.tv_usec;
#endif
  }

  for (batch = 0; batch < nof_records_total; batch++)
  {

     CHECKADQ(ADQ_ParseTriggeredStreamingHeader(adq_cu, adq_num, (void*) ((char*) header_target + batch*ts_header_size), &Timestamp, &Channel, NULL, &RegisterValue, &SerialNumber, &RecordCounter));
     printf("Waveform header:\n");
     printf("    Record counter:                %u\n", RecordCounter);
     printf("    Channel:                       %u\n", Channel);
     printf("    Timestamp:                     %llu\n", Timestamp);
     printf("    User specified register value: %i\n", RegisterValue);
     printf("    Serial number:                 %u\n", SerialNumber);
     if (WriteToFile > 0)
     {
       FILE* outfile = NULL;
       char fname[256];
       unsigned int ch = 0;
       switch (Channel)
       {
       case 1:
         ch = 0;
         break;
       case 2:
         ch = 1;
         break;
       case 4:
         ch = 2;
         break;
       case 8:
         ch = 3;
         break;
       default:
         // Error
         ch = 0xf;
         break;
       }
       if (ch == 0xf) {
          printf("ERROR: Invalid channel: %u\n", Channel);
       } else {
          sprintf(fname, "TS_data_nr%u_header.m", batch);
          outfile = fopen(fname, "wb");
          fprintf(outfile, "Batch_No = %u;\nChannel_No = %04u;\nRecord_No = %04u;\n", batch, ch, RecordCounter);
          fclose(outfile);
          outfile = NULL;
          printf("    Wrote this records header to file %s\n", fname);

          sprintf(fname, "TS_data_nr%u.out", batch);
          outfile = fopen(fname, "wb");
          fwrite((void*) ((char*) data_target[ch] + (sizeof(short) * data_offset[ch])), sizeof(short), nof_samples_per_record - ts_header_size/2, outfile);
          data_offset[ch] += nof_samples_per_record - ts_header_size/2;
          fclose(outfile);
          outfile = NULL;
          printf("    Wrote this record to file %s\n", fname);
       }
     }
  }
  printf("\n\nTotal time spent in collection (ms): %f\n", (end_time_micro - start_time_micro)*0.001);
  printf("Use the script ADQ_TriggeredStreaming_Plot.m to plot the results using matlab\n\n");

  // Print status, read out before disarming.
  printf("Status of triggered streaming unit:\n");
  printf("    In Idle: %u (expected is 1 = yes)\n", in_idle);
  printf("    Trigger skipped: %u (expected is 0 = no trigger was skipped on any channel)\n", trigger_skipped);
  printf("    Overflow: %u (expected is 0 = no overflow of data occurred)\n\n", overflow);

  error:

  for(unsigned int ch = 0; ch < nof_channels; ch++)
  {
    free(data_target[ch]);
  }
  free(header_target);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}
