// File: example_ADQ208.cpp
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
#define Sleep(interval) usleep(1000*interval)
#define _kbhit() kbhit_linux()
#include <stdlib.h>
#include <string.h>
#endif

int kbhit_linux (void);
void adq208_multirecord(void *adq_cu, int adq_num);
void adq208_multirecord_continuous(void *adq_cu, int adq_num);
void adq208_microtca_functionality_demo(void *adq_cu, int adq_num);
void adq208_streaming(void *adq_cu, int adq_num);

// Special define
#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void adq208(void *adq_cu, int adq_num)
{
  int mode;
  int* revision = ADQ_GetRevision(adq_cu, adq_num);
  printf("\nConnected to ADQ208 #1\n\n");

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

  printf("\nChoose collect mode.\n 1 = Multi-Record\n 2 = MicroTCA functionality demo\n 3 = Streaming\n 4 = Multi-Record Automatic with instant readback\n");
  scanf("%d", &mode);

  switch (mode)
  {
  case 1:
    adq208_multirecord(adq_cu, adq_num);
    break;
  case 2:
    adq208_microtca_functionality_demo(adq_cu, adq_num);
    break;
  case 3:
    adq208_streaming(adq_cu, adq_num);
    break;
  case 4:
    adq208_multirecord_continuous(adq_cu, adq_num);
  default:
    return;
    break;
  }
}

void adq208_multirecord(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_readmode;
  int trig_mode;
  int trig_level;
  int trig_flank;
  unsigned int samples_per_record;
  unsigned int n_records_collect;
  unsigned int number_of_records;
  unsigned int buffersize;
  int selected_channels;
  int interleaved;
  unsigned int max_nof_samples = 0;
  unsigned int write_to_file = 1;
  unsigned int channelsmask = 0xf; // All channels will be collected

  char* buf_a;
  char* buf_b;

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  unsigned int i;
  void* target_buffers[8]; // GetData allows for a digitizer with max 8 channels, the unused pointers should be null pointers
  FILE* outfile[4];
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

  trig_mode = ADQ_SW_TRIGGER_MODE;
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
  default :
    return;
    break;
  }

  // Uncomment one of these to change sampling frequency
  //ADQ_SetPll(adq_cu, adq_num, 300, 2, 1, 1); // 1500MHz/375.0MHz (6.0GSps)
  //ADQ_SetPll(adq_cu, adq_num, 320, 2, 1, 1); // 1600MHz/400.0MHz (6.4GSps)
  //ADQ_SetPll(adq_cu, adq_num, 350, 2, 1, 1); // 1750MHz/437.5MHz (7.0GSps)

  printf("\nChoose number of records.\n");
  number_of_records = 1;
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  samples_per_record = 4095;
  scanf("%u", &samples_per_record);

  printf("\nChoose number of channels (1 or 2).\n");
  selected_channels = 1;
  scanf("%d", &selected_channels);
  interleaved = (selected_channels==2)?0:1;
  ADQ_SetInterleavingMode(adq_cu, adq_num, interleaved);

  ADQ_GetMaxNofSamplesFromNofRecords(adq_cu, adq_num, number_of_records, &max_nof_samples);
  while((samples_per_record == 0) || (samples_per_record > max_nof_samples))
  {
    printf("\nError: Invalid number of samples.\n");
    printf("\nChoose number of samples per record.\n 1 <= samples <= %u.\n", max_nof_samples);
    scanf("%u", &samples_per_record);
  }

  printf("\nChoose read mode:\n 0 = Wait for all records before readout\n 1 = Perform background readout.\n");
  trig_readmode = 0;
  scanf("%d", &trig_readmode);

  // Use only multirecord mode for data collection.
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num,number_of_records,samples_per_record));

  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  if (trig_readmode == 0)
  {
    if (trig_mode == ADQ_SW_TRIGGER_MODE)
    {
      int trigged;
      printf("Automatically triggering your device to collect data.\n");
      do
      {
        CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
        trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      }while (trigged == 0);
    }
    else
    {
      int trigged;
      printf("\nPlease trig your device to collect data.\n");
      do
      {
        trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      }while (trigged == 0);
    }
  }
  printf("\nDevice trigged\n");

  n_records_collect = 0;

  while(n_records_collect > number_of_records)
  {
    printf("\nError: The chosen number exceeds the number of available records in the memory\n");
    printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
    scanf("%u", &n_records_collect);
  }

  if (n_records_collect == 0)
    n_records_collect = number_of_records;

  buffersize = samples_per_record;
  buf_a = (char*)calloc(buffersize,sizeof(char));
  buf_b = (char*)calloc(buffersize,sizeof(char));
  if(buf_a == NULL)
    goto error;
  if(buf_b == NULL)
    goto error;

  // Create a pointer array containing the data buffer pointers
  target_buffers[0] = (void*)buf_a;
  target_buffers[1] = (void*)buf_b;

  printf("Collecting data, please wait...\n");
  for (i=0; i<n_records_collect; i++)
  {
    if (trig_readmode != 0)
    {
      unsigned int records;
      do
      {
        if (trig_mode == ADQ_SW_TRIGGER_MODE)
        {
          CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
        }
        records = ADQ_GetAcquiredRecords(adq_cu, adq_num);
        printf("\rRecords trigged: %u", records);
      }while (records <= i);
    }

    // Each data buffer must contain enough samples to store all the records consecutively
    buffersize = samples_per_record;
    // Use the GetData function
    CHECKADQ(ADQ_GetData(adq_cu, adq_num,target_buffers,buffersize,sizeof(char),i,1,channelsmask,0,samples_per_record,ADQ_TRANSFER_MODE_NORMAL));

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
            fprintf(outfile[channel], "%d\n", ((char*)target_buffers[channel])[j]);
        }
      }
    }
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  if (!interleaved) {
    printf("\n\n Done. Samples stored in data1.out (channel 1) and data2.out (channel 2).\n");
  }
  else {
    printf("\n\n Done. Samples stored in data1.out.\n");
  }


error:

  if (outfile[0] != NULL)
    fclose(outfile[0]);
  if (outfile[1] != NULL)
    fclose(outfile[1]);

  if(buf_a != NULL)
    free(buf_a);
  if(buf_b != NULL)
    free(buf_b);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}

void adq208_multirecord_continuous(void *adq_cu, int adq_num)
{
  //Setup ADQ
  int trig_readmode;
  int trig_mode;
  int trig_level;
  int trig_flank;
  int Int_Trig_Freq;
  unsigned int samples_per_record;
  unsigned int n_records_collect;
  unsigned int number_of_records;
  char* data_ptr_addr0;
  char* data_ptr_addr1;
  unsigned int max_nof_samples = 0;
  int exit=0;

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;
  FILE* outfile = NULL;
  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

  outfile = fopen("data.out", "w");
  if(outfile == NULL)
  {
    printf("Error: Failed to open output file.\n");
    return;
  }

  trig_mode = ADQ_SW_TRIGGER_MODE;
  printf("\nChoose trig mode.\n %d = SW Trigger Mode\n %d = External Trigger Mode\n %d = Level Trigger Mode\n %d = Internal trigger mode\n",
    ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE, ADQ_INTERNAL_TRIGGER_MODE);
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
    printf("\nChoose trig level.\n 0 <= level <= 255\n");
    trig_level = 127;
    //scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n 1 = Rising edge\n 0 = Falling edge\n");
    trig_flank = 1;
    //scanf("%d", &trig_flank);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_flank));
    printf("\n");
    break;
                                }
  case ADQ_INTERNAL_TRIGGER_MODE: {
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose Internal Trigger Frequency (Hz)\n");
    scanf("%d", &Int_Trig_Freq);
    CHECKADQ(ADQ_SetInternalTriggerFrequency(adq_cu, adq_num,(unsigned int) (Int_Trig_Freq)));
    break;
                                  }
  default :
    return;
    break;
  }

  // Uncomment one of these to change sampling frequency
  //ADQ_SetPll(adq_cu, adq_num, 300, 2, 1, 1); // 1500MHz/375.0MHz (6.0GSps)
  //ADQ_SetPll(adq_cu, adq_num, 320, 2, 1, 1); // 1600MHz/400.0MHz (6.4GSps)
  //ADQ_SetPll(adq_cu, adq_num, 350, 2, 1, 1); // 1750MHz/437.5MHz (7.0GSps)

  printf("\nChoose number of records.\n");
  number_of_records = 1;
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  samples_per_record = 4000;
  scanf("%u", &samples_per_record);

  ADQ_GetMaxNofSamplesFromNofRecords(adq_cu, adq_num, number_of_records, &max_nof_samples);
  while((samples_per_record == 0) || (samples_per_record > max_nof_samples))
  {
    printf("\nError: Invalid number of samples.\n");
    printf("\nChoose number of samples per record.\n 1 <= samples <= %u.\n", max_nof_samples);
    scanf("%u", &samples_per_record);
  }

  printf("\nChoose read mode:\n 0 = Wait for all records before readout\n 1 = Perform background readout.\n 2 = Background readout with auto-rearm\n");
  trig_readmode = 0;
  scanf("%d", &trig_readmode);

  // Use only multirecord mode for data collection.
  if ((trig_readmode == 0) || (trig_readmode == 1))
  {
    CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num,number_of_records,samples_per_record));
  }
  else if (trig_readmode == 2)
  {
    unsigned int mrinfo[10];
    mrinfo[0] = 0x3;
    CHECKADQ(ADQ_MultiRecordSetupGP(adq_cu, adq_num,number_of_records,samples_per_record, mrinfo));
  }

  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  if (trig_readmode == 0)
  {
    if (trig_mode == ADQ_SW_TRIGGER_MODE)
    {
      int trigged;
      printf("Automatically triggering your device to collect data.\n");
      do
      {
        CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
        trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      }while (trigged == 0);
    }
    else
    {
      int trigged;
      printf("\nPlease trig your device to collect data.\n");
      do
      {
        trigged = ADQ_GetAcquiredAll(adq_cu, adq_num);
      }while (trigged == 0);
    }
  }

  printf("\nDevice trigged\n");

  n_records_collect = 0;

  while(n_records_collect > number_of_records)
  {
    printf("\nError: The chosen number exceeds the number of available records in the memory\n");
    printf("Choose how many records to collect.\n 1 <= records <= %u, 0 == All in ADQ buffer.\n", number_of_records);
    scanf("%u", &n_records_collect);
  }
  if (n_records_collect == 0)
    n_records_collect = number_of_records;

  // Get pointer to non-streamed unpacked data
  data_ptr_addr0 = (char *)ADQ_GetPtrData(adq_cu, adq_num,1);
  data_ptr_addr1 = (char *)ADQ_GetPtrData(adq_cu, adq_num,2);

  printf("Collecting data, please wait...\n");
  if (trig_readmode < 2)
  {
    unsigned int i;
    for (i=0; i<n_records_collect; i++)
    {
      unsigned int samples_to_collect = samples_per_record;
      if (trig_readmode != 0)
      {
        unsigned int records;
        do
        {
          if (trig_mode == ADQ_SW_TRIGGER_MODE)
          {
            CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
          }
          records = ADQ_GetAcquiredRecords(adq_cu, adq_num);
          printf("\rRecords trigged: %u", records);
        } while (records <= i);
      }

      while (samples_to_collect > 0)
      {
        int collect_result = ADQ_CollectRecord(adq_cu, adq_num, i);
        unsigned int samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

        if (collect_result)
        {
          unsigned int j;
          for (j=0; j<samples_in_buffer; j++)
          {
            char data0 = data_ptr_addr0[j];
            char data1 = data_ptr_addr1[j];
            fprintf(outfile, "%d\t%d\n", data0, data1);
          }
          samples_to_collect -= samples_in_buffer;
        }
        else
        {
          printf("\n\nCollect next data page failed!\n");
          samples_to_collect = 0;
          i = n_records_collect;
        }
      }
    }
  }
  else
  {
    unsigned int TotalRead = 0;
    unsigned int records;
    unsigned int loops;
    unsigned int lastread_record = 0;
    unsigned int lastread_loop = 0;
    unsigned int quitnow = 0;
    int avail;
    int lastprintedavail = 1;
    //Empty the input buffer
#ifndef LINUX
    while (_kbhit())
    {
      _getch();
    }
#endif

    do
    {
      ADQ_GetAcquiredRecordsAndLoopCounter(adq_cu, adq_num, &records, &loops);

      avail = (loops - lastread_loop)*n_records_collect - lastread_record + records;
      if ((avail > 0) || (lastprintedavail != 0))
      {
        printf("Available records: %06d records [L=%02u, R=%06u] [TotalRead=%06u]\n", avail, loops, records, TotalRead);
        lastprintedavail = avail;
      }
      if (avail >= (int)n_records_collect)
      {
        printf("OVERFLOW: Data is overwritten!. Aborting.\n");
        avail = 0;
        quitnow = 1;
      }
      else if (avail == 0)
      {
        Sleep(1);
      }

      while (avail > 0)
      {
        unsigned int read_index = lastread_record;

        unsigned int samples_to_collect = samples_per_record;
        while (samples_to_collect > 0)
        {
          int collect_result = ADQ_CollectRecord(adq_cu, adq_num, read_index);
          unsigned int samples_in_buffer = MIN(ADQ_GetSamplesPerPage(adq_cu, adq_num), samples_to_collect);

          if (collect_result)
          {
            unsigned int j;
            for (j=0; j<samples_in_buffer; j++)
            {
              char data0 = data_ptr_addr0[j];
              char data1 = data_ptr_addr1[j];
              fprintf(outfile, "%d\t%d\n", data0, data1);
            }
            samples_to_collect -= samples_in_buffer;
          }
          else
          {
            printf("\n\nCollect next data page failed!\n");
            samples_to_collect = 0;
          }
        }

        avail--;
        lastread_record++;
        TotalRead++;
        if (lastread_record == n_records_collect)
        {
          lastread_record = 0;
          lastread_loop++;
        }

      }

    }
    while (!_kbhit() && !quitnow);
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  printf("\n\n Done. Samples stored in data.out.\n");

error:


  if (outfile != NULL)
    fclose(outfile);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}


void adq208_microtca_functionality_demo(void *adq_cu, int adq_num)
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


void adq208_streaming(void *adq_cu, int adq_num)
{
  const unsigned int USER_REG0 = 0;
  const unsigned int USER_REG1 = 1;
  const unsigned int USER_REG2 = 2;
  unsigned int buffers_filled;
  int collect_result;
  unsigned int data_counter = 0; //to check for lost data
  unsigned int data_counter_start = 0; //to check for lost data
  unsigned int data_lost = 0;
  unsigned int abort = 0;
  unsigned int retries = 0;
  unsigned int trig_mode;
  unsigned int Int_Trig_Freq;
  int WriteToFile;
  signed char* data_stream_target = NULL;
  FILE* outfile = NULL;
  int exit=0;


  //Setup ADQ

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0)/256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1)/256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2)/256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3)/256;
  unsigned int tr4 = ADQ_GetTemperature(adq_cu, adq_num, 4)/256;

  outfile = fopen("data.out", "w");
  if(outfile == NULL)
  {
    printf("Error: Failed to open output files.\n");
    return;
  }

  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\tPCB diode: %u\n\n",
    tlocal, tr1, tr2, tr3, tr4);

  // Clear user regs
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG0, 0, 0, NULL));
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG1, 0, 0, NULL));
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0, 0, NULL));
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0xFFFFFFDF, 0x20, NULL)); // Reset cnt HIGH
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0xFFFFFFDF, 0x00, NULL)); // Reset cnt LOW

  //CHECKADQ(ADQ_SetTransferTimeout(adq_cu, adq_num, 1000));

  printf("\nChoose DATA RATE CONTROL MODE.\n 2 = External Trigger Mode\n 4 = Internal Trigger Mode\n");
  scanf("%u", &trig_mode);

  switch (trig_mode)
  {
  case ADQ_EXT_TRIGGER_MODE :
    {
      CHECKADQ(ADQ_SetConfigurationTrig(adq_cu, adq_num, 0x00, 0, 0));    //Disable internal trigger and trigout port
      CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));          //enable the external trigger so that it can propagate to the user logic_module
      CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0xFFFFFFF7, 0x8, NULL)); // Select ext trig as data valid
      break;
    }
  case ADQ_INTERNAL_TRIGGER_MODE:
    {
      CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, 1));      //Turn OFF external trigger mode in case it might be ON.
      printf("\nChoose Internal Trigger Frequency (Hz)\n");
      scanf("%u", &Int_Trig_Freq);
      CHECKADQ(ADQ_SetInternalTriggerFrequency(adq_cu, adq_num,(unsigned int) (Int_Trig_Freq)));
      CHECKADQ(ADQ_SetConfigurationTrig(adq_cu, adq_num, 0x45, 800, 0));             // enable 960 ns pulse length of internal trigger to trigout port, this parameter should be multible of 32 ns
      CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0xFFFFFFF7, 0x0, NULL)); // Select int trig as data valid

      //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12288+62, 0x0, 0x1)); //sample skip register
      //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12288+59, 0xFFFFFFEF, 0x10)); //interleaving relay register
      //CHECKADQ(ADQ_WriteRegister(adq_cu, adq_num, 12288+59, 0x0, 0x10)); //interleaving relay register

      break;
    }
  default:
    return;
    break;
  }

  printf("\nWrite to file?.\n 0 = No\n 1 = Yes\n");
  scanf("%d", &WriteToFile);

  printf("\nStarting streaming!");



  printf("\nSetting up streaming...");
  CHECKADQ(ADQ_SetTransferBuffers(adq_cu, adq_num, 8, 65536/2));
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num, 1));

  printf("\nDone.");

  printf("Collecting data, please wait...\n");

  // Enable streaming test data mode
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0xFFFFFFFE, 0x1, NULL));

  // Start streaming by arming
  CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));

  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0xFFFFFFeF, 0x10, NULL)); // Enable trigger

  CHECKADQ(data_stream_target = (signed char*)malloc(65536*sizeof(signed short)));    //65536 samples per page

  while ((!_kbhit()) && (!data_lost))
  {
    collect_result = ADQ_GetTransferBufferStatus(adq_cu, adq_num, &buffers_filled);
    printf("Filled: %2u\n", buffers_filled);
    printf("GetTransferBufferStatus collect_result: %2d\n", collect_result);

    if ((buffers_filled==0)&&(collect_result))
    {
      Sleep(100);
      printf("Retries: %2u\n", retries);
      retries += 1;
    }
    else
    {
      if (collect_result)
      {
        unsigned int samples_in_buffer = 0;
        unsigned int LoopVar = 0;
        retries = 0;
        collect_result = 0;
        while(!_kbhit() && !collect_result)
        {
          collect_result = ADQ_CollectDataNextPage(adq_cu, adq_num);
          printf("CollectDataNextPage collect_result: %2d\n", collect_result);
          printf("Retries: %2u\n", retries);
          retries += 1;

          if (_kbhit() && !collect_result)
          {
            printf("ABORT DETECTED! STREAMING WILL TERMINATE! Please wait!\n");
            abort = 1;
          }
        }

        samples_in_buffer = 2*ADQ_GetSamplesPerPage(adq_cu, adq_num);
        printf("Samples per page %u\n", samples_in_buffer);

        if (ADQ_GetStreamOverflow(adq_cu, adq_num))
        {
          printf("Warning: Streaming Overflow 1!\n");
          collect_result = 0;
        }

        if (collect_result)
        {
          memcpy((void*)&data_stream_target[0], ADQ_GetPtrStream(adq_cu, adq_num), samples_in_buffer*sizeof(signed char));
        }
        else
        {
          if (abort == 0)  //Unintentional abort
          {
            printf("Collect next data page failed!\n");
            break;
          }
          else    //Intentional abort
            break;

        }

        //Set initial value to start the comparision. This is a better way to check for data lost because depending on implementation the initial value might differ
        //but it does not mean that data has been lost.
        if (data_counter == 0)
        {
          data_counter_start = (((unsigned short)data_stream_target[1] << 16) | (unsigned short)data_stream_target[0]);
          data_counter = data_counter_start;
        }

        for (LoopVar = 0; LoopVar < samples_in_buffer; LoopVar++)
        {

          if (WriteToFile == 1)
          {
            if (0 == (LoopVar%16))
              fprintf(outfile, "\n");
            fprintf(outfile, "%02x", (unsigned char) data_stream_target[LoopVar]);
          }
          /*
          if (data_counter != (((unsigned short)data_stream_target[LoopVar + 1] << 16) | (unsigned short)data_stream_target[LoopVar])) // check for lost data. This must not happen!
          {
          fprintf(outfile, "DATA LOST DETECTED!!!");
          printf("DATA LOST DETECTED!!! STREAMING TERMINATE...\n");
          data_counter = (((unsigned short)data_stream_target[LoopVar + 1] << 16) | (unsigned short)data_stream_target[LoopVar]);
          data_lost = 1;
          }
          */
          data_counter++;

        }
        //fprintf(outfile, "\n");
        //Print onscreen the current samples count that has been collected
        printf("Data sets retreaved: %u\n", data_counter-data_counter_start);
        //free(data_stream_target);
        retries = 0;  //Reset the retry counter
      }
      else
      {
        printf("ADQ_GetTransferBufferStatus() failed!\n");
        break;
      }
    }

    if (_kbhit())
    {
      printf("ABORT DETECTED! STREAMING WILL TERMINATE! Please wait!\n");
    }
  }

  free(data_stream_target);
  Sleep(2000);                 //This sleeping time is required, otherwise the device cannot be started again propertly without a power reset, might be a USB only issue

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  // Disable streaming bypass of DRAM
  CHECKADQ(ADQ_SetStreamStatus(adq_cu, adq_num,0));

  // Write to data to file after streaming to RAM, because ASCII output is too slow for realtime.
  printf("Writing stream data in RAM to disk...\n");

  // restore settings
  CHECKADQ(ADQ_WriteUserRegister(adq_cu, adq_num, 0, USER_REG2, 0, 0, NULL));
  printf("\n\nDone! %u data sets stored.\n\n", data_counter-data_counter_start);

error:
  if (NULL != outfile)
  {
    fclose(outfile);
  }
  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);
  return;
}
