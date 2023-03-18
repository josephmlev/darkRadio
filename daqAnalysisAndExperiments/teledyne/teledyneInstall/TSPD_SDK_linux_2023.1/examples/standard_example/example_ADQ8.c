/*
 *  Copyright 2019 Teledyne Signal Processing Devices Sweden AB
 */

#define _CRT_SECURE_NO_WARNINGS // This define removes warnings for printf

#include "ADQAPI.h"
#include "os.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef LINUX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
#endif

void adq8_multirecord(void *adq_cu, int adq_num);

#define CHECKADQ(f)              \
  if (!(f))                      \
  {                              \
    printf("Error in " #f "\n"); \
    goto error;                  \
  }

void adq8(void *adq_cu, int adq_num)
{
  /* DBS settings */
  unsigned int dbs_nof_inst = 0;
  unsigned char dbs_inst = 0;
  int dbs_bypass = 1;
  int dbs_dc_target = 0;
  int dbs_lower_saturation_level = 0;
  int dbs_upper_saturation_level = 0;

  int mode;
  char *serial_number = ADQ_GetBoardSerialNumber(adq_cu, adq_num);

  int *revision = ADQ_GetRevision(adq_cu, adq_num);

  unsigned int tlocal = ADQ_GetTemperature(adq_cu, adq_num, 0) / 256;
  unsigned int tr1 = ADQ_GetTemperature(adq_cu, adq_num, 1) / 256;
  unsigned int tr2 = ADQ_GetTemperature(adq_cu, adq_num, 2) / 256;
  unsigned int tr3 = ADQ_GetTemperature(adq_cu, adq_num, 3) / 256;

  double fs = 0.0;
  unsigned int nof_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  ADQ_GetSampleRate(adq_cu, adq_num, 0, &fs);

  printf("\nConnected to ADQ8 #1\n\n");

  printf("Device Serial Number: %s\n", serial_number);
  printf("Firmware Revision: %d\n", revision[0]);
  printf("%u channels, %.2f GSPs\n", nof_channels, fs / 1000 / 1000 / 1000);
  printf("Temperatures:\n\tLocal: %u\n\tADC0: %u\n\tADC1: %u\n\tFPGA: %u\n\n",
         tlocal, tr1, tr2, tr3);

  // Check features
  if (ADQ_HasFeature(adq_cu, adq_num, "FWDAQ") != 1)
  {
    printf("ERROR: This device is does not support the firmware required for "
           "this example (FWDAQ).\n");
    return;
  }
  else
  {
    printf("Device supports FWDAQ.\n");
  }

  /* Setup DBS */
  ADQ_GetNofDBSInstances(adq_cu, adq_num, &dbs_nof_inst);
  for (dbs_inst = 0; dbs_inst < dbs_nof_inst; ++dbs_inst)
  {
    printf("Setting up DBS instance %u...", dbs_inst);
    if (ADQ_SetupDBS(adq_cu, adq_num, dbs_inst, (unsigned int)dbs_bypass, dbs_dc_target,
                     dbs_lower_saturation_level, dbs_upper_saturation_level))
    {
      printf("success\n");
    }
    else
    {
      printf("failed.\n");
      return;
    }
  }
  Sleep(1000);

  mode = 0;
  while (1)
  {
    printf("\nChoose collect mode.\n"
           " 1 = Streaming\n"
           " 2 = Multirecord\n"
           " 0 = Exit\n");
    scanf("%d", &mode);
    switch (mode)
    {
    case 0:
      return;
    case 1:
      printf("This example is deprecated for the streaming acquisition mode. Refer to the example "
             "'gen3_streaming' for an updated reference implementation and to the user guide "
             "(20-2465) for its documentation.\n");
      break;
    case 2:
      adq8_multirecord(adq_cu, adq_num);
      break;
    default:
      printf("Unsupported mode '%d'.\n", mode);
      break;
    }
  }
}

void adq8_multirecord(void *adq_cu, int adq_num)
{
  int trig_mode;
  int trig_level;
  int trig_edge;
  unsigned int samples_per_record;
  unsigned int number_of_records;
  unsigned int buffer_size;
  unsigned int channel;
  unsigned char channel_mask;
  unsigned int nof_channels = 0;

  unsigned int write_to_file = 1;
  unsigned int records_to_collect;

  short *buf[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  // GetData allows for a digitizer with max 8 channels, the unused pointers
  // should be NULL.
  void *target_buffers[8];
  char file_name[256];
  FILE *outfile = NULL;
  int exit = 0;
  int i;

  nof_channels = ADQ_GetNofChannels(adq_cu, adq_num);

  printf("\nChoose trig mode.\n"
         " %d = SW trigger mode\n"
         " %d = External trigger mode\n"
         " %d = Level trigger mode\n",
         ADQ_SW_TRIGGER_MODE, ADQ_EXT_TRIGGER_MODE, ADQ_LEVEL_TRIGGER_MODE);
  scanf("%d", &trig_mode);

  switch (trig_mode)
  {
  case ADQ_SW_TRIGGER_MODE:
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
  case ADQ_EXT_TRIGGER_MODE:
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    break;
  case ADQ_LEVEL_TRIGGER_MODE:
    CHECKADQ(ADQ_SetTriggerMode(adq_cu, adq_num, trig_mode));
    printf("\nChoose trig level.\n"
           " -32768 <= level <= 32767\n");
    scanf("%d", &trig_level);
    CHECKADQ(ADQ_SetLvlTrigLevel(adq_cu, adq_num, trig_level));
    printf("\nChoose trig edge.\n"
           " 1 = Rising edge\n"
           " 0 = Falling edge\n");
    scanf("%d", &trig_edge);
    CHECKADQ(ADQ_SetLvlTrigEdge(adq_cu, adq_num, trig_edge));
    printf("\nChoose level trigger channel [1,%u].\n", nof_channels);
    scanf("%u", &channel);
    CHECKADQ(ADQ_SetLvlTrigChannel(adq_cu, adq_num, (int)channel));
    break;
  default:
    printf("ERROR: Unsupported trigger mode");
    goto error;
  }

  printf("\nChoose number of records.\n");
  scanf("%u", &number_of_records);

  printf("\nChoose number of samples per record.\n");
  scanf("%u", &samples_per_record);

  channel_mask = 0xff;

  // Use only multirecord mode for data collection.
  CHECKADQ(ADQ_MultiRecordSetup(adq_cu, adq_num, number_of_records,
                                samples_per_record));

  if (trig_mode == ADQ_SW_TRIGGER_MODE)
  {
    unsigned int has_acquired_all;
    printf("Issuing software trigger(s).\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    do
    {
      has_acquired_all = ADQ_GetAcquiredAll(adq_cu, adq_num);
      CHECKADQ(ADQ_SWTrig(adq_cu, adq_num));
    } while (has_acquired_all == 0);
  }
  else
  {
    unsigned int has_acquired_all;
    printf("\nPlease trigger your device to collect data.\n");
    CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));
    CHECKADQ(ADQ_ArmTrigger(adq_cu, adq_num));
    do
    {
      has_acquired_all = ADQ_GetAcquiredAll(adq_cu, adq_num);
    } while (has_acquired_all == 0);
  }
  printf("\nAll records triggered\n");

  printf("\nSave collected data to file?\n"
         " 0 = Do not save\n"
         " 1 = Save in ascii format (slow)\n"
         " 2 = Save in binary format (fast)\n");
  scanf("%u", &write_to_file);

  records_to_collect = number_of_records;

  printf("\nRetreiving data, please wait...\n");

  // Each data buffer must contain enough samples to store all the records
  // consecutively.
  buffer_size = records_to_collect * samples_per_record;
  for (channel = 0; channel < nof_channels; channel++)
  {
    buf[channel] = (short *)calloc(buffer_size, sizeof(short));
    target_buffers[channel] = buf[channel];
    if (buf[channel] == NULL)
      goto error;
  }

  // Use the GetData function
  CHECKADQ(ADQ_GetData(adq_cu, adq_num, target_buffers, buffer_size,
                       sizeof(short), 0, records_to_collect, channel_mask, 0,
                       samples_per_record, ADQ_TRANSFER_MODE_NORMAL));

  printf("All records collected.\n");
  switch (write_to_file)
  {
  case 0:
    // Do not save
    break;
  case 1:
    // Save as ascii
    for (channel = 0; channel < nof_channels; channel++)
    {
      sprintf(file_name, "data_ch%u.txt", channel);
      outfile = fopen(file_name, "w");

      if (outfile != NULL)
      {
        unsigned int j;
        for (j = 0; j < buffer_size; j++)
        {
          if (channel_mask & (0x01 << channel))
            fprintf(outfile, "%d\n", ((short *)target_buffers[channel])[j]);
        }
        fclose(outfile);
      }
      else
      {
        printf("Error: Failed to open output files.\n");
      }
    }
    break;
  case 2:
    // Save as binary
    for (channel = 0; channel < nof_channels; channel++)
    {
      sprintf(file_name, "data_ch%u.bin", channel);
      outfile = fopen(file_name, "wb");
      if (outfile != NULL)
      {
        if (channel_mask & (0x01 << channel))
          fwrite((short *)target_buffers[channel], sizeof(short), buffer_size,
                 outfile);
        fclose(outfile);
      }
      else
      {
        printf("Error: Failed to open output files.\n");
      }
    }
    break;
  default:
    printf("Error: Unknown format!\n");
    break;
  }

  // Only disarm trigger after data is collected
  CHECKADQ(ADQ_DisarmTrigger(adq_cu, adq_num));

  if (write_to_file > 0)
    printf("\n\nDone.\n");

error:

  for (i = 0; i < 8; i++)
    if (buf[i] != NULL)
      free(buf[i]);

  printf("Press 0 followed by ENTER to exit.\n");
  scanf("%d", &exit);

  return;
}
