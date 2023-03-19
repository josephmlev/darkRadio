/*
* SphinxAPI_example.cpp
*
* This example shows several features of the Sphinx system.
* By changing the defines and parameters under "Configuration" below
* several different functions and configurations can be tested.
*
*/

#define _CRT_SECURE_NO_DEPRECATE //Remove irrelevant warnings
#include "ADQAPI.h"
#include <assert.h>
#include <stdio.h>

#ifndef LINUX
#include <windows.h>
#else
#include <stdlib.h>
#include <string.h>
#define __declspec __attribute__
#define align(X) (aligned(X))
#endif

// Special define
#define CHECKADQ(f) if(!(f)){printf("Error in " #f "\n"); goto error;}
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Defines
#define POSTPROCESS 0 //Use post-processing and accumulate result in single batch of data? Else store all data
#define USE_PROCESSING_CARD 0 //0=Send data to CPU. 1=Send data to Processing card
#define USE_RAMP_COUNTER_FROM_PROC 0 //1=send ram data from proc (not using LS card), 0=Get data from LS Card.
// Parameters
int diagnosticMode = 1; // 0=algorithm, 1=diagnostic
int outputType = 1; // 0=Phi, 1=DeltaPhi
int sendLength = 2560;
unsigned int NofBatches = 1;
int sendStart = 1;
int decimationFactor = 1; // 1=no decimation, 2=half rate, 4=quarter rate, etc
int phiGain = 0; // 0-3
float ADCgain = 5.0f; // Set in dB
int deltaPhiMax = 20000; // -32768 <-> 32767
int NormalizationOn = 0; // 0=normal, 1=normalization enabled
unsigned int normalization_factor0 = 65535;
unsigned int normalization_factor1 = 65535;
unsigned int normalization_factor2 = 65535;
unsigned int A0 = 65535;
unsigned int A1 = 65535;
unsigned int A2 = 65535;
unsigned int A3 = 65535;
unsigned int A4 = 65535;
unsigned int diff_amp_factor = 0; // 2^0
int clockSource = 0; // 0=internal, 1=external
int computerMemory = 512; // Amount of memory (MB) in PC, used for sanity check during buffer allocation
unsigned int PulsePeriod = 20000;
unsigned int PulseLength = 60; //
unsigned int PulseDelay = 25;
unsigned int cicDecimationRate = 0;
unsigned int NofBatchesInData;
int success;
short* api_data_ptr;
unsigned int currentBufferSize;
unsigned int currentBufferSizeMult;

void sphinxAA_example(void *adq_cu, int devNum);

void sphinxaa(void *adq_cu, int devNum)
{
  int* revision = SphinxAA_GetRevision(adq_cu, devNum);
  printf("\nConnected to SphinxAA #1\n\n");

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

  printf("\nChoose collect mode.\n 0 = Algorithm\n 1 = Diagnostic\n\n\n");

  scanf("%d", &diagnosticMode);

  if ((diagnosticMode >= 0) && (diagnosticMode <= 1))
  {
    sphinxAA_example(adq_cu, devNum);
  }
}

// Store data into Matlab binary files.
void sphinxAA_write(short* data)
{
  FILE* outfile[4] = {NULL, NULL, NULL, NULL};
  int temp[4];
  // Matlab file header
  __declspec(align(2))                static struct fileheader_t
  {
    char text[116];
    int subsystem[2];
    char info[4];
    int matrix[2];
    int matrixflags[4];
    int matrixdim[4];
    int matrixname[4];
  } fileheader;

  printf("Write user buffer to file 'data#.mat'...");

  { // Init file header
    const char *text = "SphinxAPI_example";
    memset((void*) &fileheader, 0, sizeof(fileheader));
    memcpy((void*) &fileheader.text[0], (void*) text, strlen(text));
    fileheader.subsystem[0] = 0;
    fileheader.subsystem[1] = 0;
    fileheader.info[1] = 0x01;
    fileheader.info[2] = 'I';
    fileheader.info[3] = 'M';
    fileheader.matrix[0] = 14;
    fileheader.matrix[1] = sizeof(short) * currentBufferSize * NofBatches + sizeof(int) * (12 + 2);
    fileheader.matrixflags[0] = 6;
    fileheader.matrixflags[1] = 8;
    fileheader.matrixflags[2] = 10;
    fileheader.matrixdim[0] = 5;
    fileheader.matrixdim[1] = 8;
    fileheader.matrixdim[2] = currentBufferSize;
    fileheader.matrixdim[3] = NofBatches;
    fileheader.matrixname[0] = 1;
    fileheader.matrixname[1] = 5;
    fileheader.matrixname[2] = ('a' << 24) | ('t' << 16) | ('a' << 8) | 'd';
    fileheader.matrixname[3] = '1';
  }

  // Set up file handles and temporaries
  outfile[0] = fopen("data1.mat", "wbS");

  if (!diagnosticMode) //If normal mode, write all data to single file
  {

    fwrite((void*) &fileheader, sizeof(fileheader), 1, outfile[0]);

    temp[0] = 3;
    temp[1] = NofBatches * currentBufferSize * sizeof(short);
    fwrite((void*) &temp, sizeof(int), 2, outfile[0]);

    fwrite((void*) &data[0], sizeof(short) * currentBufferSize * NofBatches, 1, outfile[0]);
  }
  else // Diagnostics mode, use separate files for all channels
  {
    unsigned int i;
    unsigned int j;
    outfile[1] = fopen("data2.mat", "wbS");
    outfile[2] = fopen("data3.mat", "wbS");
    outfile[3] = fopen("data4.mat", "wbS");

    for (i = 0; i < 4; i++) // Generate new file headers
    {
      fileheader.matrixname[3] = '1' + (char) i;
      fwrite((void*) &fileheader, sizeof(fileheader), 1, outfile[i]);
      temp[0] = 3;
      temp[1] = NofBatches * currentBufferSize * sizeof(short);
      fwrite((void*) &temp, sizeof(int), 2, outfile[i]);
    }

    for (j = 0; j < NofBatches; j++) // De-interleave data and write to files
    {
      short *dataptr = &data[currentBufferSize * j * 4];
      for (i = 0; i < currentBufferSize * 4; i += 4)
      {
        fwrite((void*) &dataptr[i], sizeof(short), 1, outfile[0]);
        fwrite((void*) &dataptr[i + 1], sizeof(short), 1, outfile[1]);
        fwrite((void*) &dataptr[i + 2], sizeof(short), 1, outfile[2]);
        fwrite((void*) &dataptr[i + 3], sizeof(short), 1, outfile[3]);
      }
    }
  if (outfile[1] != NULL) fclose(outfile[1]);
  if (outfile[2] != NULL) fclose(outfile[2]);
  if (outfile[3] != NULL) fclose(outfile[3]);
  }

  if (outfile[0] != NULL) fclose(outfile[0]);

  printf("done!\n");
}

void sphinxAA_example(void *adq_cu, int devNum)
{
  short* data; //Local buffer
  unsigned int j;
  unsigned long memsize;

  // Set ADC gain for AA card
  SphinxAA_SetAdcGain(adq_cu, devNum, (unsigned int) ((ADCgain + 11.5) * 2.0));
  SphinxAA_DisarmTrigger(adq_cu, devNum);
  SphinxAA_SetTriggerMode(adq_cu, devNum, 1); // 1= EXT trigger, 3 = SW trigger

  // Pulse Generator
  SphinxAA_ConfigurePulseGenerator(adq_cu, devNum, 1, PulsePeriod, PulseLength, 3.3f, PulseLength, 3.3f, PulseDelay);

  SphinxAA_SetPostDecimationFactor(adq_cu, devNum, 1);
  SphinxAA_SetDecimationFactor(adq_cu, devNum, decimationFactor);
  SphinxAA_SetPreTrigSamples(adq_cu, devNum, 90);
  SphinxAA_SetOffsetLength(adq_cu, devNum, 64);
  SphinxAA_SetClockSource(adq_cu, devNum, clockSource);
  SphinxAA_SetPllFreqDivider(adq_cu, devNum, 2);

  if (NormalizationOn == 1)
  {
    SphinxAA_NormEnable(adq_cu, devNum);
  }
  else
  {
    SphinxAA_NormDisable(adq_cu, devNum);
  }

  SphinxAA_SetNormalizationFactor(adq_cu, devNum, 0, normalization_factor0);
  SphinxAA_SetNormalizationFactor(adq_cu, devNum, 1, normalization_factor1);
  SphinxAA_SetNormalizationFactor(adq_cu, devNum, 2, normalization_factor2);
  SphinxAA_SetDPhiFilterFactors(adq_cu, devNum, 0, A0);
  SphinxAA_SetDPhiFilterFactors(adq_cu, devNum, 1, A1);
  SphinxAA_SetDPhiFilterFactors(adq_cu, devNum, 2, A2);
  SphinxAA_SetDPhiFilterFactors(adq_cu, devNum, 3, A3);
  SphinxAA_SetDPhiFilterFactors(adq_cu, devNum, 4, A4);
  SphinxAA_SetDiffAmplifyFactor(adq_cu, devNum, diff_amp_factor);
  //SphinxAA_SetNormGain(adq_cu, devNum,1<<20);
  //SphinxAA_SetPhiGain(adq_cu, devNum,phiGain);
  SphinxAA_EnableIntegration(adq_cu, devNum);
  SphinxAA_SetIntegrationCoeff(adq_cu, devNum, 130810);

  SphinxAA_SetOffsetRange(adq_cu, devNum, 0, -32768, 32767);
  SphinxAA_SetOffsetRange(adq_cu, devNum, 1, -32768, 32767);
  SphinxAA_SetOffsetRange(adq_cu, devNum, 2, -32768, 32767);
  SphinxAA_SetDeltaphiRange(adq_cu, devNum, -32768, 32767);

  SphinxAA_SetMode(adq_cu, devNum, diagnosticMode);
  SphinxAA_SetSendLength(adq_cu, devNum, sendLength);
  SphinxAA_SetSendStart(adq_cu, devNum, sendStart);

  SphinxAA_SetOutputControlTestPattern(adq_cu, devNum, 0); // Enable Debug test ramp

  SphinxAA_SetNofBatches(adq_cu, devNum, NofBatches);
  SphinxAA_SetOutputType(adq_cu, devNum, outputType);

  // Set up local buffer
  currentBufferSize = SphinxAA_GetSendLength(adq_cu, devNum);
  currentBufferSizeMult = 1;

  if (diagnosticMode)
  {
    currentBufferSizeMult = 4;
  }

  memsize = computerMemory * 1024 * 1024;

  if ((double) NofBatches * currentBufferSize * sizeof(short) * currentBufferSizeMult > (double) memsize * 0.8)
  {
    printf("User data buffer larger than %dkB (%dMB): aborting.\n", (int) ((double) memsize * 0.8 / (1024 * 1024)),
      (int) ((double) NofBatches * currentBufferSize * sizeof(short) * currentBufferSizeMult / (1024 * 1024)));
    goto exit;
  }

  printf("Allocating user buffer (%dkB)...", (int) ((double) NofBatches * currentBufferSize * sizeof(short) * currentBufferSizeMult / (1024)));
  data = (short*) malloc(NofBatches * currentBufferSize * sizeof(short) * currentBufferSizeMult);

  if (!data)
  {
    printf("user data buffer allocation failed!\n");
    goto exit;
  }

  printf("done!\n");

  SphinxAA_ArmTrigger(adq_cu, devNum);

  for (j = 0; j < NofBatches; j += NofBatchesInData)
  {
    int overflow;
    success = SphinxAA_GetData(adq_cu, devNum); // Get data
    NofBatchesInData = SphinxAA_GetNofBatchesPerGetData(adq_cu, devNum); // Get amount of data in DMA buffer
    success = success && NofBatchesInData; // Error check

    if (!success) // Abort if Error
    {
      if (SphinxAA_GetFifoOverflow(adq_cu, devNum))
        printf("Collect data page %u failed! Overflow detected.\n", j);
      else if (!NofBatchesInData)
        printf("Collect data page %u failed! No data returned.\n", j);
      else
        printf("Collect data page %u failed!\n", j);
      goto exit;
    }

    api_data_ptr = SphinxAA_GetPtrData(adq_cu, devNum); // Get DMA buffer address, new after each GetData()

    // Copy data from DMA buffer to local buffer
    if (NULL != api_data_ptr)
    {
      memcpy((void*) &data[currentBufferSize * j * currentBufferSizeMult], (void*) api_data_ptr,
        NofBatchesInData * currentBufferSize * sizeof(short) * currentBufferSizeMult);
    }

    overflow = SphinxAA_GetOOR(adq_cu, devNum);

    if (overflow != 0)
    {
      printf("Overflow in batch, overflow code is: 0x%x.\n", overflow);
    }

  }

  sphinxAA_write(data);

exit: ;
}
