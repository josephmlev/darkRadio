// File: example_ADQAPI.cpp
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
#include <sys/ioctl.h>
#include <termios.h>

  int _kbhit()
  {
      static const int STDIN = 0;
      static int initialized = 0;

      if (!initialized) {
          // Use termios to turn off line buffering
          struct termios term;
          tcgetattr(STDIN, &term);
          term.c_lflag &= ~ICANON;
          tcsetattr(STDIN, TCSANOW, &term);
          setbuf(stdin, NULL);
          initialized = 1;
      }

      int bytesWaiting;
      ioctl(STDIN, FIONREAD, &bytesWaiting);
      return bytesWaiting;
  }

#endif

extern void adq108(void *adq_cu, int adq_num);
extern void adq112(void *adq_cu, int adq_num);
extern void adq114(void *adq_cu, int adq_num);
extern void adq214(void *adq_cu, int adq_num);
extern void adq212(void *adq_cu, int adq_num);
extern void adq412(void *adq_cu, int adq_num);
extern void sdr14(void *adq_cu, int adq_num);
extern void adq1600(void *adq_cu, int adq_num);
extern void sphinxaa(void *adq_cu, int adq_num);
extern void adq208(void *adq_cu, int adq_num);
extern void adq14(void *adq_cu, int adq_num);
extern void adq7(void *adq_cu, int adq_num);
extern void adq8(void *adq_cu, int adq_num);

int main(int argc, char* argv[])
{
  unsigned int n_of_devices = 0;
  int n_of_failed = 0;
  unsigned int adq_num = 0;
  unsigned int tmp_adq_num = 0;
  int n_of_opened_devices = 0;
  unsigned int pID = 0;
  int n_of_ADQ = 0;
  int apirev = 0;
  int exit = 0;
  char* product_name;
  void* adq_cu;
  struct ADQInfoListEntry* ADQlist;
  unsigned int err;

//START:

  apirev = ADQAPI_GetRevision();

  printf("ADQAPI Example\n");
  printf("API Revision: %6d\n", apirev);

  adq_cu = CreateADQControlUnit(); //creates an ADQControlUnit
  if(!adq_cu)
  {
    printf("Failed to create adq_cu!\n");
    return 0;
  }

  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");


  if(!ADQControlUnit_ListDevices(adq_cu, &ADQlist, &n_of_devices))
  {
    printf("ListDevices failed!\n");
    err = ADQControlUnit_GetLastFailedDeviceError(adq_cu);
    printf(" Last error reported is: %08X.\n", err);
    if (err == 0x00000001)
    {
      printf("ERROR: The linked ADQAPI is not for the correct OS, please select correct x86/x64 platform when building.\n");
    }
    return 0;
  }

  printf("Select info array entry to open.\n\n");

  adq_num = 0xFFFFFFFF;

  if(n_of_devices == 0)
  {
    printf("No devices found!\n");
    DeleteADQControlUnit(adq_cu);
    // Check if there is a stored device structure file
    if (ADQData_GetDeviceStructPID(NULL, "adq_device_struct.bin", &pID)) {
      switch (pID) {
      case PID_ADQ7:
        printf("Found an ADQ7 device structure file \"adq_device_struct.bin\".\n");
        adq7(NULL, 0);
        break;
      default:
        break;
      }
    }
    return 0;
  }

  while(adq_num >= n_of_devices)
  {
    for(tmp_adq_num = 0; tmp_adq_num < n_of_devices; tmp_adq_num++)
    {
      printf("Entry #%u - ",tmp_adq_num);
      switch (ADQlist[tmp_adq_num].ProductID)
      {
      case PID_ADQ214: printf("ADQ214"); break;
      case PID_ADQ114: printf("ADQ114"); break;
      case PID_ADQ212: printf("ADQ212"); break;
      case PID_ADQ112: printf("ADQ112"); break;
      case PID_ADQDSP: printf("ADQDSP"); break;
      case PID_ADQ108: printf("ADQ108"); break;
      case PID_ADQ412: printf("ADQ412"); break;
      case PID_ADQ1600: printf("ADQ1600"); break;
      case PID_SDR14: printf("SDR14"); break;
      case PID_ADQ208: printf("ADQ208"); break;
      case PID_DSU: printf("DSU"); break;
      case PID_SphinxAA14: printf("SphinxAA14"); break;
      case PID_ADQ14: printf("ADQ14"); break;
      case PID_ADQ12: printf("ADQ12"); break;
      case PID_ADQ7: printf("ADQ7"); break;
      case PID_ADQ8: printf("ADQ8"); break;
      }
      printf("    [PID %04X; Addr1 %04X; Addr2 %04X; HWIF %i; Setup %i]\n",
        ADQlist[tmp_adq_num].ProductID, ADQlist[tmp_adq_num].AddressField1,  ADQlist[tmp_adq_num].AddressField2, ADQlist[tmp_adq_num].HWIFType, ADQlist[tmp_adq_num].DeviceSetupCompleted);
    }

    if (n_of_devices > 1)
    {
      printf("\nEntry to open: ");
      scanf("%u", &adq_num);
    }
    else
    {
      adq_num = 0;
      printf("\nOnly one entry found. Opening entry: %u\n", adq_num);
    }
  }

  printf("Opening device...    ");

  if(ADQControlUnit_OpenDeviceInterface(adq_cu, adq_num))
    printf("success!\n");
  else
  {
    printf("failed!\n");
    goto error;
  }

  printf("Setting up device... ");

  if(ADQControlUnit_SetupDevice(adq_cu, adq_num))
    printf("success!\n");
  else
  {
    printf("failed!\n");
    goto error;
  }

  n_of_ADQ = ADQControlUnit_NofADQ(adq_cu);

  printf("Total opened units: %i\n\n", n_of_ADQ);

  n_of_failed = ADQControlUnit_GetFailedDeviceCount(adq_cu);

  if (n_of_failed > 0)
  {
    printf("Found but failed to start %d ADQ devices.\n", n_of_failed);
    goto error;
  }

  if (n_of_devices == 0)
  {
    printf("No ADQ devices found.\n");
    goto error;
  }

  n_of_opened_devices = ADQControlUnit_NofADQ(adq_cu);
  printf("\n\nNumber of opened ADQ devices found: %d \n", n_of_opened_devices);

  for (adq_num = 1; adq_num <= (unsigned int) n_of_opened_devices; adq_num++)
  {
    product_name = ADQ_GetBoardProductName(adq_cu, adq_num);
    printf("%2u: ", adq_num);
    printf(product_name, "\n");
  }

  adq_num = 0;
  while(adq_num > (unsigned int) n_of_opened_devices || adq_num < 1)
  {
    if (n_of_opened_devices > 1)
    {
      printf("\nSelect Device to operate: ");
      scanf("%u", &adq_num);
    }
    else
    {
      adq_num = 1;
      printf("\n\nOnly one device detected. Selected device to operate: %u\n", adq_num);
    }

    if(adq_num > (unsigned int) n_of_opened_devices || adq_num < 1)
      printf("\nIncorrect number, try again!\n");
  }

  pID = ADQ_GetProductID(adq_cu, adq_num);
  switch (pID)
  {
  case PID_ADQ214: adq214(adq_cu, adq_num); break;
  case PID_ADQ114: adq114(adq_cu, adq_num); break;
  case PID_ADQ212: adq212(adq_cu, adq_num); break;
  case PID_ADQ112: adq112(adq_cu, adq_num); break;
  case PID_ADQ108: adq108(adq_cu, adq_num); break;
  case PID_ADQ412: adq412(adq_cu, adq_num); break;
  case PID_ADQ1600: adq1600(adq_cu, adq_num); break;
  case PID_SDR14: sdr14(adq_cu, adq_num); break;
  case PID_ADQ208: adq208(adq_cu, adq_num); break;
  case PID_SphinxAA14: sphinxaa(adq_cu, adq_num); break;
  case PID_ADQ14: adq14(adq_cu, adq_num); break;
  case PID_ADQ12: printf("\nNOTE: ADQ12 is run as an ADQ14 in all examples.\n"); adq14(adq_cu, adq_num); break;
  case PID_ADQ7: adq7(adq_cu, adq_num); break;
  case PID_ADQ8: adq8(adq_cu, adq_num); break;
  default: printf("This example does not contain code for the selected device.\n\n"); break;
  }

  DeleteADQControlUnit(adq_cu);
  //goto START;

  return 0;

error:
  printf("Type 0 and ENTER to exit.\n");
  scanf("%d", &exit);
  DeleteADQControlUnit(adq_cu);
  return 0;
}
