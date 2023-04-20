/*
 *  Copyright 2018 Teledyne Signal Processing Devices Sweden AB
 */

#ifdef __GNUC__
#ifndef LINUX
#define LINUX
#endif
#endif

#include "ADQAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void adq14(void *adq_cu, unsigned int adq_num);
extern void adq7(void *adq_cu, unsigned int adq_num);

void *adq_cu;
unsigned int adq_num;

#ifdef LINUX
#include <unistd.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <termios.h>

int _kbhit()
{
  static const int STDIN = 0;
  static int initialized = 0;

  if (!initialized)
  {
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

static void sig_handler(int signal)
{
  (void)signal;

  if (adq_cu != NULL)
  {
    ADQ_ATDStopWFA(adq_cu, adq_num);
    ADQ_ATDWaitForWFACompletion(adq_cu, adq_num);
  }
}

struct sigaction sa;

int catch_signals[] = {SIGHUP,  SIGINT,  SIGQUIT, SIGILL,  SIGABRT, SIGFPE,
                       SIGKILL, SIGSEGV, SIGPIPE, SIGTERM, SIGSTOP, -1};
#else
// For kbhit
#include <conio.h>
#endif

int main(int argc, char **argv)
{
  // ADQControlUnit
  unsigned int nof_adq = 0;
  unsigned int nof_failed_adq;

#ifdef LINUX
  unsigned int i;
#endif

  // API and device info
  char *serial_number;
  char *product_name;
  uint32_t *fw_rev;
  unsigned int product_id;
  char version_date[] = __DATE__ ", " __TIME__;

  adq_cu = NULL;
  adq_num = 1;
  (void)argc;
  (void)argv;

  printf("Version 1.4.0, Compiled: %s\n", version_date);
  printf("API revision: %08x\n", ADQAPI_GetRevision());

  /*
   * Verify the size of the ATDWFABufferStruct.
   */
  if (ADQAPI_GetObjectSize(ADQAPI_OBJECT_ATD_WFA_STRUCT) != sizeof(struct ATDWFABufferStruct))
  {
    printf("ERROR: ATDWFABufferStruct size mismatch. Recompile the example "
           "with the correct struct definition.\n");
    return -1;
  }

  // Create a control unit
  adq_cu = CreateADQControlUnit();

  // Enable Logging
  // Errors, Warnings and Info messages will be logged
  ADQControlUnit_EnableErrorTrace(adq_cu, 0x00080000, ".");

  // Find Devices
  // We will only connect to the first device in this example (adq_num = 1)
  nof_adq = ADQControlUnit_FindDevices(adq_cu);
  printf("Number of ADQ devices found:  %u\n", nof_adq);
  nof_failed_adq = ADQControlUnit_GetFailedDeviceCount(adq_cu);
  printf("Number of failed ADQ devices: %u\n", nof_failed_adq);

  if (nof_adq == 0)
  {
    printf("\nNo ADQ device found, aborting...\n");
    // Exit gracefully.
    if (adq_cu)
      DeleteADQControlUnit(adq_cu);
    return 0;
  }

#ifdef LINUX
  // Install signal handlers stopping WFA
  sa.sa_handler = sig_handler;
  sa.sa_flags = SA_NODEFER;
  for (i = 0; catch_signals[i] != -1; i++)
  {
    sigaction(catch_signals[i], &sa, NULL);
  }
#endif

  // Print product name, serial number and API revision
  fw_rev = ADQ_GetRevision(adq_cu, adq_num);
  serial_number = ADQ_GetBoardSerialNumber(adq_cu, adq_num);
  product_name = ADQ_GetBoardProductName(adq_cu, adq_num);
  product_id = ADQ_GetProductID(adq_cu, adq_num);
  printf("Firmware revision: %u\n", fw_rev[0]);
  printf("Board serial number: %s\n", serial_number);
  printf("Board product name: %s\n", product_name);

  // Check that feature FWATD is enabled on this device
  if (ADQ_HasFeature(adq_cu, adq_num, "FWATD") > 0)
  {
    printf("FWATD option correctly loaded on device.\n");
  }
  else
  {
    printf("ERROR: FWATD is not enabled for this device. "
           "License and/or correct firmware is missing.\n");
    goto exit;
  }

  if (ADQ_IsUSB3Device(adq_cu, adq_num))
  {
    unsigned int port_speed = 0;
    ADQ_GetUSB3Config(adq_cu, adq_num, 1, &port_speed);
    printf("Running over a USB%u interface.\n", port_speed);
    if (port_speed < 3)
      printf("\nWARNING: This interface significantly limits the performance "
             "of FWATD due to low transfer rate.\n\n");
  }
  else if (ADQ_IsPCIeDevice(adq_cu, adq_num))
  {
    printf("Running over a PCIe interface.\n");
  }
  else if (ADQ_IsEthernetDevice(adq_cu, adq_num))
  {
    printf("Running over an Ethernet interface.\n");
  }

  switch (product_id)
  {
  case PID_ADQ14:
    adq14(adq_cu, adq_num);
    break;
  case PID_ADQ7:
    adq7(adq_cu, adq_num);
    break;
  default:
    printf("Unsupported product ID %u.\n", product_id);
    break;
  }

  // Exit gracefully.
exit:
  if (adq_cu)
    DeleteADQControlUnit(adq_cu);

  return 0;
}
