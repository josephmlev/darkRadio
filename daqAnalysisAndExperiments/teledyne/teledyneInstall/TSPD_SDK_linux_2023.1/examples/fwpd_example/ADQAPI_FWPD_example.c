/*
 * Copyright 2018 Teledyne Signal Processing Devices Sweden AB
 *
 * NOTE: This example is only intended for ADQ14 and ADQ7 devices with the
 * firmware option FWPD (pulse detection) installed.
 */

#ifdef __GNUC__
#ifndef LINUX
#define LINUX
#endif
#else
/* Remove unsafe function warning */
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ADQAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "formatter.h"

extern void adq14(void *adq_cu, unsigned int adq_num);
extern void adq7(void *adq_cu, unsigned int adq_num);

int main(int argc, char **argv)
{
  /* ADQControlUnit */
  void *adq_cu = NULL;
  unsigned int nof_adq = 0;
  unsigned int adq_num = 1;
  unsigned int nof_failed_adq;

  /* API and device info */
  char *serial_number;
  char *product_name;
  int api_rev = 0;
  int *fw_rev;
  unsigned int product_id;

  /* Date and time */
  time_t t;
  struct tm *ts;

  char version_date[] = __DATE__ ", " __TIME__;

  t = time(NULL);
  ts = localtime(&t);

  printf("FWPD example started %2.2d:%2.2d:%2.2d.\n\n", ts->tm_hour, ts->tm_min, ts->tm_sec);
  printf("Version 1.0.0, Compiled: %s\n", version_date);

  /* Create an ADQ control unit. */
  adq_cu = CreateADQControlUnit();

  /* Enable trace logging. Errors, warnings and info messages will be logged. */
  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");

  /* Find devices */
  printfw("Identifying connected devices...");
  nof_adq = ADQControlUnit_FindDevices(adq_cu);
  nof_failed_adq = ADQControlUnit_GetFailedDeviceCount(adq_cu);

  if ((nof_adq == 0) && (nof_failed_adq == 0))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  printf("Number of ADQ devices found:  %u\n", nof_adq);
  printf("Number of failed ADQ devices: %u\n", nof_failed_adq);

  if (nof_adq > 1)
  {
    printf("Found %u devices. This example only supports 1 device, "
           "aborting...\n",
           adq_num);
    goto cleanup_exit;
  }

  if ((nof_failed_adq > 0) && (nof_adq == 0))
  {
    printf("The connected device(s) failed to start, aborting...\n");
    goto cleanup_exit;
  }

  if (nof_adq == 0)
  {
    printf("\nNo ADQ device found, aborting...\n");
    /* Exit gracefully */
    goto cleanup_exit;
  }

  /* Print product name, serial number and API revision */
  api_rev = ADQAPI_GetRevision();
  fw_rev = ADQ_GetRevision(adq_cu, adq_num);
  serial_number = ADQ_GetBoardSerialNumber(adq_cu, adq_num);
  product_name = ADQ_GetBoardProductName(adq_cu, adq_num);
  product_id = ADQ_GetProductID(adq_cu, adq_num);
  printf("\nAPI revision:        %d\n", api_rev);
  printf("Firmware revision:   %d\n", fw_rev[0]);
  printf("Board serial number: %s\n", serial_number);
  printf("Board product name:  %s\n", product_name);

  /* Check that feature FWPD is enabled on this device. */
  printfw("Checking for FWPD feature ...");
  if (!(ADQ_HasFeature(adq_cu, adq_num, "FWPD") > 0))
  {
    printfwrastatus("FAILED", 2);
    goto cleanup_exit;
  }
  printfwrastatus("OK", 1);

  if (ADQ_IsUSB3Device(adq_cu, adq_num))
  {
    unsigned int port_speed = 0;
    ADQ_GetUSB3Config(adq_cu, adq_num, 1, &port_speed);
    printf("Running over a USB%u interface.\n", port_speed);
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
  }

  /* Exit gracefully */
cleanup_exit:
  if (adq_cu)
    DeleteADQControlUnit(adq_cu);

  return 0;
}
