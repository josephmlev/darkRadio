/*
 *
 * Copyright Signal Processing Devices Sweden AB. All rights reserved.
 * See document "08-0175 EULA" for specific license terms regarding this file.
 *
 * Description   : Implementation of debugging functions for ADQ14-Tarsier
 * Documentation :
 *
 */

#include "helpers.h"
#include "debugging.h"
#include "ADQAPI.h"

int PrintEventCounters(void* adq_cu, int adq_num) {
  int success = 1;
  unsigned int lt_tevent_counter     = 0xdeafbeed;
  unsigned int lt_revent_counter     = 0xdeafbeed;
  unsigned int coin_tevent_counter   = 0xdeafbeed;
  unsigned int coin_revent_counter   = 0xdeafbeed;
  unsigned int pt_tevent_counter     = 0xdeafbeed;
  unsigned int pt_revent_counter     = 0xdeafbeed;
  unsigned int acq_tevent_counter    = 0xdeafbeed;
  unsigned int acq_revent_counter    = 0xdeafbeed;
  unsigned int acq_revent_pt_counter = 0xdeafbeed;
  // unsigned int ltt_tevent_counter    = 0xdeafbeed;
  // unsigned int ltt_revent_counter    = 0xdeafbeed;
  // unsigned int regval;

  success = ADQ_PDGetEventCounters(adq_cu, adq_num,
                                   &lt_tevent_counter,
                                   &lt_revent_counter,
                                   &coin_tevent_counter,
                                   &coin_revent_counter,
                                   &pt_tevent_counter,
                                   &pt_revent_counter,
                                   &acq_tevent_counter,
                                   &acq_revent_counter,
                                   &acq_revent_pt_counter);

  // regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50230000+4*11);
  // ltt_tevent_counter = (regval & 0x0000FFFF);
  // ltt_revent_counter = ((regval >> 16) & 0x0000FFFF);

  printf("LT  tevent:    %u\n", lt_tevent_counter);
  printf("LT  revent:    %u\n", lt_revent_counter);
  printf("UL1 tevent:    %u\n", coin_tevent_counter);
  printf("UL1 revent:    %u\n", coin_revent_counter);
  printf("PT  tevent:    %u\n", pt_tevent_counter);
  printf("PT  revent:    %u\n", pt_revent_counter);
  printf("ACQ tevent:    %u\n", acq_tevent_counter);
  printf("ACQ revent:    %u\n", acq_revent_counter);
  printf("ACQ revent_pt: %u\n", acq_revent_pt_counter);
  // printf("LTT tevent:    %u\n", ltt_tevent_counter);
  // printf("LTT revent:    %u\n", ltt_revent_counter);

  return success;
}

int PrintRecordCounters(void* adq_cu, int adq_num) {
  int success = 1;

  unsigned int regval;
  printf("******* RECORD STATUS COUNTERS *******\n");
  // success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
  //                                           (UL2_CTR_VALID_IN_CHA_OFFSET),
  //                                           &regval);

  // printf("UL2 DVIN CHA:              %u\n", regval);

  // success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
  //                                           (UL2_CTR_VALID_OUT_CHA_OFFSET),
  //                                           &regval);

  // printf("UL2 DVOUT CHA:             %u\n", regval);

  // success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
  //                                           (UL2_CTR_RECBITS_IN_CHA_OFFSET),
  //                                           &regval);
  // printf("UL2 RECSTARTIN  CHA:       %u\n", regval & 0x0000FFFF);
  // printf("UL2 RECSTOPIN   CHA:       %u\n", (regval >> 16) & 0x0000FFFF);

  // success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
  //                                           (UL2_CTR_RECBITS_OUT_CHA_OFFSET),
  //                                           &regval);
  // printf("UL2 RECSTARTOUT CHA:       %u\n", regval & 0x0000FFFF);
  // printf("UL2 RECSTOPOUT  CHA:       %u\n", (regval >> 16) & 0x0000FFFF);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR +
                                             UL2_CHAR_CTR_VALID_TO_MDGEN_OFFSET),
                                            &regval);
  printf("CHAR0 DV TO MDGEN:         %u\n", regval);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR +
                                             UL2_CHAR_CTR_VALID_FROM_MDGEN_OFFSET),
                                            &regval);
  printf("CHAR0 DV FROM MDGEN:       %u\n", regval);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR +
                                             UL2_CHAR_CTR_VALID_FROM_OGEN_OFFSET),
                                            &regval);
  printf("CHAR0 DV FROM OGEN:        %u\n", regval);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR +
                                             UL2_CHAR_CTR_RECBITS_TO_MDGEN_OFFSET),
                                            &regval);
  printf("CHAR0 RECSTART TO MDGEN:   %u\n", regval & 0x0000FFFF);
  printf("CHAR0 RECSTOP TO MDGEN:    %u\n", (regval >> 16) & 0x0000FFFF);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR +
                                             UL2_CHAR_CTR_RECBITS_FROM_MDGEN_OFFSET),
                                            &regval);

  printf("CHAR0 RECSTART FROM MDGEN: %u\n", regval & 0x0000FFFF);
  printf("CHAR0 RECSTOP FROM MDGEN:  %u\n", (regval >> 16) & 0x0000FFFF);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR +
                                             UL2_CHAR_CTR_RECBITS_FROM_OGEN_OFFSET),
                                            &regval);

  printf("CHAR0 RECSTART FROM OGEN:  %u\n", regval & 0x0000FFFF);
  printf("CHAR0 RECSTOP FROM OGEN:   %u\n", (regval >> 16) & 0x0000FFFF);




  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR + UL2_CHAR_OFFSET +
                                             UL2_CHAR_CTR_VALID_TO_MDGEN_OFFSET),
                                            &regval);
  printf("CHAR1 DV TO MDGEN:         %u\n", regval);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR + UL2_CHAR_OFFSET +
                                             UL2_CHAR_CTR_VALID_FROM_MDGEN_OFFSET),
                                            &regval);
  printf("CHAR1 DV FROM MDGEN:       %u\n", regval);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR + UL2_CHAR_OFFSET +
                                             UL2_CHAR_CTR_VALID_FROM_OGEN_OFFSET),
                                            &regval);
  printf("CHAR1 DV FROM OGEN:        %u\n", regval);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR + UL2_CHAR_OFFSET +
                                             UL2_CHAR_CTR_RECBITS_TO_MDGEN_OFFSET),
                                            &regval);
  printf("CHAR1 RECSTART TO MDGEN:   %u\n", regval & 0x0000FFFF);
  printf("CHAR1 RECSTOP TO MDGEN:    %u\n", (regval >> 16) & 0x0000FFFF);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR + UL2_CHAR_OFFSET +
                                             UL2_CHAR_CTR_RECBITS_FROM_MDGEN_OFFSET),
                                            &regval);

  printf("CHAR1 RECSTART FROM MDGEN: %u\n", regval & 0x0000FFFF);
  printf("CHAR1 RECSTOP FROM MDGEN:  %u\n", (regval >> 16) & 0x0000FFFF);

  success = success && ADQ_ReadUserRegister(adq_cu, adq_num, 2,
                                            (UL2_CHAR0_BASE_ADDR + UL2_CHAR_OFFSET +
                                             UL2_CHAR_CTR_RECBITS_FROM_OGEN_OFFSET),
                                            &regval);

  printf("CHAR1 RECSTART FROM OGEN:  %u\n", regval & 0x0000FFFF);
  printf("CHAR1 RECSTOP FROM OGEN:   %u\n", (regval >> 16) & 0x0000FFFF);
  printf("**************************************\n");

  return success;
}

int RegisterDump(void* adq_cu, int adq_num, unsigned int base_addr, unsigned int range) {
  printf("***** REGISTER DUMP 0-%u (0x%08X) *****\n", range-1, base_addr);
  for (unsigned int i = 0; i < range; ++i) {
    unsigned int regval = ADQ_ReadRegister(adq_cu, adq_num, base_addr+i*4);
    printf("%2u: 0x%08X\n", i, regval);
  }
  printf("***********************************************\n");

  return 1;
}

int ShowDRAMThresholds(void* adq_cu, int adq_num)
{
  unsigned int regval = 0xdeadbeef;

  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0x8C);
  printf("FIFO Fill: 0x%08X\n", regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0x90);
  printf("FIFO fill threshold < 12.5%%: 0x%08X %u\n", regval, regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0x94);
  printf("FIFO fill threshold < 25.0%%: 0x%08X %u\n", regval, regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0x98);
  printf("FIFO fill threshold < 37.5%%: 0x%08X %u\n", regval, regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0x9C);
  printf("FIFO fill threshold < 50.0%%: 0x%08X %u\n", regval, regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0xA0);
  printf("FIFO fill threshold < 62.5%%: 0x%08X %u\n", regval, regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0xA4);
  printf("FIFO fill threshold < 75.0%%: 0x%08X %u\n", regval, regval);
  regval = ADQ_ReadRegister(adq_cu, adq_num, 0x50080000+0xA8);
  printf("FIFO fill threshold < 87.5%%: 0x%08X %u\n", regval, regval);

  return 1;
}
