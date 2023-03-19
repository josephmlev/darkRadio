/*
 *
 * Copyright Signal Processing Devices Sweden AB. All rights reserved.
 * See document "08-0175 EULA" for specific license terms regarding this file.
 *
 * Description   : Declaration of debugging functions for ADQ14-Tarsier
 * Documentation :
 *
 */

#ifndef DEBUGGING_H_
#define DEBUGGING_H_

int PrintEventCounters(void* adq_cu, int adq_num);
int PrintRecordCounters(void* adq_cu, int adq_num);
int RegisterDump(void* adq_cu, int adq_num, unsigned int base_addr, unsigned int range);
int ShowDRAMThresholds(void* adq_cu, int adq_num);

#endif
