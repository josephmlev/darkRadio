/*
 *  Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */
#ifndef SETTINGS_H_UR7M3_NV
#define SETTINGS_H_UR7M3_NV

#include "ADQAPI.h"

#include "ADQAPI.h"

/* The trigger event source for each channel. Valid values are:
   - ADQ_EVENT_SOURCE_SOFTWARE
   - ADQ_EVENT_SOURCE_TRIG
   - ADQ_EVENT_SOURCE_LEVEL
   - ADQ_EVENT_SOURCE_SYNC
   - ADQ_EVENT_SOURCE_PERIODIC
  NOTE: Size of all data to collect must be < 8GB with ADQ_EVENT_SOURCE_SOFTWARE in this example */

#define TRIGGER_SOURCE ADQ_EVENT_SOURCE_PERIODIC

#define CH0_TRIGGER_SOURCE TRIGGER_SOURCE
#define CH1_TRIGGER_SOURCE TRIGGER_SOURCE

/* The edge sensitivity of the selected trigger event source. The support for
   these values varies between the event sources, e.g. ADQ_EVENT_SOURCE_SOFTWARE
   only supports ADQ_EDGE_RISING. */
#define CH0_TRIGGER_EDGE ADQ_EDGE_RISING
#define CH1_TRIGGER_EDGE ADQ_EDGE_RISING

/* The trigger threshold in volts. */
#define TRIGGER_THRESHOLD_V 0.00005

/* Periodic event source settings. The 'period' value takes precedence over the
   frequency. Set the period to zero to configure the event source using the
   frequency value.  */
#define PERIODIC_EVENT_SOURCE_PERIOD 1<<4
#define PERIODIC_EVENT_SOURCE_FREQUENCY 0

/* Level trigger settings. */
#define CH0_LT_LEVEL 100
#define CH0_LT_ARM_HYSTERESIS 500
#define CH1_LT_LEVEL -1000
#define CH1_LT_ARM_HYSTERESIS 100

/* Horizontal offset of the trigger point. Offset zero places the trigger. */
#define CH0_HORIZONTAL_OFFSET 0
#define CH1_HORIZONTAL_OFFSET 0


/* The number of active channels: valid values are '1' or '2'. When there's only
   one active channel, that channel is channel 0. */
#define NOF_CHANNELS 1

/* Record lengths per channel. */
#define RECORD_LEN 1<<24

#define CH0_RECORD_LEN RECORD_LEN
#define CH1_RECORD_LEN RECORD_LEN

/* The number of records per received buffer. */
// This number and the recrod size can be used to tweak transfer performance
#define NOF_RECORDS_PER_BUFFER 1

/* Number of target GPU buffers (per channel). */
#define NOF_GPU_BUFFERS 2

#define TOTAL_NOF_GPU_BUFFERS NOF_GPU_BUFFERS*NOF_CHANNELS

/* Number of ADQ transfer buffers (per channel). */
#define NOF_TRANSFER_BUFFERS 8

/* The total number of buffers to receive before the measurement is complete. */
#define NOF_BUFFERS_TO_RECEIVE 4

/* The timeout in milliseconds to wait for a record buffer. */
#define WAIT_TIMEOUT_MS 1000


/* Sample skip factor per channel. */
#define SAMPLE_SKIP_FACTOR 1
#define CH0_SAMPLE_SKIP_FACTOR SAMPLE_SKIP_FACTOR
#define CH1_SAMPLE_SKIP_FACTOR SAMPLE_SKIP_FACTOR

/* Digital baseline stabilization (DBS) */
#define CH0_DBS_LEVEL 1024
#define CH0_DBS_BYPASS 1
#define CH1_DBS_LEVEL -3192
#define CH1_DBS_BYPASS 1

/* Analog front-end (AFE) */
#define CH0_DC_OFFSET 0.0
#define CH1_DC_OFFSET 0.0

/* Available test pattern signals:
   - ADQ_TEST_PATTERN_SOURCE_DISABLE (required to see ADC data)
   - ADQ_TEST_PATTERN_SOURCE_COUNT_UP
   - ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN
   - ADQ_TEST_PATTERN_SOURCE_TRIANGLE */
#define CH0_TEST_PATTERN_SOURCE ADQ_TEST_PATTERN_SOURCE_DISABLE
#define CH1_TEST_PATTERN_SOURCE ADQ_TEST_PATTERN_SOURCE_DISABLE

/* Enable or disable printout after run. */
#define PRINT_LAST_BUFFERS 1


#endif // ifndef SETTINGS_H_UR7M3_NV
