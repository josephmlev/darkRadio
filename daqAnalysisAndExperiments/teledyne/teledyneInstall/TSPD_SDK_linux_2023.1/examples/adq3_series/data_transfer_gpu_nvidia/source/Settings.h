/*
 *  Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */
#ifndef SETTINGS_H_UR7M3_NV_LX
#define SETTINGS_H_UR7M3_NV_LX

#include "ADQAPI.h"

/* The trigger event source for each channel. Valid values are:
   - ADQ_EVENT_SOURCE_TRIG
   - ADQ_EVENT_SOURCE_LEVEL
   - ADQ_EVENT_SOURCE_SYNC
   - ADQ_EVENT_SOURCE_PERIODIC */
#define CH0_TRIGGER_SOURCE ADQ_EVENT_SOURCE_PERIODIC
#define CH1_TRIGGER_SOURCE ADQ_EVENT_SOURCE_PERIODIC

/* The edge sensitivity of the selected trigger event source. The support for
   these values varies between the event sources. */
#define CH0_TRIGGER_EDGE ADQ_EDGE_RISING
#define CH1_TRIGGER_EDGE ADQ_EDGE_RISING

/* The trigger threshold in volts. */
#define TRIGGER_THRESHOLD_V 0.5

/* Periodic event source settings. The 'period' value takes precedence over the
   frequency. Set the period to zero to configure the event source using the
   frequency value.  */
#define PERIODIC_EVENT_SOURCE_PERIOD (1024 * 2)
#define PERIODIC_EVENT_SOURCE_FREQUENCY 0

/* Level trigger settings. */
#define CH0_LT_LEVEL 2000
#define CH0_LT_ARM_HYSTERESIS 100
#define CH1_LT_LEVEL -1000
#define CH1_LT_ARM_HYSTERESIS 100

/* Horizontal offset of the trigger point. Offset zero places the trigger. */
#define CH0_HORIZONTAL_OFFSET 0
#define CH1_HORIZONTAL_OFFSET 0

/* Number of active digitizers */
#define NOF_DIGITIZERS 1

/* The number of active channels per digitizer: valid values are '1' or '2'. When there's only
   one active channel, that channel is channel 0. */
#define NOF_CHANNELS 1

/* Record lengths per channel. */
#define CH0_RECORD_LEN 1024
#define CH1_RECORD_LEN 1024

/* The number of records per received buffer. */
#define NOF_RECORDS_PER_BUFFER 1024

/* Number of target GPU buffers (per channel, per digitizer). */
#define NOF_GPU_BUFFERS 4

/* The total number of buffers to receive per channel before the measurement is complete. */
#define NOF_BUFFERS_TO_RECEIVE 1000

/* The timeout in milliseconds to wait for a record buffer, 0 recommended when NOF_DIGITIZERS > 1.
 */
#define WAIT_TIMEOUT_MS 1000

/* Sample skip factor per channel. */
#define CH0_SAMPLE_SKIP_FACTOR 1
#define CH1_SAMPLE_SKIP_FACTOR 1

/* Digital baseline stabilization (DBS) */
#define CH0_DBS_LEVEL 1024
#define CH0_DBS_ENABLED 0
#define CH1_DBS_LEVEL -3192
#define CH1_DBS_ENABLED 0

/* Analog front-end (AFE) */
#define CH0_DC_OFFSET 0.0
#define CH1_DC_OFFSET 0.0

/* Available test pattern signals:
   - ADQ_TEST_PATTERN_SOURCE_DISABLE (required to see ADC data)
   - ADQ_TEST_PATTERN_SOURCE_COUNT_UP
   - ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN
   - ADQ_TEST_PATTERN_SOURCE_TRIANGLE */
#define CH0_TEST_PATTERN_SOURCE ADQ_TEST_PATTERN_SOURCE_COUNT_UP
#define CH1_TEST_PATTERN_SOURCE ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN

/* Enable or disable printout after run. */
#define PRINT_LAST_BUFFERS 1

/* Period in seconds for status printouts during acquisition , set to 0 to disable*/
#define PRINTOUT_PERIOD 2

#endif // ifndef SETTINGS_H_UR7M3_NV_LX
