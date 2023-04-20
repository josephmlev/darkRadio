/*
 *  Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */
#ifndef DATA_READOUT_SETTINGS_H_8BEI7N
#define DATA_READOUT_SETTINGS_H_8BEI7N

#include "ADQAPI.h"

/* The timeout in milliseconds to wait for a record buffer. */
#define WAIT_TIMEOUT_MS 1000

/* The trigger event source for each channel. Valid values are:
   - ADQ_EVENT_SOURCE_SOFTWARE
   - ADQ_EVENT_SOURCE_TRIG
   - ADQ_EVENT_SOURCE_LEVEL
   - ADQ_EVENT_SOURCE_SYNC
   - ADQ_EVENT_SOURCE_PERIODIC */
#define TRIGGER_SOURCE ADQ_EVENT_SOURCE_LEVEL

/* The edge sensitivity of the selected trigger event source. The support for
   these values varies between the event sources, e.g. ADQ_EVENT_SOURCE_SOFTWARE
   only supports ADQ_EDGE_RISING. */
#define TRIGGER_EDGE ADQ_EDGE_RISING

/* The trigger threshold in volts. */
#define TRIGGER_THRESHOLD_V 0.5

/* Periodic event source settings. The 'period' value takes precedence over the
   frequency. Set the period to zero to configure the event source using the
   frequency value.  */
#define PERIODIC_EVENT_SOURCE_PERIOD 0
#define PERIODIC_EVENT_SOURCE_FREQUENCY 1000.0

/* Level trigger settings. */
#define LT_LEVEL 2000
#define LT_ARM_HYSTERESIS 100

/* Horizontal offset of the trigger point. Offset zero places the trigger. */
#define HORIZONTAL_OFFSET 0

/* The number of active channels: valid values are '1' or '2'. When there's only
   one active channel, that channel is channel 0. */
#define NOF_ACTIVE_CHANNELS 2

/* Record length. */
#define RECORD_LENGTH 1024

/* The number of records per received buffer. */
#define NOF_RECORDS_PER_BUFFER 4

/* Between 2 and 16 transfer buffers are supported. */
#define NOF_TRANSFER_BUFFERS 4

/* The total number of buffers to receive before the measurement is complete. */
#define NOF_BUFFERS_TO_RECEIVE 5

/* Sample skip factor. */
#define SAMPLE_SKIP_FACTOR 1

/* Digital baseline stabilization (DBS) */
#define DBS_LEVEL 1024
#define DBS_ENABLED 0

/* Analog front-end (AFE) */
#define DC_OFFSET 0.0

/* Available test pattern signals:
   - ADQ_TEST_PATTERN_SOURCE_DISABLE (required to see ADC data)
   - ADQ_TEST_PATTERN_SOURCE_COUNT_UP
   - ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN
   - ADQ_TEST_PATTERN_SOURCE_TRIANGLE */
#define TEST_PATTERN_SOURCE ADQ_TEST_PATTERN_SOURCE_TRIANGLE

/* Accumulation setting for the ATD signal processing model. This parameter is
   only used if the digitizer is running FWATD firmware. */
#define NOF_ACCUMULATIONS 100

/* Enable or disable the file writing. */
#define WRITE_TO_FILE 1
#define BASE_FILENAME "./data/data"

#endif
