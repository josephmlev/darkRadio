#ifndef GEN3_STREAMING_SETTINGS_H_JPH8GB
#define GEN3_STREAMING_SETTINGS_H_JPH8GB

#include "ADQAPI.h"

/* The timeout in milliseconds to wait for a record buffer. */
#define WAIT_TIMEOUT_MS 1000

/* The trigger event source for each channel. Valid values are:
   - ADQ_EVENT_SOURCE_SOFTWARE
   - ADQ_EVENT_SOURCE_TRIG
   - ADQ_EVENT_SOURCE_LEVEL
   - ADQ_EVENT_SOURCE_SYNC
   - ADQ_EVENT_SOURCE_PERIODIC */
static const enum ADQEventSource TRIGGER_SOURCE[ADQ_MAX_NOF_CHANNELS] = {
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
  ADQ_EVENT_SOURCE_SOFTWARE,
};

/* The edge sensitivity of the selected trigger event source. The support for
   these values varies between the event sources, e.g. ADQ_EVENT_SOURCE_SOFTWARE
   only supports ADQ_EDGE_RISING. */
static const enum ADQEdge TRIGGER_EDGE[ADQ_MAX_NOF_CHANNELS] = {
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
  ADQ_EDGE_RISING,
};

/* The threshold of the TRIG port in volts. */
static const double TRIG_THRESHOLD = 0.5;

/* Periodic event source settings. The period is measured in ADC sampling
   periods. */
static const int PERIODIC_EVENT_SOURCE_PERIOD = 1024 * 1024;

/* Level trigger settings. */
static const int LT_LEVEL[ADQ_MAX_NOF_CHANNELS] = {2000, -1000, -1000, -1000, -1000, -1000, -1000, -1000};
static const int LT_RESET_LEVEL[ADQ_MAX_NOF_CHANNELS] = {100, 100, 100, 100, 100, 100, 100, 100};

/* Horizontal offset of the trigger point. Offset zero places the trigger point
   at the first sample in the record. */
static const int64_t HORIZONTAL_OFFSET[ADQ_MAX_NOF_CHANNELS] = {-32, -32, 0, 0, 0, 0, 0, 0};

/* The number of records to acquire for each channel. Set the value to zero to
   disable the channel. Set to ADQ_INFINITE_NOF_RECORDS to specify an unbounded
   acquisition. */
static const int64_t NOF_RECORDS[ADQ_MAX_NOF_CHANNELS] = {10, 10, 0, 0, 0, 0, 0, 0};

/* Record lengths per channel */
static const int64_t RECORD_LENGTH[ADQ_MAX_NOF_CHANNELS] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

/* Sample skip factor per channel */
static const int SAMPLE_SKIP_FACTOR[ADQ_MAX_NOF_CHANNELS] = {1, 1, 1, 1, 1, 1, 1, 1};

/* Digital baseline stabilization (DBS) */
static const int DBS_LEVEL[ADQ_MAX_NOF_CHANNELS] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
static const int DBS_BYPASS[ADQ_MAX_NOF_CHANNELS] = {1, 1, 1, 1, 1, 1, 1, 1};

/* DC offset (in ADC codes [-32768, 32767]). */
static const int DC_OFFSET[ADQ_MAX_NOF_CHANNELS] = {0, 0, 0, 0, 0, 0, 0, 0};

/* Enable or disable the file writing. */
#define WRITE_TO_FILE 0
#define BASE_FILENAME "./data/data"

/* MEMORY_OWNER_USER specifies whether the user application is reponsible for
   the memory used to store the record data (if nonzero) or if this is the
   responsibility of the API (if set to zero). The memory allocation performed
   by the API is constrained by the data readout parameters. Refer to the
   streaming user guide (20-2465) for more information. */
#define MEMORY_OWNER_USER 0
#define NOF_RECORD_BUFFERS (16)
#define USER_RECORD_BUFFER_SIZE (1024 * 1024)

/* INCOMPLETE_RECORDS is required if the record length is set to
   'ADQ_INFINITE_RECORD_LENGTH'. This aquisition mode is also known as
   'continuous streaming'. */
#define INCOMPLETE_RECORDS 0

#endif
