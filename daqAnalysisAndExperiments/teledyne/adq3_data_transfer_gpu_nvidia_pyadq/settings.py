# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
import pyadq

NOF_CHANNELS                    = 2 #can't currently change. Need to load 1 ch firmware

#Can't go highter than 2^24 on DR2 without modifying bios settings. Stay above 2^16
CH0_RECORD_LEN                  = int(2**24) 
CH1_RECORD_LEN                  = CH0_RECORD_LEN #Different lengths is untested

NOF_BUFFERS_TO_RECEIVE          = 100 #for each call of avgFFT()
NOF_ACQUISITIONS_TO_TAKE        = 8000  #number of times to call avgFFT()

CH0_SAMPLE_SKIP_FACTOR          = 2  #For factors below 8, only power-of-two values are allowed 
CH1_SAMPLE_SKIP_FACTOR          = CH0_SAMPLE_SKIP_FACTOR #Different values is untested

PERIODIC_EVENT_SOURCE_PERIOD    = int(2800) #for contunious aquation needs to be smaller
                                    #than record_len. It like 2800 for some reason

CLOCK_RATE                      = 2.5e9 #default. Add external clock soon
SAMPLE_RATE                     = CLOCK_RATE/CH0_SAMPLE_SKIP_FACTOR 

NOF_GPU_BUFFERS                 = 2 #per channel

##########################################################
# testing
VERBOSE_MODE                    = 0 #for testing

pltTimeDomain                   = 0
chToTest                        = 0 #Sets which channel to plot and save. assumes VERBOSE_MODE = 1
saveBuffer                      = 0 #Sets if the buffer should be saved.   //      //      //

# Available test pattern signals:
#   - ADQ_TEST_PATTERN_SOURCE_DISABLE (required to see ADC data)
#   - ADQ_TEST_PATTERN_SOURCE_COUNT_UP
#   - ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN
#   - ADQ_TEST_PATTERN_SOURCE_TRIANGLE
CH0_TEST_PATTERN_SOURCE = pyadq.ADQ_TEST_PATTERN_SOURCE_DISABLE
CH1_TEST_PATTERN_SOURCE = pyadq.ADQ_TEST_PATTERN_SOURCE_DISABLE

##########################################################
# legacy settings that likely are unused for avg FFT
# or shouldn't be modified 

NOF_RECORDS_PER_BUFFER = 1 

WAIT_TIMEOUT_MS = 5000

CH0_TRIGGER_SOURCE = pyadq.ADQ_EVENT_SOURCE_PERIODIC
CH1_TRIGGER_SOURCE = pyadq.ADQ_EVENT_SOURCE_PERIODIC

BYTES_PER_SAMPLES = 2 #cant change this :(

CH0_TRIGGER_EDGE = pyadq.ADQ_EDGE_RISING
CH1_TRIGGER_EDGE = pyadq.ADQ_EDGE_RISING

TRIGGER_THRESHOLD_V = 0.5

CH0_DBS_LEVEL = 1024
CH0_DBS_BYPASS = 1
CH1_DBS_LEVEL = 1024
CH1_DBS_BYPASS = 1

CH0_DC_OFFSET = 0.0
CH1_DC_OFFSET = 0.0

PERIODIC_EVENT_SOURCE_FREQUENCY = 0 #overwritten by period. See above

CH0_LT_LEVEL = 2000
CH0_LT_ARM_HYSTERESIS = 100
CH1_LT_LEVEL = -1000
CH1_LT_ARM_HYSTERESIS = 100

CH0_HORIZONTAL_OFFSET = 0
CH1_HORIZONTAL_OFFSET = 0