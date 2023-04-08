# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
import pyadq

NOF_CHANNELS                    = 2 #can't currently change. Need to load 1 ch firmware

#Can't go highter than 2^24 on DR2 without modifying bios settings. Stay above 2^16
CH0_RECORD_LEN                  = int(2**24) 
CH1_RECORD_LEN                  = CH0_RECORD_LEN #Different lengths is untested

NOF_BUFFERS_TO_RECEIVE          = 1000000 #for each call of avgFFT()
NOF_ACQUISITIONS_TO_TAKE        = 1  #number of times to call avgFFT()

NOF_GPU_BUFFERS                 = 2 #per channel. Crashes if not 2.

##########################################################
# CLOCKING
ADQ_CLOCK_SOURCE                = 1 # 0 = INTERNAL, 1 = EXTERNAL
				    # If external clock is used, it must be set seperatly!!!

CH0_SAMPLE_SKIP_FACTOR          = 2  #For factors below 8, only power-of-two values are allowed 
CH1_SAMPLE_SKIP_FACTOR          = CH0_SAMPLE_SKIP_FACTOR #Different values is untested

# Must be 2.5e9 if ADQ_CLOCK_SOURCE==1
# Else, must be between 1e9 and  2.5e9. 
# Use sample skip to get lower than 1GHz
# Should be a multple of 20MHz for best performance
CLOCK_RATE                      = 1.280e9

VALON_EXT_10MHZ                 = 1

SAMPLE_RATE                     = CLOCK_RATE/CH0_SAMPLE_SKIP_FACTOR 

PERIODIC_EVENT_SOURCE_PERIOD    = int(CH0_RECORD_LEN + 100)  

#PERIODIC_EVENT_SOURCE_PERIOD    =  int((CH0_RECORD_LEN/SAMPLE_RATE * 2.5E9) + 100)

#PERIODIC_EVENT_SOURCE_PERIOD    = int(2800) #For contunious acquisition use 2800. API bug? 
                                            #Can be used to reduce data rate 


##########################################################
# SAVING
# NOF_ACQUISITIONS_TO_TAKE should be 1
SAVE_W_SPEC                     = 0 #Saves last spec to .npy. Not tested
PATH_TO_SAVE_SINGLE_SPEC        ='/drBigBoy/darkRadio/daqAnalysisAndExperiments/run1p3/rfInterferenceTesting/data_gitignore/W_dict_ultraParanoid_everythingoff_1MFFTs_3_26_23_v2' #Where to save above

NUM_SPEC_PER_FILE               = 16 #How many spectra to put in a file. Keep files around 1GB. 16 is good
SAVE_DIRECTORY                  = '/drBiggerBoy/moreTesting_3_23_23/' #directory to save data. Note this needs to be created ahead of time
                                                                #and there should be a subdirectory called data. 

SAVE_H5                         = 0 # Should h5 be saved and database be updated. If 0, just plotting and SAVE_W_SPEC

SWITCH                          = 0 #Controlls if a switch should be used
SWITCH_SLEEP_TIME		= .005 #Time in seconds to sleep after switching

MANUAL_ACQNUM                   = -1 # Manually set acquisition number. Set to -1 to 
                                     # get from database. NOT TESTED

READ_ONLY_H5                    = 1 # currently unused

ANT_POS_IDX                     = 0 # Written to H5 and database, does not affect behavor.
                                    # Should be in setup dict below, but I want it to be obvious

SAVE_AMP_CHAIN                  = 1 # save the following dictonary. Can be modified
SETUP_DICT                      = { 'AMP1'          : 'PNK_1012',
                                    'AMP2'          : 'ZKL_9VNom',
                                    #'VOLTAGE'       : '9P051V',
                                    'PATCH PANNEL'  : 'YES',
                                    'ATTENUATOR'    : '10dB_HP_Variable', 
                                    'HPF'           : '288S+',
                                    'LPF'           : 'HSP50+',
                                    'ADC'           : 'ADQ32',
                                    'CLOCK'         : 'SRS_VIA_VALON',
                                    }


##########################################################
# TESTING
VERBOSE_MODE                    = 0 #Print info
PRINT_BUF_COUNT                 = 0 #Prints how many buffers have been collected
TEST_MODE                       = 0 #Copy to cpu, ~100ms overhead
chToTest                        = 0 #Sets which channel to plot and save. assumes TEST_MODE = 1
saveBuffer                      = 0 #Sets if the buffer should be saved.   //      //      //
pltTimeDomain                   = 0 #Plots time domain data                //      //      // 
# Available test pattern signals:
#   - ADQ_TEST_PATTERN_SOURCE_DISABLE (required to see ADC data)
#   - ADQ_TEST_PATTERN_SOURCE_COUNT_UP
#   - ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN
#   - ADQ_TEST_PATTERN_SOURCE_TRIANGLE
CH0_TEST_PATTERN_SOURCE = pyadq.ADQ_TEST_PATTERN_SOURCE_DISABLE
CH1_TEST_PATTERN_SOURCE = pyadq.ADQ_TEST_PATTERN_SOURCE_DISABLE

##########################################################
# Legacy settings that likely are unused for avg FFT
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
