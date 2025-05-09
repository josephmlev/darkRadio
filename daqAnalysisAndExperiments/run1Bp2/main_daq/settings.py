# Copyright 2023 Teledyne Signal Processing Devices Sweden AB

##########################################################
# Run 1Bp2 main daq settings
##########################################################
import pyadq
##########################################################
# General settings
##########################################################
NOF_CHANNELS                    = 2 #can't currently change. Need to load 1 ch firmware

#Can't go highter than 2^24 on DR2 without modifying bios settings. Stay above 2^16
CH0_RECORD_LEN                  = int(2**23) 
CH1_RECORD_LEN                  = CH0_RECORD_LEN #Different lengths not currently accepted

NOF_BUFFERS_TO_RECEIVE          = 81000 #for each call of avgFFT() #run 1Bp1 used 81000
NOF_ACQUISITIONS_TO_TAKE        = 143 #run 1Bp1 used 143
NOF_GPU_BUFFERS                 = 2 #per channel. Crashes if not 2.

INIT_SLEEP_TIME			= 0 #Time in seconds to sleep before starting anything

##########################################################
# CLOCKING
##########################################################
ADQ_CLOCK_SOURCE                = 1 # 0 = INTERNAL, 1 = EXTERNAL
				    # If external clock is used, it must be set up seperatly!!!
				    # See valon_Init.py

CH0_SAMPLE_SKIP_FACTOR          = 1   #For factors below 8, only power-of-two values are allowed 
CH1_SAMPLE_SKIP_FACTOR          = CH0_SAMPLE_SKIP_FACTOR #Different values are untested

# Must be 2.5e9 if ADQ_CLOCK_SOURCE==0
# Else, must be between 1e9 and 2.5e9. 
# Use sample skip to get lower than 1GHz
# Should be a multple of 20MHz for best performance
CLOCK_RATE                      = 2.4e9 #Hz #run 1Bp1 used 2.4e9

VALON_EXT_10MHZ                 = 1

SAMPLE_RATE                     = CLOCK_RATE/CH0_SAMPLE_SKIP_FACTOR 

DEAD_TIME_RATIO			= 2 # 1 = 100%, 2 = 50%, 4 = 25% etc #slows down to allow for data to transfer for high sample rate

PERIODIC_EVENT_SOURCE_PERIOD    = int((CH0_RECORD_LEN + 100) * DEAD_TIME_RATIO)  

##########################################################
# SAVING
##########################################################

# Save a single spectrum for testing or simple DAQ stuff
# NOF_ACQUISITIONS_TO_TAKE should be 1
SAVE_W_SPEC                     = 0 #Saves last spec to .npy.
PATH_TO_SAVE_SINGLE_SPEC        ='/drBigBoy/darkRadio/daqAnalysisAndExperiments/run1B/daq/systemTesting/data_gitignore/antSpec28asdf' #Where to save above

NUM_SPEC_PER_FILE               = 32 #How many spectra to put in a file. Keep files around 1GB. 16 is good
SAVE_DIRECTORY                  = '/drBiggerBoy/run1Bp1_TEMP/' #directory to save data. Note this needs to be created ahead of time
                                                                #and there should be a subdirectory called data. 

SAVE_H5                         = 0 # Should h5 be saved AND database.txt be updated. 

SWITCH                          = 1 #Controlls if a switch should be used. Requires Arduino to be connected. Should configure automatically
SWITCH_SLEEP_TIME		= .5 #Time in seconds to sleep after switching
TEMPERATURE			= 1 # Should temperature be collected with arduino

SWITCH_DUTYCYCLE 		= 15 # Really 1/duty cycle. If loopNum % SWITCH_DUTYCYCLE == 0 then switch. Buggy, test if using.

STIRRER 			= 1 #Controlls if the mode stirrer will be used
STIR_USTEPS_PER_MOVE		= 16 #Must be a multple of 16 to ensure it stops on an A+ position. Assuming 4 usteps

MANUAL_ACQNUM                   = -1 # Manually set acquisition number. Set to -1 to 
                                     # get from database. NOT TESTED
                                     
READ_ONLY_H5                    = 0 # Makes H5 files read only. Annoying for testing, good for actual data. Note, last H5 file will not be made read only

ANT_POS_IDX                     = 8 # Written to H5 and database, does not affect behavor.
                                    # Should be in setup dict below, but I want it to be obvious

SAVE_AMP_CHAIN                  = 1 # save the following dictonary. Can be modified
SETUP_DICT                      = { 'AMP1'          : '1012_E_PbAcid',
                                    'AMP2'          : 'ZKL_9p05VReg',
                                    'ATT1'           : '3_PLUS_1dB_FIXED',
                                    'CABLE'	    : 'PINK_6FT',
                                    'ATT2'           : '1dB',
                                    'LPF'           : 'ZXR250',
				    'HPF'           : '1050',
                                    'ADC'           : 'ADQ32',
                                    'CLOCK'         : 'SRS_VIA_VALON',
                                    }

##########################################################
# TESTING
PLOT_FINAL_SPEC			= 1 # Plots the last call from avgFFT
VERBOSE_MODE                    = 0 # Print info
PRINT_BUF_COUNT                 = 0 # Prints how many buffers have been collected
TEST_MODE                       = 0 # Copy to cpu, ~100ms overhead
CH_TO_TEST                      = 1 # Sets which channel to plot and save. assumes TEST_MODE = 1
SAVE_BUFFER                     = 0 # Sets if the buffer should be saved.   //      //      //
PLOT_TIME_DOMAIN                = 0 # Plots time domain data                //      //      // 
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
##########################################################

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

##########################################################
# Logic to check for valid settings
##########################################################

if SAVE_W_SPEC and NOF_ACQUISITIONS_TO_TAKE > 1:
    raise ValueError("Settings.py: SAVE_W_SPEC requires NOF_ACQUISITIONS_TO_TAKE = 1")
    
if ADQ_CLOCK_SOURCE:
    print(f"\n WARNING: settings.ADQ_CLOCK_SOURCE is set to external and CLOCK_RATE is set to {CLOCK_RATE/1e6}MHz. There are no checks on this. The user is responsable for verifying the clock settings match their expectations. If using Valon, see valon_Init.py \n")
    
if TEST_MODE == 0:
    if SAVE_BUFFER:
        raise ValueError("Settings.py: SAVE_BUFFER requires TEST_MODE == 1")
    if PLOT_TIME_DOMAIN:
        raise ValueError("Settings.py: PLOT_TIME_DOMAIN requires TEST_MODE == 1")
  
if ADQ_CLOCK_SOURCE == 0 and CLOCK_RATE != 2.5E9:
    raise ValueError(f"Settings.py: ADQ_CLOCK_SOURCE is set to 0 (internal). This requires CLOCK_RATE = 2.5 GHz. 2.5GHz is not a strict limit (see ADQ32 manual) but this code is not set up to change it and requires 2.5GHz clock. You can change the sample rate by dividing the clock by a factor of 2 using settings.sample skip. If more flexible clocking is required, use an external clock.")
    
if ADQ_CLOCK_SOURCE == 1 and not(1e9<CLOCK_RATE<=2.5e9):
    raise ValueError(f"Settings.py: ADQ_CLOCK_SOURCE is set to 1 (external) and CLOCK_RATE is set to {CLOCK_RATE/1E6} MHz. This requires 1<CLOCK_RATE<2.5GHz. See valonInit.py to set up the Valon, or change ADQ_CLOCK_SOURCE = 0 (internal) and set CLOCK_RATE = 2.5GHz")
    
if (SAMPLE_RATE*BYTES_PER_SAMPLES*NOF_CHANNELS > 7e9 
    and PERIODIC_EVENT_SOURCE_PERIOD < CH0_RECORD_LEN):
    print('####################') 
    print('WARNING!!! YOUR SETTINGS REQUEST A DATA RATE GREATER THAN THE ALLOWED 7GB/s')
    print('THIS MAY WORK FOR ~< 100 RECORDS TO COLLECT, BUT WILL CRASH')
    print('####################')
    print()

if CLOCK_RATE % 20e6 != 0:
	raise ValueError("Settings.py: Clock rate must be a multple of 20MHz else Valon has to do math and has bad spurs.")


if TEMPERATURE and not SWITCH:
	raise ValueError('TEMPERTURE only works if SWITCH is on')
