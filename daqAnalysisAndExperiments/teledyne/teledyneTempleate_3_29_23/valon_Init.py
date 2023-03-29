import settings as s
from daqHelpers import valonCom
import time


'''
Sets up valon for run 1.
No mixers, uses only ch 1 output for clock.
Cleans valon settings using "CLE"
'''

if s.ADQ_CLOCK_SOURCE:
    print("Begin Valon Config! \n")

    # Cleanse all settings
    valonCom('cle')

    ################# Set Amplitude ################

    # Valon output ~+17dBm (23MHz) to +13dBm (6GHz) (frequenc).
    # Default attenuator = -15dB
    # Teledyne expects 0.5 to 3V ~ -2 to +13dBm
    # Set att to -10dB for between +3 and +7dBm clock to teledyne 
    valonCom('s1;att10')

    # If using external 10MHz source for valon refrence 
    if s.VALON_EXT_10MHZ:
        valonCom('s1;refs1')
        valonCom('s1;ref 10m')
        #valonCom('')
    
    ################# Set Frequency #################

    # Teledyne only excepts between 1 and 2.5GHz
    if s.CLOCK_RATE < 1e9 or s.CLOCK_RATE > 2.5e9:
            print('******** CLOCK_RATE set out of bounds ********')
            raise Exception('Clock Error')

    freqReturn = valonCom(f's1;f{s.CLOCK_RATE}')

    lockReturn = valonCom('LK?')
    for lockedStr in lockReturn:
        if 'not locked' in lockedStr:
            print('******** Valon not locked ********')
            raise Exception('Lock Error')

    print('\n################')
    print(f"Valon set to {freqReturn[1]} with Teledyne sample skip = {s.CH0_SAMPLE_SKIP_FACTOR}")
    print(f"ADQ sample rate of {s.SAMPLE_RATE/1e6} MHz")
    print(f"Span: 0 - {s.SAMPLE_RATE/2/1e6} MHz")
    print('################\n')


    #valonStatus = valonCom('stat')



    print('Valon config success!\n')
    print(f"Have a good run!")

