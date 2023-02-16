"""Test ability of dev object to be passed to a method"""
import pyadq
import ctypes as ct

acu: pyadq.ADQControlUnit = pyadq.ADQControlUnit()
# Enable trace logging
acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

device_list = acu.ListDevices()
print(f"Found {len(device_list)} device(s)")
# Set up the first available device
device_to_open = 0
dev: pyadq.ADQ = acu.SetupDevice(device_to_open)
'''
if dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
    print("Success setting up dev object \n")
else:
    print("Failed setting up dev object \n")'''

class testADQ:
    '''
    Test class for ADQ api
    Args:
        dev: device object as defined by teledyne API
    Methods:
        init: inits self.dev = dev which is passed in
        checkDev: simple check to see if dev was passed correctly
        start: call dev.startDataAcquisition method and check for errors
        acquire: call dev.waitForP2pBuffers method and check for errors
    '''
    def __init__(self,
                dev
                ):
        self.dev = dev

    def checkDev(self):
        print('called checkDev method')
        print(f"Setting up data collection for: {self.dev} \n")

    def start(self):
        print('called start method')
        if self.dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
            print("Success starting in main\n")
        else:
            print('Failed to start \n')

    def acquire(self):
        print('Called acquire method')
        status = pyadq.ADQP2pStatus()._to_ct()
        print('status:', status)
        result = dev.ADQ_WaitForP2pBuffers(ct.byref(status), 
                                        3000)
        if result == pyadq.ADQ_EAGAIN:
            print("Timeout")
        elif result < 0:
            print(f"Failed with retcode {result}")
            exit(1)
        else:
            print('sucess!')

    def startAndAcquire(self):
        print('called start and acquire method')
        if self.dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
            print("Success starting in main\n")
        else:
            print('Failed to start \n')

        status = pyadq.ADQP2pStatus()._to_ct()
        print('status:', status)
        result = dev.ADQ_WaitForP2pBuffers(ct.byref(status), 
                                        3000)
        if result == pyadq.ADQ_EAGAIN:
            print("Timeout")
        elif result < 0:
            print(f"Failed with retcode {result}")
            exit(1)
        else:
            print('sucess!')
        

            
avgObj = testADQ(dev) # init
#avgObj.checkDev() #print
#avgObj.start()
#avgObj.acquire()
avgObj.startAndAcquire()