# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
""" P2P streaming ADQ -> GPU in python """

import pyadq
from typing import List, Tuple
import cupy as cp
import settings as s
import ctypes as ct
import streaming_helpers as sh
import helper_cupy as hc
import gdrapi as g
from gdrapi import gdr_check_error_exit_func
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import h5py
import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import time

def writeH5(specDict,
            runInfoDict,
            acqNum,
            numSpecPerFile,
            dataDir):
    '''
    Given a spectrum and some info, packs an H5 file every numSpecPerFile
    spectra.
    
    Inputs:
    specDict         : dictionary of np arrays
        Dict containing arrays of spectra
    runInfoDict      : dictionary
        Dict containing info about the run
    acqNum          : int
        Number of acqusition since start of run
    numSpecPerFile  : int
        How many spectra to write per file. Should be set
        to keep files around 1GB (about 16 for two 2^24 FFTs)
    dataDir         : str
        directory to save data. Note this needs to be created ahead of time
        and there should be a subdirectory called data. 
    '''

    #number games to figure out what file to write
    mod         = acqNum%numSpecPerFile
    fileNum     = int((acqNum-mod)/numSpecPerFile)
    

    #create file object. Creates h5 file if needed, else appends to existing file ('a' flag) 
    fileName    = dataDir+'data/'+str(fileNum)+'.hdf5'
    f       = h5py.File(fileName, 'a') 
    #create new group object for each acqusition
    acqGrp  = f.create_group(str(acqNum))

    #pack spectra as dataseta
    for specName in specDict:
        acqGrp.create_dataset(specName, data=specDict[specName], dtype = 'f')
    
    #pack run into as attributes
    for infoName in runInfoDict:
        acqGrp.attrs[infoName] = runInfoDict[infoName]
    acqGrp.attrs['File Number'] = fileNum
    
    #if on a new file, make previous file read only. Note last file wont be read only
    if mod == 0 and fileNum != 0:
        os.chmod(dataDir+'data/'+str(fileNum-1)+'.hdf5', S_IREAD|S_IRGRP|S_IROTH)

    #write to text file
    if not os.path.exists(dataDir+'database.txt'):
        infoStr = ''
        for infoKey in runInfoDict:
            infoStr += infoKey + ','
        infoStr += 'File Number\n' 

        file = open(dataDir + 'database.txt', 'w')
        file.writelines(infoStr)
        file.close()

    lineToWrite = ''
    for infoKey in runInfoDict:
        infoData = str(runInfoDict[infoKey])
        lineToWrite += infoData + ', '
    lineToWrite += str(fileNum) +'\n'

    file = open(dataDir + 'database.txt', 'a')
    file.writelines(lineToWrite)
    file.close()

def allocate_and_pin_buffer(
    buffer_size: int,
    memory_handle: g.GdrMemoryHandle,
    gdr: g.Gdr,
    bar_ptr_data: sh.BarPtrData,
) -> Tuple[ct.c_void_p, ct.c_uint64, cp.ndarray]:
    """Allocate and pin buffers.

    Args:
        `buffer_size`: Size to allocate on GPU.
        `memory_handle`: Wrapped memory_handle struct from gdrapi.
        `gdr`: Wrapped gdr object from gdrapi.
        `bar_ptr_data`: Pointer to data on bar.

    Returns:
        `buffer_pointer`: Pointer to GPU buffer.
        `buffer_address`: Physical address to buffer.
        `buffer`: Buffer object.
    """
    info = g.GdrInfo()

    buffer_size = (buffer_size + g.GPU_PAGE_SIZE - 1) & g.GPU_PAGE_MASK
    buffer = cp.zeros(buffer_size // 2, dtype=cp.short)  # Allocate memory in GPU
    buffer_ptr = buffer.data.ptr  # Pointer of memory

    # Map device memory buffer on GPU BAR1, returning an handle.
    gdr_status = gdrapi.gdr_pin_buffer(
        gdr, ct.c_ulong(buffer_ptr), buffer_size, 0, 0, ct.byref(memory_handle)
    )
    gdr_check_error_exit_func(gdr_status or (memory_handle == 0), "gdr_pin_buffer")
    # Create a user-space mapping for the BAR1 info, length is bar1->buffer
    gdr_status = gdrapi.gdr_map(gdr, memory_handle, ct.byref(bar_ptr_data), buffer_size)
    gdr_check_error_exit_func(gdr_status or (bar_ptr_data == 0), "gdr_map")
    # Bar physical address will be aligned to the page size before being mapped in user-space
    # so the pointer returned might be affected by an offset.
    # gdr_get_info is used to calculate offset.

    gdr_status = gdrapi.gdr_get_info(gdr, memory_handle, info)

    gdr_check_error_exit_func(gdr_status, "gdr_info")
    offset_data = info.va - buffer_ptr

    buffer_address = ct.c_uint64(info.physical)
    gdr_status = gdrapi.gdr_validate_phybar(gdr, memory_handle)
    gdr_check_error_exit_func(gdr_status, "gdr_validate_phybar")
    buffer_pointer = ct.c_void_p(bar_ptr_data.value + offset_data)

    return buffer_pointer, buffer_address, buffer



gdrapi = g.GdrApi()

acu: pyadq.ADQControlUnit = pyadq.ADQControlUnit()
# Enable trace logging
acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

# List the available devices
device_list: List[pyadq.ADQInfoListEntry] = acu.ListDevices()

print(f"Found {len(device_list)} device(s)")

# Ensure that at least one device is available
assert device_list

# Set up the first available device
device_to_open = 0
dev: pyadq.ADQ = acu.SetupDevice(device_to_open)

print(f"Setting up data collection for: {dev}")

# Initialize the parameterss with default values
parameters: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)
parameters.event_source.periodic.period = s.PERIODIC_EVENT_SOURCE_PERIOD
parameters.event_source.periodic.frequency = s.PERIODIC_EVENT_SOURCE_FREQUENCY

parameters.test_pattern.channel[0].source = s.CH0_TEST_PATTERN_SOURCE
parameters.test_pattern.channel[1].source = s.CH1_TEST_PATTERN_SOURCE

parameters.signal_processing.sample_skip.channel[0].skip_factor = s.CH0_SAMPLE_SKIP_FACTOR
parameters.signal_processing.sample_skip.channel[1].skip_factor = s.CH1_SAMPLE_SKIP_FACTOR
parameters.acquisition.channel[0].nof_records = (
    s.NOF_RECORDS_PER_BUFFER * s.NOF_BUFFERS_TO_RECEIVE
)
parameters.acquisition.channel[0].record_length = s.CH0_RECORD_LEN
parameters.acquisition.channel[0].trigger_source = s.CH0_TRIGGER_SOURCE
parameters.acquisition.channel[0].trigger_edge = s.CH0_TRIGGER_EDGE
parameters.acquisition.channel[0].horizontal_offset = s.CH0_HORIZONTAL_OFFSET

if s.NOF_CHANNELS > 1:
    parameters.acquisition.channel[1].nof_records = (
        s.NOF_RECORDS_PER_BUFFER * s.NOF_BUFFERS_TO_RECEIVE
    )
    parameters.acquisition.channel[1].record_length = s.CH1_RECORD_LEN
    parameters.acquisition.channel[1].trigger_source = s.CH1_TRIGGER_SOURCE
    parameters.acquisition.channel[1].trigger_edge = s.CH1_TRIGGER_EDGE
    parameters.acquisition.channel[1].horizontal_offset = s.CH1_HORIZONTAL_OFFSET

parameters.transfer.common.write_lock_enabled = 1
parameters.transfer.common.transfer_records_to_host_enabled = 0
parameters.transfer.common.marker_mode = pyadq.ADQ_MARKER_MODE_HOST_MANUAL

parameters.transfer.channel[0].record_length_infinite_enabled = 0
parameters.transfer.channel[0].record_size = (
    s.BYTES_PER_SAMPLES * parameters.acquisition.channel[0].record_length
)
parameters.transfer.channel[0].record_buffer_size = (
    s.NOF_RECORDS_PER_BUFFER * parameters.transfer.channel[0].record_size
)
parameters.transfer.channel[0].metadata_enabled = 0
parameters.transfer.channel[0].nof_buffers = s.NOF_GPU_BUFFERS
parameters.transfer.channel[0].metadata_buffer_size = (
    s.NOF_RECORDS_PER_BUFFER * pyadq.SIZEOF_ADQ_GEN4_HEADER
)

if s.NOF_CHANNELS > 1:
    parameters.transfer.channel[1].record_length_infinite_enabled = 0
    parameters.transfer.channel[1].record_size = (
        s.BYTES_PER_SAMPLES * parameters.acquisition.channel[1].record_length
    )
    parameters.transfer.channel[1].record_buffer_size = (
        s.NOF_RECORDS_PER_BUFFER * parameters.transfer.channel[1].record_size
    )
    parameters.transfer.channel[1].metadata_enabled = 0
    parameters.transfer.channel[1].nof_buffers = s.NOF_GPU_BUFFERS
    parameters.transfer.channel[1].metadata_buffer_size = (
        s.NOF_RECORDS_PER_BUFFER * pyadq.SIZEOF_ADQ_GEN4_HEADER
    )

# Create pointers, buffers and GDR object
memory_handles = [
    [g.GdrMemoryHandle() for x in range(s.NOF_CHANNELS)] for y in range(s.NOF_GPU_BUFFERS)
]
bar_ptr_data = sh.BarPtrData(s.NOF_CHANNELS, s.NOF_GPU_BUFFERS)
#print('bar', self.bar_ptr_data.pointers)
gpu_buffer_ptr = sh.GpuBufferPointers(s.NOF_CHANNELS, s.NOF_GPU_BUFFERS)
gdr = gdrapi.gdr_open()
gpu_buffers = sh.GpuBuffers(
    s.NOF_CHANNELS,
    s.NOF_GPU_BUFFERS,
    s.CH0_RECORD_LEN,
)
gpu_buffer_address = 0

# Allocate GPU buffers
for ch in range(s.NOF_CHANNELS):
    for b in range(s.NOF_GPU_BUFFERS):
        #print(f"allocating ch {ch}, buffer num {b}")
        (
            gpu_buffer_ptr.pointers[ch][b],
            gpu_buffer_address,
            gpu_buffers.buffers[ch][b],
        ) = allocate_and_pin_buffer(
            parameters.transfer.channel[ch].record_buffer_size,
            memory_handles[ch][b],
            gdr,
            bar_ptr_data.pointers[ch][b],
        )
        parameters.transfer.channel[ch].record_buffer_bus_address[b] = gpu_buffer_address
        parameters.transfer.channel[ch].record_buffer[b] = gpu_buffer_ptr.pointers[ch][b]
# Configure digitizer parameterss
dev.SetParameters(parameters)
print('done allocating and configuring \n')

if (s.SAMPLE_RATE*s.BYTES_PER_SAMPLES*s.NOF_CHANNELS > 7e9 
    and s.PERIODIC_EVENT_SOURCE_PERIOD < s.CH0_RECORD_LEN):
    print('####################') 
    print('WARNING!!! YOUR SETTINGS REQUEST A DATA RATE GREATER THAN THE ALLOWED 7GB/s')
    print('THIS MAY WORK FOR ~< 100 RECORDS TO COLLECT, BUT WILL CRASH')
    print('####################')
    print()

class avgFft:
    def __init__(self,
                s
                ):
        self.parameters         = parameters
        self.dev                = dev
        self.gpu_buffer_address = gpu_buffer_address
        self.gpu_buffer_ptr     = gpu_buffer_ptr
        self.gpu_buffers        = gpu_buffers
        self.gdr                = gdr
        self.memory_handles     = memory_handles
        self.bar_ptr_data       = bar_ptr_data

        self.s                  = s


        self.bytes_received = 0
        '''PYADQ STATUS:
        pg 167. 
        ADQP2pStatus Members:
            channel: Array of ADQP2pChannel structs. Each element represents a channel
            flags: N/A
        ADQP2pChannel members:
            flags: N/A
            nof_completed_buffers:
                Number of valid entries in completed_buffers
            completed_buffers:
                indicies of buffers with data available to read. 
        '''
        self.status = pyadq.ADQP2pStatus()._to_ct()

        self.settingsDict = {
            'CH0_SAMPLE_SKIP_FACTOR'    :self.s.CH0_SAMPLE_SKIP_FACTOR,
            'CH1_SAMPLE_SKIP_FACTOR'    :self.s.CH1_SAMPLE_SKIP_FACTOR,
            'CH0_RECORD_LEN'            :self.s.CH0_RECORD_LEN,
            'CH1_RECORD_LEN'            :self.s.CH1_RECORD_LEN
        }
       

    def exit(self):
        print('exiting')
        for ch in range(self.s.NOF_CHANNELS):
            for b in range(self.s.NOF_GPU_BUFFERS):
                if 0:
                    print(f"********ch {ch}, buffer num {b}********")
                    print('buffer size:         ',self.parameters.transfer.channel[ch].record_buffer_size)
                    print('buffer address:      ',self.gpu_buffer_address)
                    print('buffer ptr:          ',self.gpu_buffer_ptr.pointers[ch][b])
                    print('type buffer ptr:     ',type(self.gpu_buffer_ptr.pointers[ch][b]))
                    print('buffer              ', len(self.gpu_buffers.buffers[ch][b]))
                    print('type buffer         ', type(self.gpu_buffers.buffers[ch][b]))

                gdrapi.gdr_unmap(
                    self.gdr,
                    self.memory_handles[ch][b],
                    self.bar_ptr_data.pointers[ch][b],
                    self.s.NOF_RECORDS_PER_BUFFER * self.s.CH0_RECORD_LEN * self.s.BYTES_PER_SAMPLES,
                )
                gdrapi.gdr_unpin_buffer(self.gdr, self.memory_handles[ch][b])
                #print(self.bar_ptr_data.pointers[ch][b], '\n')
        # Free GPU memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        print('done exiting')

    def allocate_and_pin_buffer(
        self,
        buffer_size: int,
        memory_handle: g.GdrMemoryHandle,
        gdr: g.Gdr,
        bar_ptr_data: sh.BarPtrData,
    ) -> Tuple[ct.c_void_p, ct.c_uint64, cp.ndarray]:
        """Allocate and pin buffers.

        Args:
            `buffer_size`: Size to allocate on GPU.
            `memory_handle`: Wrapped memory_handle struct from gdrapi.
            `gdr`: Wrapped gdr object from gdrapi.
            `bar_ptr_data`: Pointer to data on bar.

        Returns:
            `buffer_pointer`: Pointer to GPU buffer. ct void pointer
            `buffer_address`: Physical address to buffer. ct uint64
            `buffer`: Buffer object. cp.ndarray
        """
        info = g.GdrInfo()

        buffer_size = (buffer_size + g.GPU_PAGE_SIZE - 1) & g.GPU_PAGE_MASK
        buffer = cp.zeros(buffer_size // 2, dtype=cp.short)  # Allocate memory in GPU
        buffer_ptr = buffer.data.ptr  # Pointer of memory

        # Map device memory buffer on GPU BAR1, returning an handle.
        gdr_status = gdrapi.gdr_pin_buffer(
            gdr, ct.c_ulong(buffer_ptr), buffer_size, 0, 0, ct.byref(memory_handle)
        )
        gdr_check_error_exit_func(gdr_status or (memory_handle == 0), "gdr_pin_buffer")
        # Create a user-space mapping for the BAR1 info, length is bar1->buffer
        gdr_status = gdrapi.gdr_map(gdr, memory_handle, ct.byref(bar_ptr_data), buffer_size)
        gdr_check_error_exit_func(gdr_status or (bar_ptr_data == 0), "gdr_map")
        # Bar physical address will be aligned to the page size before being mapped in user-space
        # so the pointer returned might be affected by an offset.
        # gdr_get_info is used to calculate offset.

        gdr_status = gdrapi.gdr_get_info(gdr, memory_handle, info)

        gdr_check_error_exit_func(gdr_status, "gdr_info")
        offset_data = info.va - buffer_ptr

        buffer_address = ct.c_uint64(info.physical)
        gdr_status = gdrapi.gdr_validate_phybar(gdr, memory_handle)
        gdr_check_error_exit_func(gdr_status, "gdr_validate_phybar")
        buffer_pointer = ct.c_void_p(bar_ptr_data.value + offset_data)

        return buffer_pointer, buffer_address, buffer

    def avgFft(self):
        # Start timer for measurement
        start_time = time.time()
        # Start timer for regular printouts
        start_print_time = time.time()
        # Init timer for FFTs
        self.fftTime = [] 
        print(f"Start acquiring data for: {self.dev}")
        if self.dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
            print("Success. Begin Acquiring")
        else:
            print("Failed")


        data_transfer_done = 0
        nof_buffers_received = [0,0]
        self.fftSum = torch.as_tensor(cp.zeros(((self.s.CH0_RECORD_LEN//2 + 1),2)), device='cuda')
        self.numFft = [0,0]
        #main transfer loop. Happens NOF_BUFFERS_TO_RECEIVE times
        master_start_time = time.time()
        
        while not data_transfer_done:
            #ti = time.time()
            #wait for buffers to fill. Code locks here until one transfer buffer fills 
            self.result = self.dev.ADQ_WaitForP2pBuffers(ct.byref(self.status), self.s.WAIT_TIMEOUT_MS)
            #handle errors
            if self.result == pyadq.ADQ_EAGAIN:
                print("Timeout")
            elif self.result < 0:
                print(f"Failed with retcode {self.result}")
                exit(1)
            else:  #We have a full buffer
                buf = 0
                while (buf < self.status.channel[0].nof_completed_buffers) or (
                    buf < self.status.channel[1].nof_completed_buffers
                ):
                    for ch in range(self.s.NOF_CHANNELS):
                        if buf < self.status.channel[ch].nof_completed_buffers:  
                            self.buffer_index = self.status.channel[ch].completed_buffers[buf]
                            
                            ####################DO FFT##################
                            bufferTensor            = torch.as_tensor(self.gpu_buffers.buffers[ch][self.buffer_index], device='cuda')
                            fftTime_start           = time.time()
                            self.fftSum[:,ch]       +=torch.pow(torch.abs(torch.fft.rfft(bufferTensor)),2)
                            torch.cuda.synchronize()
                            fftTime_stop            = time.time()
                            
                            self.numFft[ch] += 1
                            self.fftTime.append(fftTime_stop - fftTime_start)
                            
                            self.dev.ADQ_UnlockP2pBuffers(ch, (1 << self.buffer_index)) # A mask of buffer indexes to unlock. p194

                            nof_buffers_received[ch] += 1
                            self.bytes_received += (
                                self.s.NOF_RECORDS_PER_BUFFER * self.s.CH0_RECORD_LEN * self.s.BYTES_PER_SAMPLES
                            )
                    buf += 1
                data_transfer_done = nof_buffers_received[1] >= self.s.NOF_BUFFERS_TO_RECEIVE
                now_time = time.time() - start_time
                print_time = time.time() - start_print_time


                if print_time > 5 and self.s.VERBOSE_MODE:
                    # Check for overflow, stop if overflow
                    overflow_status = self.dev.GetStatus(pyadq.ADQ_STATUS_ID_OVERFLOW)
                    if overflow_status.overflow:
                        print("Overflow, stopping data acquisition...")
                        self.dev.ADQ_StopDataAcquisition()
                        exit("Exited because of overflow.")

                    print("Nof buffers received:", nof_buffers_received)
                    start_print_time = time.time()
        print(f'######## TESTING TIME: {time.time() - start_time}')
        print("Done Acquiring Data \n")
        stop_time = time.time()

        ###################AVERAGE AND CONVERT#################### 
        #average FFT^2 spectra
        self.avgSpec_fftsq_Ch0    = np.asarray(self.fftSum.cpu()[:,0]/self.numFft[0])
        self.avgSpec_fftsq_Ch1    = np.asarray(self.fftSum.cpu()[:,1]/self.numFft[1])

        #convert FFT^2 to power spectra (Watts)
        #P_W = V^2/R
        #V   = 2 * 1/len_FFT * FFT * 2^(ADC code to V) NOTE: 2 is because Rfft
        #P_W = 2 * FFT^2 * (adcCode2V)/R  
        adcCode2V               = 0.5/2**(8*self.s.BYTES_PER_SAMPLES) #max V/max code 
        self.avgSpec_W_Ch0      = 2 * self.avgSpec_fftsq_Ch0 * (adcCode2V**2) /(50*self.s.CH0_RECORD_LEN**2)
        self.avgSpec_W_Ch1      = 2 * self.avgSpec_fftsq_Ch1 * (adcCode2V**2) /(50*self.s.CH1_RECORD_LEN**2)
        
        self.dev.ADQ_StopDataAcquisition()

        if self.s.VERBOSE_MODE:
            self.gbps = self.bytes_received / (stop_time - start_time)
            self.meanFftTime = np.mean(self.fftTime)
            print('########Stats########')
            print(f'Total buffers received: {nof_buffers_received}')
            print(f'Total Time: {stop_time-start_time}')
            print(f'Time/buffer: {(stop_time-start_time)/self.s.NOF_BUFFERS_TO_RECEIVE}')
            print(f'Expected time/buffer (CH0): {s.CH0_RECORD_LEN/s.SAMPLE_RATE}')
            print(f"Total GB received: {self.bytes_received / 10**9}")
            print(f"Total GB/s: {self.gbps / 10**9}")
            print('number of FFTs:', self.numFft)
            print('mean fft time (ms) :', self.meanFftTime)

            self.data_buffer = np.zeros(
                self.parameters.transfer.channel[0].record_buffer_size // 2, dtype=np.short
            )
            print(self.gpu_buffers.buffers[0][0], '\n')
            hc.cudaMemcpy(
                self.data_buffer.ctypes.data,                                #destantation
                self.gpu_buffers.buffers[self.s.chToTest][1].data.ptr,                #source
                self.parameters.transfer.channel[1].record_buffer_size, #size
                hc.cudaMemcpyDeviceToHost,                              #kind
            )

            if self.s.saveBuffer:
                np.save('timeDomainBuffer', self.data_buffer)

            #plotting
            print(f"number of clips: {(self.data_buffer==2**15).sum() + (self.data_buffer==-2**15).sum()}")

            plt.close('all')
            if s.pltTimeDomain: #time domain
                plt.figure()
                plt.title('time domain')
                pts = [i for i in range(0, len(self.data_buffer))]
                plt.plot(pts, self.data_buffer)
                plt.xlabel('samples')
                plt.ylabel('ACD code')
                plt.show()

            if 0: #time domain diff
                plt.figure()
                diff = np.diff(np.asarray(pts))
                plt.plot(diff)
                plt.show()

            if 0: #bathtub
                plt.figure()
                plt.hist(self.data_buffer, bins = int(2**8))
                plt.show()

            if 0: #fft
                length = len(self.data_buffer)
                fft = np.abs(np.fft.fft(self.data_buffer)[0:length//2])

                plt.figure()
                plt.title('FFT')
                fft = fft**2 * 2**-34*2/(50*length**2)*1000
                plt.plot(np.linspace(0,1.25,length//2)[1:],10*np.log10(fft[1:]))
                plt.xlabel('Frequency (GHz)')
                plt.ylabel('Power (dBm)')
                plt.show()