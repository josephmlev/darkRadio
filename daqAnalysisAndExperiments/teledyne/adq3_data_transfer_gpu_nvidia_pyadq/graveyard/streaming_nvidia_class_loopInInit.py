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
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)


gdrapi = g.GdrApi()

class avgFft:
    def __init__(self):
        """Main streaming function"""
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
        self.dev: pyadq.ADQ = acu.SetupDevice(device_to_open)

        print(f"Setting up data collection for: {self.dev}")

        # Initialize the parameterss with default values
        self.parameters: pyadq.ADQParameters = self.dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)
        self.parameters.event_source.periodic.period = s.PERIODIC_EVENT_SOURCE_PERIOD
        self.parameters.event_source.periodic.frequency = s.PERIODIC_EVENT_SOURCE_FREQUENCY

        self.parameters.test_pattern.channel[0].source = s.CH0_TEST_PATTERN_SOURCE
        self.parameters.test_pattern.channel[1].source = s.CH1_TEST_PATTERN_SOURCE

        self.parameters.signal_processing.sample_skip.channel[0].skip_factor = s.CH0_SAMPLE_SKIP_FACTOR
        self.parameters.signal_processing.sample_skip.channel[1].skip_factor = s.CH1_SAMPLE_SKIP_FACTOR
        self.parameters.acquisition.channel[0].nof_records = (
            s.NOF_RECORDS_PER_BUFFER * s.NOF_BUFFERS_TO_RECEIVE
        )
        self.parameters.acquisition.channel[0].record_length = s.CH0_RECORD_LEN
        self.parameters.acquisition.channel[0].trigger_source = s.CH0_TRIGGER_SOURCE
        self.parameters.acquisition.channel[0].trigger_edge = s.CH0_TRIGGER_EDGE
        self.parameters.acquisition.channel[0].horizontal_offset = s.CH0_HORIZONTAL_OFFSET

        if s.NOF_CHANNELS > 1:
            self.parameters.acquisition.channel[1].nof_records = (
                s.NOF_RECORDS_PER_BUFFER * s.NOF_BUFFERS_TO_RECEIVE
            )
            self.parameters.acquisition.channel[1].record_length = s.CH1_RECORD_LEN
            self.parameters.acquisition.channel[1].trigger_source = s.CH1_TRIGGER_SOURCE
            self.parameters.acquisition.channel[1].trigger_edge = s.CH1_TRIGGER_EDGE
            self.parameters.acquisition.channel[1].horizontal_offset = s.CH1_HORIZONTAL_OFFSET

        self.parameters.transfer.common.write_lock_enabled = 1
        self.parameters.transfer.common.transfer_records_to_host_enabled = 0
        self.parameters.transfer.common.marker_mode = pyadq.ADQ_MARKER_MODE_HOST_MANUAL

        self.parameters.transfer.channel[0].record_length_infinite_enabled = 0
        self.parameters.transfer.channel[0].record_size = (
            s.BYTES_PER_SAMPLES * self.parameters.acquisition.channel[0].record_length
        )
        self.parameters.transfer.channel[0].record_buffer_size = (
            s.NOF_RECORDS_PER_BUFFER * self.parameters.transfer.channel[0].record_size
        )
        self.parameters.transfer.channel[0].metadata_enabled = 0
        self.parameters.transfer.channel[0].nof_buffers = s.NOF_GPU_BUFFERS

        if s.NOF_CHANNELS > 1:
            self.parameters.transfer.channel[1].record_length_infinite_enabled = 0
            self.parameters.transfer.channel[1].record_size = (
                s.BYTES_PER_SAMPLES * self.parameters.acquisition.channel[1].record_length
            )
            self.parameters.transfer.channel[1].record_buffer_size = (
                s.NOF_RECORDS_PER_BUFFER * self.parameters.transfer.channel[1].record_size
            )
            self.parameters.transfer.channel[1].metadata_enabled = 0
            self.parameters.transfer.channel[1].nof_buffers = s.NOF_GPU_BUFFERS

        # Create pointers, buffers and GDR object
        self.memory_handles = [
            [g.GdrMemoryHandle() for x in range(s.NOF_CHANNELS)] for y in range(s.NOF_GPU_BUFFERS)
        ]
        self.bar_ptr_data = sh.BarPtrData(s.NOF_CHANNELS, s.NOF_GPU_BUFFERS)
        #print('bar', self.bar_ptr_data.pointers)
        self.gpu_buffer_ptr = sh.GpuBufferPointers(s.NOF_CHANNELS, s.NOF_GPU_BUFFERS)
        self.gdr = gdrapi.gdr_open()
        self.gpu_buffers = sh.GpuBuffers(
            s.NOF_CHANNELS,
            s.NOF_GPU_BUFFERS,
            s.CH0_RECORD_LEN,
        )
        self.gpu_buffer_address = 0

        # Allocate GPU buffers
        for ch in range(s.NOF_CHANNELS):
            for b in range(s.NOF_GPU_BUFFERS):
                #print(f"allocating ch {ch}, buffer num {b}")
                (
                    self.gpu_buffer_ptr.pointers[ch][b],
                    self.gpu_buffer_address,
                    self.gpu_buffers.buffers[ch][b],
                ) = self.allocate_and_pin_buffer(
                    self.parameters.transfer.channel[ch].record_buffer_size,
                    self.memory_handles[ch][b],
                    self.gdr,
                    self.bar_ptr_data.pointers[ch][b],
                )
                self.parameters.transfer.channel[ch].record_buffer_bus_address[b] = self.gpu_buffer_address
                self.parameters.transfer.channel[ch].record_buffer[b] = self.gpu_buffer_ptr.pointers[ch][b]
        # Configure digitizer parameterss
        self.dev.SetParameters(self.parameters)
        print('done allocating and configuring \n')

        # Start timer for measurement
        start_time = time.time()
        # Start timer for regular printouts
        start_print_time = time.time()
        print(f"Start acquiring data for: {self.dev}")
        if self.dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
            print("Success")
        else:
            print("Failed")

        data_transfer_done = 0
        nof_buffers_received = [0, 0]
        bytes_received = 0
        status = pyadq.ADQP2pStatus()._to_ct()
        print('status done')
        #main transfer loop
        while not data_transfer_done:
            self.result = self.dev.ADQ_WaitForP2pBuffers(ct.byref(status), s.WAIT_TIMEOUT_MS)
            print("result done")
            if self.result == pyadq.ADQ_EAGAIN:
                print("Timeout")
            elif self.result < 0:
                print(f"Failed with retcode {self.result}")
                exit(1)

            else:
                buf = 0
                while (buf < status.channel[0].nof_completed_buffers) or (
                    buf < status.channel[1].nof_completed_buffers
                ):
                    for ch in range(s.NOF_CHANNELS):
                        if buf < status.channel[ch].nof_completed_buffers:
                            buffer_index = status.channel[ch].completed_buffers[buf]
                            self.dev.ADQ_UnlockP2pBuffers(ch, (1 << buffer_index)) # A mask of buffer indexes to unlock. p194
                            nof_buffers_received[ch] += 1
                            bytes_received += (
                                s.NOF_RECORDS_PER_BUFFER * s.CH0_RECORD_LEN * s.BYTES_PER_SAMPLES
                            )
                    buf += 1
                data_transfer_done = nof_buffers_received[1] >= s.NOF_BUFFERS_TO_RECEIVE
                now_time = time.time() - start_time
                print_time = time.time() - start_print_time
                if print_time > 5:
                    # Check for overflow, stop if overflow
                    overflow_status = self.dev.GetStatus(pyadq.ADQ_STATUS_ID_OVERFLOW)
                    if overflow_status.overflow:
                        print("Overflow, stopping data acquisition...")
                        self.dev.ADQ_StopDataAcquisition()
                        exit("Exited because of overflow.")

                    print("Nof buffers received:", nof_buffers_received)
                    print("Total GB received:", bytes_received / 10**9)
                    print("Average transfer speed:", bytes_received / 10**9 / now_time)
                    start_print_time = time.time()

        stop_time = time.time()

        gbps = bytes_received / (stop_time - start_time)
        self.dev.ADQ_StopDataAcquisition()
        print(f"Total GB received: {bytes_received / 10**9}")

        print(f"Total GB/s: {gbps / 10**9}")

    def exit(self):
        for ch in range(s.NOF_CHANNELS):
            for b in range(s.NOF_GPU_BUFFERS):
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
                    s.NOF_RECORDS_PER_BUFFER * s.CH0_RECORD_LEN * s.BYTES_PER_SAMPLES,
                )
                gdrapi.gdr_unpin_buffer(self.gdr, self.memory_handles[ch][b])
                #print(self.bar_ptr_data.pointers[ch][b], '\n')
        # Free GPU memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

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

    def printAndPlot(self):
        if s.PRINT_LAST_BUFFERS:
            data_buffer = np.zeros(
                self.parameters.transfer.channel[0].record_buffer_size // 2, dtype=np.short
            )
            print(self.gpu_buffers.buffers[0][0])
            hc.cudaMemcpy(
                data_buffer.ctypes.data,                                #destantation
                self.gpu_buffers.buffers[1][1].data.ptr,                #source
                self.parameters.transfer.channel[1].record_buffer_size, #size
                hc.cudaMemcpyDeviceToHost,                              #kind
            )

            data_buffer.tofile("data.bin")
            plt.close('all')
            if 1: #time domain
                plt.figure()
                plt.title('time domain')
                pts = [i for i in range(0, len(data_buffer))]
                plt.plot(pts, data_buffer)
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
                plt.hist(data_buffer, bins = int(2**8))
                plt.show()

            if 0: #fft
                length = len(data_buffer)
                fft = np.abs(np.fft.fft(data_buffer)[0:length//2])

                plt.figure()
                plt.title('FFT')
                fft = fft**2 * 2**-34*2/(50*length**2)*1000
                plt.plot(np.linspace(0,1.25,length//2)[1:],10*np.log10(fft[1:]))
                plt.xlabel('Frequency (GHz)')
                plt.ylabel('Power (dBm)')
                plt.show()
        exit()

    def checkDev(self):
        print('made it to checkDev method')
        if self.dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
            print("Success")
        else:
            print('Fail')
    def doFFT(self):
        scipy.fft.rfft(self.gpu_buffers[ch][b])




if __name__ == "__main__":
    
    myclass = avgFft()
    myclass.exit()
    myclass.printAndPlot()
    #myclass.checkDev()