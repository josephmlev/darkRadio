#import
#define functions
# Create pointers, buffers and GDR object
# Allocate GPU buffers
# Configure digitizer parameters

class avgFft:
    def exit():
        for ch in range(s.NOF_CHANNELS):
            for b in range(s.NOF_GPU_BUFFERS):
                gdrapi.gdr_unmap(
                    self.gdr,
                    self.memory_handles[ch][b],
                    self.bar_ptr_data.pointers[ch][b],
                    s.NOF_RECORDS_PER_BUFFER * s.CH0_RECORD_LEN * s.BYTES_PER_SAMPLES,
                )
                gdrapi.gdr_unpin_buffer(self.gdr, self.memory_handles[ch][b])
        # Free GPU memory

    def acquireData(self, sem):
        # Start timer for measurement
        start_time = time.time()
        # Start timer for regular printouts
        start_print_time = time.time()
        print(f"Start acquiring data for: {self.dev}")
        if self.dev.ADQ_StartDataAcquisition() == pyadq.ADQ_EOK: #check for errors starting
            print("Success. Begin Acquiring")
        else:
            print("Failed")

        #main transfer loop. Happens NOF_BUFFERS_TO_RECEIVE times
        while not self.data_transfer_done:
            #wait for buffers to fill. Code locks here until one transfer buffer fills 
            self.result = self.dev.ADQ_WaitForP2pBuffers(ct.byref(self.status), s.WAIT_TIMEOUT_MS)
            #handle errors
            if errors:#deal with them 
            else: #We have a full buffer,
                sem.release()
                while (buffers to fill):
                    for ch in range(s.NOF_CHANNELS):
                        if buf < self.status.channel[ch].nof_completed_buffers:
                            self.buffer_index = self.status.channel[ch].completed_buffers[buf]

                            #I think unlockp2pbuffers (along with this loop
                            #needs to go in the FFT thread, but unlockp2pbuffers is
                            #not thread safe. This is incorrect, but sort of works
                            #because the FFT is so fast. Buggy though
                            self.dev.ADQ_UnlockP2pBuffers(ch, (1 << self.buffer_index))

                self.data_transfer_done = self.nof_buffers_received[1] >= s.NOF_BUFFERS_TO_RECEIVE

        print("Done Acquiring Data \n")

    def doFFT(self, sem):
        fftCompleted = 0
        
        #init tensor of zeros
        self.fft = torch.as_tensor(cp.zeros(s.CH0_RECORD_LEN//2 + 1), device='cuda')

        while not self.data_transfer_done:
            #acquire the semaphore once acquireData thread has released it
            sem.acquire()
            bufferTensor    = torch.as_tensor(self.gpu_buffers.buffers[1][readyBuffer], device='cuda')
            self.fft        +=torch.abs(torch.fft.rfft(bufferTensor))
            #should unlock p2pbuffer here once the FFT stream completes 
            fftCompleted+=1

if __name__ == "__main__":
    myclass = avgFft()
    sem             = threading.Semaphore(0)
    acquireThread   = threading.Thread(target=myclass.acquireData,
                                    args=(sem,))
    doFFTThread     = threading.Thread(target=myclass.doFFT,
                                    args=(sem,))
    acquireThread.start()
    doFFTThread.start()
    acquireThread.join() # Wait for acquireThread to complete
    doFFTThread.join()
    sumFft = cp.asarray(myclass.fft).get() #convert from torch tensor to cp.array and get to cpu
    myclass.exit()
