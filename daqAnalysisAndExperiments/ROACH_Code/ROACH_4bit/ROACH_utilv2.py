from torch.utils.dlpack import to_dlpack
import matplotlib.pyplot as plt
import multiprocessing as mp
#mp.set_start_method('spawn', force=True)
from multiprocessing import shared_memory
import corr
import cupy as cp 
import logging
import numba as nb
import numpy as np 
import socket
import sys
import time
sys.path.append("../") 
import ROACH_funcsv2
import torch

def _conv_cuda(array):
	return 

def _pin_memory(array):
	mem = cp.cuda.alloc_pinned_memory(array.nbytes)
	
	ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
	ret[...] = array
	return ret

@nb.njit('void(uint8[::1],uint8[::1])', parallel=True)
def _nibbleize(A, AU):
	for i in nb.prange(AU.shape[0]):
			offset = i * 2
			A[offset] = (AU[i] >> 4) & 0xf
			A[offset + 1] = (AU[i]) & 0xf

class ROACH(mp.Process):
	def __init__(self, designFile = 'roach2_spec_k_2022_Dec_22_2055.bof',
					   IPROACH = '192.168.60.20',
					   IPPC = '192.168.60.1',
					   UDPPort = 60000,
					   GBEPort = 60000,
					   MACBase = (2<<40) + (2<<32),
					   MACTransmit = (0xf8<<40) + (0xf2<<32) + (0x1e<<24) + (0x7d << 16) + (0xc2 << 8) + (0xf5),
					   TXCoreName = 'gbe0',
					   FFTPow = int(26),
					   IPFPGA = '192.168.40.76',
					   recompile = True,
					   bufferLength = int(16),
					   clockRate = 2*1E9):


		self.designFile = designFile
		self.IPROACH = IPROACH
		self.IPPC = IPPC
		self.UDPPort = UDPPort
		self.GBEPort = GBEPort
		self.MACBase = MACBase
		self.MACTransmit = MACTransmit
		self.TXCoreName = TXCoreName
		self.IPFPGA = IPFPGA
		self.FFTPow = FFTPow
		self.recompile = recompile
		self.bufferSize = 32
		self.clockRate = clockRate
		self.writeFrequency = 0.25 # In minutes

		self.gbeMem = shared_memory.SharedMemory(create = True, size = self.bufferSize * 2**(self.FFTPow))
		self.fftMem = shared_memory.SharedMemory(create = True, size = 2 * self.bufferSize * 2**(self.FFTPow))
		self.writeMem = shared_memory.SharedMemory(create = True, size = (2 ** self.FFTPow + 2) * 4)

		self.totalAvgMem = shared_memory.SharedMemory(create = True, size = 2*8)	

		self.nibbleLockMem = shared_memory.SharedMemory(create = True, size = 2 * self.bufferSize)
		#self.nibbleLock = np.zeros((2, self.bufferSize), dtype = np.dtype('u1'), buffer = (self.nibbleLockMem).buf)
		self.fftLockMem = shared_memory.SharedMemory(create = True, size = 2 * self.bufferSize)
		self.fileLockMem = shared_memory.SharedMemory(create = True, size = 2)


		#self.fftLock = np.zeros((2, self.bufferSize), dtype = np.dtype('u1'), buffer = (self.fftLockMem).buf)


		# Memory for data coming from ADC
		self.sharedGBE = (np.ndarray((2, self.bufferSize, 2**(self.FFTPow - 1)), dtype = np.dtype('u1'), buffer = (self.gbeMem).buf))
		(self.sharedGBE).fill(0)
		
		# Memory for 'nibble-ized' data
		self.sharedFFT = (np.ndarray((2, self.bufferSize, 2**(self.FFTPow)), dtype = np.dtype('u1'), buffer = (self.fftMem).buf))
		(self.sharedFFT).fill(0)

		self.sharedWrite = (np.ndarray((2, 2**(self.FFTPow-1)+1), dtype = np.dtype('float32'), buffer = (self.writeMem).buf))
		(self.sharedWrite).fill(0)
		
		self.totalAvg = np.ndarray(2, dtype = np.dtype('int64'), buffer = (self.totalAvgMem).buf)
		(self.totalAvg).fill(0)

		self.nibbleLock = (np.ndarray((2, self.bufferSize), dtype = np.dtype('u1'), buffer = (self.nibbleLockMem).buf))
		(self.nibbleLock).fill(0)
		self.fftLock = (np.ndarray((2, self.bufferSize), dtype = np.dtype('u1'), buffer = (self.fftLockMem).buf))
		(self.fftLock).fill(0)
		self.fileLock = (np.ndarray(2, dtype = np.dtype('u1'), buffer = (self.fileLockMem).buf))
		(self.fileLock).fill(0)
		(self.fileLock)[1] = 1

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		(self.socket).setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		
		lh=corr.log_handlers.DebugLogHandler()
		print(self.IPFPGA)
		logger = logging.getLogger(self.IPFPGA)
		logger.addHandler(lh)
		logger.setLevel(10)
		self.fpga = corr.katcp_wrapper.FpgaClient(self.IPFPGA, logger=logger)
		time.sleep(1)
		if recompile:
			ROACH_funcsv2.progROACH(self.fpga, BOFFile = self.designFile)

	
	def gbe0(self):
		firstRun = True
		i = 0
		bufIndex = 0
		totalClears = 1
		clearOut = 0
		while True:
			try:
				while i < self.bufferSize:
					#if not(i):
					#	newStartTime = time.time()
					pos = 0
					#print((self.nibbleLock)[bufIndex])
					if not((self.fftLock)[bufIndex][i]): #and not((self.fileLock)[bufIndex]):
						#newStartTime = time.time()
						view = memoryview((self.sharedGBE)[bufIndex][i])
						#print(len((self.sharedGBE)[bufIndex][i]))
						view = view[pos:]
						if firstRun:
							(self.socket).bind((self.IPPC, self.GBEPort))
							(self.socket).setblocking(1)
							firstRun = False
						pos = 0
						newStartTime = time.time()
						while pos < 2 ** (self.FFTPow - 1):
							cr0 = (self.socket).recv_into(view, 8192)
							pos += cr0
							view = view[cr0:]
						print('TOOK ' + str(round((time.time() - newStartTime)*1000, 4)) + ' MILLISECONDS TO COLLECT DATA')	
						#print(((self.sharedGBE)[bufIndex][i][123456:124456]))
						i += 1 

					else:
						(self.socket).recv(8192)
						clearOut = clearOut + 1
						#if not(clearOut % 10000):
						print('\n\n\n\nCLEARING OUT THINGS ' + str(clearOut) + '\n\n\n\n')
				#print('TOOK ' + str(round((time.time() - newStartTime)*1000, 4)) + ' MILLISECONDS TO COLLECT DATA')	
				for counter in range(self.bufferSize):
					#print('SETTING THEM NIBBLES ' + ' AT ' + str(bufIndex) + ' TO ' + str(counter))
					(self.nibbleLock)[bufIndex][counter] = 1
				i = 0
				if bufIndex == 0:
					bufIndex = 1
				else:
					bufIndex = 0
				#testTime = time.time()
				#print('\n\nFINISHED FILLING BUFFER FOR ` ~ 1/2 TOOK ABOUT ' + str(round((time.time() - newStartTime)/2, 3)) + 'S \n\n')
				#print('GBE0 TOOK ' + str(round(testTime - newStartTime, 3)) + ' SECONDS TO READ ' + str((ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT/2*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT) + ' BYTES OF DATA')
				#print('IN THIS TIME '+ str(((testTime - newStartTime) * (ROACH_SETUP.PAYLOAD_LEN + 4)* ROACH_SETUP.CLOCK_SPEED)) + ' BYTES WERE TRANSMITTED') 
				#print('THE EFFICIENCY IS: ' + str(round(((ROACH_SETUP.PAYLOAD_LEN + 4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT/2) * 100/ ((testTime - newStartTime) * ROACH_SETUP.CLOCK_SPEED), 3)) + '%\n\n')

			except KeyboardInterrupt as e:
				print('KEYBOARD INTERRUPT')
				sys.exit(0)
			except Exception as e:
				print(e)
				sys.exit(1)

	def nibbleize(self):
		bufIndex = 0
		currentBuf = 0
		while True:
			while currentBuf < self.bufferSize:
				#print('FUCK THIS ' + ' AT ' + str(bufIndex) + ' INDEX ' + str(currentBuf) + ': ' +  str((self.nibbleLock)[bufIndex][currentBuf]))
				#plt.plot((self.sharedGBE)[bufIndex][currentBuf])
				#plt.show()
				if ((self.nibbleLock)[bufIndex][currentBuf]) and not((self.fftLock)[bufIndex][currentBuf]):
					currentTime = time.time()
					_nibbleize((self.sharedFFT)[bufIndex][currentBuf], (self.sharedGBE)[bufIndex][currentBuf])
					(self.nibbleLock)[bufIndex][currentBuf] = 0
					(self.fftLock)[bufIndex][currentBuf] = 1
					print('TOOK ' + str(round((time.time() - currentTime)*1000, 4)) + ' MILLISECONDS TO NIBBLEIZE DATA')	
					currentBuf += 1
			if bufIndex == 0:
				bufIndex = 1
			else:
				bufIndex = 0       
			currentBuf = 0

	def performFFT(self):
		#Taken from https://github.com/cupy/cupy/blob/master/examples/stream/cupy_memcpy.py
		#x_cpu_src = np.zeros(2**self.FFTPow, np.dtype('u1'))
		#x_pinned_cpu_src = _pin_memory(x_cpu_src)
		#x_pinned_cpu_dst = _pin_memory(self.sharedWrite)
		# gpuData = cp.zeros(int(2**(self.FFTPow - 1)+1), dtype = cp.float32)
		#copyStream = cp.cuda.Stream(non_blocking = False)
		#copyData = cp.empty_like((self.sharedFFT)[0])
		saveFreq = np.ceil((self.writeFrequency)*60 / (2**(self.FFTPow + 1)/(self.clockRate)*self.bufferSize))

		#sharedFFTMem = _pin_memory(self.sharedFFT)
		copyStream = cp.cuda.Stream(non_blocking = False)
		copyData = cp.empty_like((self.sharedFFT)[0][0], dtype = cp.uint8)

		sharedWriteMem = _pin_memory(self.sharedWrite)
		gpuToCPUStream = cp.cuda.Stream(non_blocking = False)

		currentBuf = 0
		bufIndex = 0
		while True:
			try:
				#with cp.cuda.stream.Stream() as stream_dtoh:
				totalTime = 0
				while currentBuf < self.bufferSize:

					if ((self.fftLock)[bufIndex][currentBuf]) and not((self.nibbleLock)[bufIndex][currentBuf]):
						#ding = cp.array((self.sharedFFT)[bufIndex][currentBuf])
						

						#print('SPOON')
						#dong = ding.astype(cp.float32)
						#print(torch.as_tensor(np.asarray(range(2**10)), device = torch.device('cuda')))
						currentTime = time.time() 
						#testData = torch.as_tensor(((self.sharedFFT)[bufIndex][currentBuf]), device='cuda')
						#testData = torch.from_numpy((self.sharedFFT)[bufIndex][currentBuf]).to('cuda')
						#testData = torch.cuda.ByteTensor((self.sharedFFT)[bufIndex][currentBuf])
						#if (self.totalAvg)[0] > 2:
						#plt.plot((self.sharedFFT)[bufIndex][currentBuf])
						#plt.show()
						
						#plt.show()
						#diff = ((self.sharedFFT)[bufIndex][currentBuf][1:]).astype('int8') - ((self.sharedFFT)[bufIndex][currentBuf][:-1]).astype('int8') 
						#diffVals = np.where((self.sharedFFT)[bufIndex][currentBuf] == 8)[0]
						#diff = diffVals[1:] - diffVals[:-1]
						#hist, bin_edges = np.histogram(diff, bins = [0, 1, 2, 4256, 4257, 4258, 9000])
						#print(len(diff))
						#print(hist)
						#plt.plot((self.sharedGBE)[bufIndex][currentBuf])
						#plt.plot(diff, 'bo')
						#plt.show()
						#print(min((self.sharedFFT)[bufIndex][currentBuf]))
						copyData.set((self.sharedFFT)[bufIndex][currentBuf])
						#copyData.set((sharedFFTMem)[bufIndex][currentBuf], stream = copyStream)

						#copyStream.synchronize()
						#plt.plot((self.sharedGBE)[bufIndex][currentBuf])
						#plt.show()
						#x = np.asarray([counter for counter, x in enumerate(((self.sharedFFT)[bufIndex][currentBuf] - 7)) if x == 1])
						#diff = x[1:] - x[:-1] 
						#print(diff[:1000])

						#hist, bin_edges = np.histogram((self.sharedFFT)[bufIndex][currentBuf], bins = list(range(16)))
						#print(hist)

						#plt.show()
						testData = torch.as_tensor((copyData.astype('int8') - np.int8(7)) , device = 'cuda')
						print('TOOK ' + str(round((time.time() - currentTime)*1000, 4)) + ' MILLISECONDS TO PROCESS DATA')

						#testData = _conv_cuda((self.sharedFFT)[bufIndex][currentBuf])
						#torch.as_tensor(array.astype(cp.float32), device='cuda')

						if not(currentBuf):
							(gpuData) = cp.square((torch.abs(((torch.fft.rfft(testData, n = 2**(self.FFTPow)))))))
							#gpuData *= 0.940/(2**4)**2/(self.bufferSize)*(2/2**(self.FFTPow*2))*1000/50							
							#plt.plot(10*np.log10(gpuData.get()))
							#plt.show()
						else:
							(gpuData) += cp.square((torch.abs(((torch.fft.rfft(testData, n = 2**(self.FFTPow)))))))

						if currentBuf < (self.bufferSize - 1):
							(self.fftLock)[bufIndex][currentBuf] = 0
						currentBuf += 1
					else:
						pass
					#with cp.cuda.stream.Stream() as stream_htod:

				while (self.totalAvg)[0] < saveFreq and not((self.fileLock)[1]):
					print('WAITING UNTIL DONE COPYING FILE DATA')
					#time.sleep(1)
				
				currentTime = time.time()
				
				while not(self.fileLock[1]):
					print('CANT WRITE WAITING TO FINISH COPYING DATA IN WRITE')	
					
				#(self.sharedWrite)[bufIndex] = (self.sharedWrite)[bufIndex] + gpuData.get(stream = gpuToCPUStream, out = (self.sharedWrite)[bufIndex])
				#(self.sharedWrite)[bufIndex] = gpuData.get(stream = gpuToCPUStream, out = (self.sharedWrite)[bufIndex])

				#gpuToCPUStream.synchronize()
				(self.sharedWrite)[bufIndex] += gpuData.get()
				(self.totalAvg)[bufIndex] += 1

				print('TOOK ' + str(round((time.time() - currentTime)*1000, 4)) + ' ms TO COPY')
				if (self.totalAvg)[0] % saveFreq == 0 and (self.totalAvg)[0] == (self.totalAvg)[1]:
					while (self.fileLock[0]):
						print('CANT WRITE WAITING TO FINISH WRITE PROCESS')
					(self.fileLock[0]) = 1
					(self.fftLock)[bufIndex][currentBuf - 1] = 0
				else:
					(self.fftLock)[bufIndex][currentBuf - 1] = 0

					#stream_htod.synchronize()
					#stream_dtoh.synchronize()
				currentBuf = 0
				if bufIndex == 0:
					bufIndex = 1
				else:
					bufIndex = 0
			except KeyboardInterrupt as e:
				print('KEYBOARD INTERRUPT')
				sys.exit(0)
			except Exception as e:
				print(e)
				sys.exit(1)	

	
	def writeFile(self):
		currentBuf = 0
		fileIndex = 1
		acqCount = 0
		freqs = np.asarray(range(2**(self.FFTPow - 1) + 1))*self.clockRate / 2**(self.FFTPow)
		while True:
			try:
				if (self.fileLock)[0]:
					(self.fileLock)[1] = 0
					writeData = np.array(self.sharedWrite, copy = True)
					acqCount += 1
					(self.fileLock)[1] = 1
					writeData[0] = (0.940/2**4)**2/(acqCount*self.bufferSize)*(2/2**(self.FFTPow*2))*1000/50*writeData[0]
					writeData[1] = (0.940/2**4)**2/(acqCount*self.bufferSize)*(2/2**(self.FFTPow*2))*1000/50*writeData[1]
					print('\n\n\n\n\n\n')
					print((self.totalAvg))
					print('\n\n\n\n\n\n')
					#plt.title(str((self.totalAvg)[0]) + ' AVERAGES')
					plt.title(acqCount*self.bufferSize)
					plt.plot((freqs/10**6)[1:], (10*np.log10(writeData[0]))[1:], alpha = 0.7)
					plt.plot((freqs/10**6)[1:], (10*np.log10(writeData[1]))[1:], alpha = 0.7)

					plt.show()
					(self.fileLock)[0] = 0
			except KeyboardInterrupt as e:
				print('KEYBOARD INTERRUPT')
				sys.exit(0)
			except Exception as e:
				print(e)
				sys.exit(1)	

	def startRun(self):
		eth1 = mp.Process(target = self.gbe0, name = 'gbe0', args = ())
		nibbleProc = mp.Process(target = self.nibbleize, name = 'nibbleize', args = ())
		fftProc = mp.Process(target = self.performFFT, name = 'performFFT', args = ())
		writeProc = mp.Process(target = self.writeFile, name = 'writeFile', args = ())

		ROACH_funcsv2.enableROACH((self.fpga))
		
		print('SLEEPING FOR A BIT...')
		time.sleep(3)
		print('DONE SLEEPING')			
		eth1.start()
		nibbleProc.start()
		fftProc.start()
		writeProc.start()
		#time.sleep(3)
		#eth1.join() 
		#nibbleProc.join()
		#fftProc.join()

		#(self.gbeMem).unlink()
		#(self.gbeMem).close()

		#(self.fftMem).unlink()
		#(self.fftMem).close()

		#(self.writeMem).unlink()
		#(self.writeMem).close()

		#(self.nibbleLockMem).unlink()
		#(self.nibbleLockMem).close()

		#(self.fftLockMem).unlink()
		#(self.fftLockMem).close()