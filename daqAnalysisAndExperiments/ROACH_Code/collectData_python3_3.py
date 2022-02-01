import numpy as np 
from datetime import datetime
from numpy import loadtxt
import ctypes
import cupy as cp
import cupyx as cpx
import h5py
import numba as nb
import ROACH_SETUP
import ROACH_funcs
import time
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import os
import ScpiInstrumentWrapper
import queue
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt
import serial
import signal
import socket
import struct
import subprocess
import sys 
import timeit
import pyvisa
import usb.core
import usb.util 
#!/usr/bin/env python

import corr, time, struct, sys, logging, select, socket, numpy, threading
import ROACH_SETUP

def exit_fail():
	print('FAILURE DETECTED. Log entries:\n',lh.printMessages())
	print('Transmit buffer overflow status: %i' %fpga.read_int('tx_over_status_gbe0'))
#    try:
#        fpga.stop()
#    except: pass
#    raise
	exit()

def exit_clean():
	try:
		for f in fpgas: f.stop()
	except: pass
	exit()

def convertIPToNum(addr):
	vals = addr.split('.')
	numberVal = int(vals[0])*(2**24) + int(vals[1])*(2**16) + int(vals[2])*(2**8) + int(vals[3])
	return numberVal

def plotData(socketName0, socketName1, payloadLength):
	ready0 = select.select([socketName0], [], [], 10)
	ready1 = select.select([socketName1], [], [], 10)
	totalIts = 100000
	readBytes = payloadLength * 8
	good0 = 0
	good1 = 0
	#fig = plt.gcf()
	#fig.show()
	#fig.canvas.draw()
	if ready0[0] and ready1[0]:
		startTime = time.time()
		for acqCount in range(totalIts):
			#startTime = time.time()
			#print 'ITERATION ' + str(blah)
			#sys.exit(1)
			try:
				#data = struct.unpack('>500Q', socketName0.recvfrom(readBytes)[0]) 
				socketName0.recvfrom(readBytes)
				good0 = good0 + 1
				socketName1.recvfrom(readBytes)
				good1 = good1 + 1
				#print 'HUH?'
				#time.sleep(1)
				#print 'IS IT ALMOST FULL ON GBE1?' + str(fpga.read_int('tx_afull_status_gbe1'))
				#print 'ON ITERATION ' + str(acqCount)
				#acqCount = acqCount + 1
				# dataNew = struct.unpack('>' + str(readBytes/8) + 'Q', data[0])
				# #print 'TOTAL SAMPLES: ' + str(len(dataNew)*8)
				# mask1 = 0xff 
				# mask2 = 0xff00
				# mask3 = 0xff0000
				# mask4 = 0xff000000
				# mask5 = 0xff00000000
				# mask6 = 0xff0000000000
				# mask7 = 0xff000000000000
				# mask8 = 0xff00000000000000
				# newData = [[] for x in range(8)]
				# plotData = []
				# for val in dataNew:
				#     newData[0].append((int((val & mask8)>>56) - 128.)/256.)
				#     newData[1].append((int((val & mask7)>>48) - 128.)/256.)
				#     newData[2].append((int((val & mask6)>>40) - 128.)/256.)
				#     newData[3].append((int((val & mask5)>>32) - 128.)/256.)
				#     newData[4].append((int((val & mask4)>>24) - 128.)/256.)
				#     newData[5].append((int((val & mask3)>>16) - 128.)/256.)
				#     newData[6].append((int((val & mask2)>>8) - 128.)/256.)
				#     newData[7].append((int((val & mask1)) - 128.)/256)
				#     plotData.append(newData[0][-1])
				#     plotData.append(newData[1][-1])
				#     plotData.append(newData[2][-1])
				#     plotData.append(newData[3][-1])
				#     plotData.append(newData[4][-1])
				#     plotData.append(newData[5][-1])
				#     plotData.append(newData[6][-1])
				#     plotData.append(newData[7][-1])
			except Exception as e:
				pass
				#print e
		
#sigGen = pyvisa.ResourceManager().open_resource('USB0::1689::836::C020120::0::INSTR')
#sigGen = ScpiInstrumentWrapper.ScpiInstrumentWrapper('PROLOGIX::/dev/ttyUSB4::GPIB::10')
#sigGen.write('CF 123.45 MZ')
#sigGen.write('LV -124 DB')
#sigGen.write('VOLTAGE:AMPLITUDE -20DBM')
#arduino =  serial.Serial(port='/dev/ttyACM0', baudrate = 115200, timeout=.1)
fpga = subprocess.call(['python2', './progROACH.py', '192.168.40.76', '--arp'])
#3ROACH_funcs.progROACH('192.168.40.76')
idVendor = 0x16c0
idProduct = 0x05df
#dev = usb.core.find(idVendor = idVendor, idProduct = idProduct)
#print(cfg)
#intf = cfg[(0,0)]
#endPoint = intf[0]
#if dev.is_kernel_driver_active(0):
#    try:
#        dev.detach_kernel_driver(0)
#        print("kernel driver detached")
#    except usb.core.USBError as e:
#        sys.exit("Could not detach kernel driver: %s" % str(e))
# requestType = usb.util.build_request_type(usb.util.CTRL_OUT, usb.util.CTRL_TYPE_CLASS, usb.util.CTRL_RECIPIENT_INTERFACE)

@nb.njit(fastmath=True,parallel=True,cache=True)
def nb_einsum(A):
	#check the input's at the beginning
	#I assume that the asserted shapes are always constant
	#This makes it easier for the compiler to optimize 
	assert A.shape[1]==ROACH_SETUP.NUM_FFT

    #allocate output
	res=np.empty(A.shape[0],dtype=np.int32)

	for s in nb.prange(A.shape[0]):
		acc = 0.
		for val in np.nditer(A[s]):
			acc += val.item()
		res[s] = acc
        #Using a syntax like that is also important for performance
		#acc=0
	return res

def write_read(x):
	arduino.write(bytes(x, 'utf-8'))

def _pin_memory(array):
	mem = cp.cuda.alloc_pinned_memory(array.nbytes)
	ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
	ret[...] = array
	return array

sock0 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock0.setblocking(1)

sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.setblocking(1)

def gbe0(memName, writeName, startName):
	startTime = time.time()
	toggle = False
	existing_shm = [shared_memory.SharedMemory(name=memName[i][0].name) for i in range(len(memName))]
	sharedData =  [np.ndarray((ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN+4)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('B'), buffer=existing_shm[i].buf) for i in range(len(existing_shm))] 
	write_mem = [shared_memory.SharedMemory(name=writeName[i].name) for i in range(len(writeName))]
	writeStatus = [np.ndarray((ROACH_SETUP.NUM_FFT, 2), dtype=np.dtype('B'), buffer=write_mem[i].buf) for i in range(len(write_mem))]
	start_mem = shared_memory.SharedMemory(name=startName.name)
	startArr = np.ndarray((1,), dtype=np.dtype('B'), buffer=start_mem.buf)

	firstRun = True
	sock0.setblocking(1)
	i = 0
	totalClears = 1
	while time.time() - startTime < ROACH_SETUP.TOTAL_TIME:
		newStartTime = time.time()
		
		try:
			while i < ROACH_SETUP.NUM_BUFFERS:
				j = 0				
				while j < ROACH_SETUP.NUM_FFT:

					pos = 0
					if startArr[0] and ((not(writeStatus[i][j][0]))):
						view = memoryview(sharedData[i][j])
						view = view[pos:]
						if firstRun:
							sock0.bind((ROACH_SETUP.UDP0_IP, ROACH_SETUP.GBE0_PORT))
							sock0.setblocking(1)
							firstRun = False
						if not(firstRun):
							pos = 0
							timeA = time.time()
							initTime = timeA
							holdTime = timeA
							#sock0.setblocking(0)
							clearPacket = 0
							#currentClears = 0
							while 1:
								try:
									#if holdTime - initTime < 0.2:
									#	sock0.recv(8192)
									#	clearPacket += 1
									#	holdTime = time.time()
									#	if holdTime - timeA > 0.005:
									#		timeA = holdTime
									#		time.sleep(0.01)
									#else:
									blah = sock0.recv(8192)
									#clearPacket += 1
									time.sleep(0.1)	
									if clearPacket == 8201:
										time.sleep(0.04)
										break
									#elif clearPacket == 8192*totalClears:
									#	break
								except:
									print(clearPacket)
									print('FUCK 0')
									#time.sleep(0.05)
									#if currentClears == totalClears:
										#print(clearPacket)
									#	currentClears = 0
									#	break
									#else:
									#	currentClears += 1

							print('TOOK ' + str(time.time() - initTime) + 'S TO CLEAR BUFFER 0')
							sock0.setblocking(1)
						pos = 0
						view = memoryview(sharedData[i][j])
						view = view[pos:]
						while pos < int((ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.EXTRA_MULT):
							#cr0 = sock0.recv_into(memoryview(sharedData[i][j])[pos:])
							cr0 = sock0.recv_into(view, 8192)
							pos += 2*cr0
							view = view[2*cr0:]
						#plt.plot(sharedData[i][j][0:8192])
						#plt.show()
						writeStatus[i][j][0] = 1
					j += 1
				i += 1 
			i = 0
			testTime = time.time()
			#print('\n\nFINISHED FILLING BUFFER FOR GBE0 ~ 1/2 TOOK ABOUT ' + str(round((time.time() - newStartTime)/2, 3)) + 'S \n\n')
			#print('GBE0 TOOK ' + str(round(testTime - newStartTime, 3)) + ' SECONDS TO READ ' + str((ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT/2*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT) + ' BYTES OF DATA')
			#print('IN THIS TIME '+ str(((testTime - newStartTime) * (ROACH_SETUP.PAYLOAD_LEN + 4)* ROACH_SETUP.CLOCK_SPEED)) + ' BYTES WERE TRANSMITTED') 
			#print('THE EFFICIENCY IS: ' + str(round(((ROACH_SETUP.PAYLOAD_LEN + 4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT/2) * 100/ ((testTime - newStartTime) * ROACH_SETUP.CLOCK_SPEED), 3)) + '%\n\n')
			#if endThings:
			#	print('GBE0 HANGING SO KILLED')
		except Exception as e:
			print(e)
			sys.exit(1)

def gbe1(memName, writeName, startName):
	startTime = time.time()
	toggle = False
	existing_shm = [shared_memory.SharedMemory(name=memName[i][0].name) for i in range(len(memName))]
	sharedData =  [np.ndarray((ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN+4)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('B'), buffer=existing_shm[i].buf) for i in range(len(existing_shm))] 
	#testArr = np.ndarray((8192,), dtype = np.dtype('B'))
	#testArrMem = memoryview(testArr)
	write_mem = [shared_memory.SharedMemory(name=writeName[i].name) for i in range(len(writeName))]
	writeStatus = [np.ndarray((ROACH_SETUP.NUM_FFT, 2), dtype=np.dtype('B'), buffer=write_mem[i].buf) for i in range(len(write_mem))]
	
	start_mem = shared_memory.SharedMemory(name=startName.name)
	startArr = np.ndarray((1,), dtype=np.dtype('B'), buffer=start_mem.buf)

	firstRun = True
	sock1.setblocking(1)
	i = 0
	totalClears = 1
	while time.time() - startTime < ROACH_SETUP.TOTAL_TIME:
		newStartTime = time.time()
		try:
			while i < ROACH_SETUP.NUM_BUFFERS:
				j = 0
				while j < ROACH_SETUP.NUM_FFT:
					pos = 8192
					if startArr[0] and ((not(writeStatus[i][j][1]))):
						view = memoryview(sharedData[i][j])
						view = view[pos:]
						if firstRun:
							sock1.bind((ROACH_SETUP.UDP1_IP, ROACH_SETUP.GBE1_PORT))
							sock1.setblocking(1)
							#[sock1.recv_into(view, 8192) for x in range(1000)]
							firstRun = False
						if not(firstRun):
							pos = 8192
							timeA = time.time()
							initTime = timeA
							holdTime = timeA
							sock1.setblocking(1)
							#k = 0
							clearPacket = 0
							#currentClears = 0
							while 1:
								try:
									#if holdTime - initTime < 0.2:
										#sock1.recv(8192)
										#holdTime = time.time()
										#if holdTime - timeA > 0.01:
										#	timeA = holdTime
										#	time.sleep(0.01)
									#else:
									sock1.recv(8192)
									clearPacket += 1	
									if clearPacket == 8201:
										time.sleep(0.04)
										break
									#elif clearPacket == 8192*totalClears:
									#	break
									#clearPacket += 1	
								except:
									print(clearPacket)
									print('FUCK 1')
									#time.sleep(0.05)
									#if currentClears == totalClears:
										#print(clearPacket)
									#	currentClears = 0
									#	break
									#else:
									#	currentClears += 1

							print('TOOK ' + str(time.time() - initTime) + 'S TO CLEAR BUFFER 1')
							sock1.setblocking(1)
						pos = 8192
						view = memoryview(sharedData[i][j])
						view = view[pos:]
						#time.sleep(1)
						while pos < int((ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.EXTRA_MULT):
							cr1 = sock1.recv_into(view, 8192)
							pos += 2*cr1
							view = view[2*cr1:]
						#plt.plot(sharedData[i][j][8192:8192*2])
						#plt.show()
						writeStatus[i][j][1] = 1
						#time.sleep(1)
					j += 1
				i += 1
			i = 0
			testTime = time.time()	
			#print('\n\nFINISHED FILLING BUFFER FOR GBE1 ~ 1/2 TOOK ABOUT ' + str(round((time.time() - newStartTime)/2, 3)) + 'S \n\n')
			#print('GBE1 TOOK ' + str(round(testTime - newStartTime, 3)) + ' SECONDS TO READ ' + str((ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT/2*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT) + ' BYTES OF DATA')
			#print('IN THIS TIME '+ str(((testTime - newStartTime) * (ROACH_SETUP.PAYLOAD_LEN + 4)* ROACH_SETUP.CLOCK_SPEED)) + ' BYTES WERE TRANSMITTED') 
			#print('THE EFFICIENCY IS: ' + str(round(((ROACH_SETUP.PAYLOAD_LEN + 4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT*ROACH_SETUP.NUM_BUFFERS/2) * 100/ ((testTime - newStartTime) * ROACH_SETUP.CLOCK_SPEED), 3)) + '%\n\n')
		except Exception as e:
			print(e)
			sys.exit(1)

def createNewFile(fileName):
	hf = h5py.File(fileName, 'w')
	runInfo = hf.create_dataset('runInfo', dtype = 'f', data = [])
	runInfo.attrs['amplifer'] = 'Pasternack PE15A1012'
	runInfo.attrs['antenna'] = 'AB-900A Biconical'
	runInfo.attrs['averages'] = str(ROACH_SETUP.NUM_FFT) + '//' + str(2*ROACH_SETUP.NUM_FFT) + '//' + str(4*ROACH_SETUP.NUM_FFT) + '//' + str(8*ROACH_SETUP.NUM_FFT) + '//' + str(16*ROACH_SETUP.NUM_FFT) 
	acquisitionLength = 1000 * ROACH_SETUP.DATA_ITS*ROACH_SETUP.PAYLOAD_LEN*8 / ROACH_SETUP.CLOCK_SPEED
	runInfo.attrs['acquisition length'] = str(round(acquisitionLength, 3)) + 'ms' + '//' + str(round(acquisitionLength/2, 3)) + 'ms' + '//' +str(round(acquisitionLength/4, 3)) + 'ms' + '//' +str(round(acquisitionLength/8, 3)) + 'ms' + '//' +str(round(acquisitionLength/16, 3)) + 'ms'
	runInfo.attrs['clock speed'] = '2GHz'
	latestTime = datetime(1970, 1, 1, 0, 0)
	earliestTime = datetime(9999, 12, 31, 0, 0)
	runInfo.attrs['date range'] = str(earliestTime.strftime('%m/%d/%Y %H:%M:%SUTC')) + '-' + str(latestTime.strftime('%m/%d/%Y %H:%M:%SUTC'))
	runInfo.attrs['extra description'] = ''
	runInfo.attrs['packet length'] = '756'
	runInfo.attrs['packet period'] = '760'
	runInfo.attrs['temperature'] = '0K'
	runInfo.attrs['time format'] = 'UTC'
	runInfo.attrs['version'] = 'ROACH v00.1'
	runInfo.attrs['window'] = 'Kaiser'
	hf.create_group('0-1000')
	hf['0-1000'].create_dataset('freqs', dtype=np.float64, data = np.linspace(0, 1000, 2**25))
	#hf['31.25-62.5'].create_dataset('freqs', dtype=np.float64, data = np.linspace(31.25, 62.5, 2**20))
	#hf.create_group('62.5-125')
	#hf['62.5-125'].create_dataset('freqs', dtype=np.float64, data = np.linspace(62.5, 125, 2**20))
	#hf.create_group('125-250')
	#hf['125-250'].create_dataset('freqs', dtype=np.float64, data = np.linspace(125, 250, 2**20))
	#hf.create_group('250-500')
	#hf['250-500'].create_dataset('freqs', dtype=np.float64, data = np.linspace(250, 500, 2**20))
	#hf.create_group('500-1000')
	#hf['500-1000'].create_dataset('freqs', dtype=np.float64, data = np.linspace(500, 1000, 2**20))
	return hf

def writeData(dataMem):
	
	data_mem = shared_memory.SharedMemory(name=dataMem.name)
	dataArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'), buffer=data_mem.buf)

	#ROACH_SETUP: Root group - file with date-range
	#		Sub-group: YYYY-MM identifying year and month
	#		Sub-sub-group: MM-DD identifying month and day
	#		Sub-sub-sub group: Center frequency range to 

	dataCounter = 0
	fileCounter = 0
	fileName = 'data' + '_' + str(dataCounter) + '.h5'
	#hf = h5py.File(fileName, 'w')
	maxFileSize = 1000
	currentTime = datetime(1970, 1, 1, 0, 0)
	earliestTime = datetime(9999, 12, 31, 0, 0)
	firstFile = True
	#groupNames = ['31.25-62.5', '62.5-125', '125-250', '250-500', '500-1000']
	groupNames = ['0-1000']

	hf = createNewFile(fileName)

	# Headers for the columns of the dataframe

	oldFirst = 0

	#while time.time() - testTime < ROACH_SETUP.TOTAL_TIME:
	while True:
		if dataArr[0][0] != oldFirst:
			oldFirst = dataArr[0][0]	
			testTime = time.time()
			currentFileSize = os.path.getsize(fileName)/1024.**2
			if currentFileSize > maxFileSize:
				hf['runInfo'].attrs['date range'] = str(earliestTime.strftime('%H:%M:%S:%f')) + '-' + str(currentTime.strftime('%H:%M:%S:%f'))
				hf.close()
				fileCounter = fileCounter + 1
				fileName = 'data' + '_' + str(fileCounter) + '.h5'
				hf = createNewFile(fileName)
				firstFile = True
				print('ON FILE ' + str(fileCounter))
				print('MADE NEW FILE')
			
			currentTime = datetime.today()
			yearGroup = str(currentTime.year)
			monthGroup = str(currentTime.month)
			dayGroup = str(currentTime.day)
			dataSetName =  'measdata_' + str(dataCounter) + '_' +  str(currentTime.strftime('%Y-%m-%d--%H-%M-%S-%f'))
		
			for counter, val in enumerate(groupNames):
				if not(str(yearGroup) in hf[val]):
					hf[val].create_group(yearGroup)
				if not(str(monthGroup) in hf[val][yearGroup]):
					hf[val][yearGroup].create_group(monthGroup)
				if not(str(dayGroup) in hf[val][yearGroup][monthGroup]):
					hf[val][yearGroup][monthGroup].create_group(dayGroup)
				if not('BACKGROUND') in hf[val][yearGroup][monthGroup][dayGroup]:
					hf[val][yearGroup][monthGroup][dayGroup].create_group('BACKGROUND')
				if not('DATA') in hf[val][yearGroup][monthGroup][dayGroup]:
					hf[val][yearGroup][monthGroup][dayGroup].create_group('DATA')
				
				hf[val][yearGroup][monthGroup][dayGroup]['BACKGROUND'].create_dataset(dataSetName, dtype=np.float32, data = dataArr[0])
				#print(dataArr[0:100])
				hf[val][yearGroup][monthGroup][dayGroup]['DATA'].create_dataset(dataSetName, dtype=np.float32, data = dataArr[1])

				#plt.plot(writeData[counter][writeIndex[counter] - 1])
				#plt.show()
				#writeIndex[counter] = writeIndex[counter] - 1
				dataCounter = dataCounter + 1

			#if firstFile:
			#	earliestTime = currentTime
			#	firstFile = False

			hf['runInfo'].attrs['date range'] = str(earliestTime.strftime('%H:%M:%S:%f')) + '-' + str(currentTime.strftime('%H:%M:%S:%f'))
			stoppingTime = time.time()
			print('WRITING TOOK ' + str(round(stoppingTime-testTime, 3)) + 'S TO COMPLETE') 
	hf.close()


@cp.fuse
def squareMag(x):
	return cp.square(cp.abs(x))

def performFFT(memName, writeName, windowData, startName, fftName):
	startTime = time.time()
	ignore = False
	write_mem = [shared_memory.SharedMemory(name=writeName[i].name) for i in range(len(writeName))]
	writeStatus = [np.ndarray((ROACH_SETUP.NUM_FFT, 2), dtype=np.dtype('B'), buffer=write_mem[i].buf) for i in range(len(write_mem))]
	toggle = True
	start_mem = shared_memory.SharedMemory(name=startName.name)
	startArr = np.ndarray((1,), dtype=np.dtype('B'), buffer=start_mem.buf)

	existing_shm = [shared_memory.SharedMemory(name=memName[i][0].name) for i in range(len(memName))]
	sharedData =  [np.ndarray((ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN+4)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('B'), buffer=existing_shm[i].buf) for i in range(len(existing_shm))] 

	fft_mem = shared_memory.SharedMemory(name=fftName.name)
	fftArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'), buffer=fft_mem.buf)

	plan = cpx.scipy.fftpack.get_fft_plan((sharedData[0][0][0:2**ROACH_SETUP.FFT_POW].astype(np.float32)).reshape(1, 2**ROACH_SETUP.FFT_POW), axes = 1, value_type = 'C2C')
	#plan = cpx.scipy.fftpack.get_fft_plan(sharedData[0][0].astype(np.float32).reshape(8, 2**23), axes = 1, value_type = 'C2C')
	
	processStream = cp.cuda.stream.Stream(non_blocking = True)
	copyStream = cp.cuda.stream.Stream(non_blocking = True)

	x_cpu_src = np.arange(2**ROACH_SETUP.FFT_POW, dtype=np.float32)
	
	#x_cpu_dst
	#total = [np.asarray([]) for x in range(3)]
	#total[0] = np.arange(2**(ROACH_SETUP.FFT_POW-1), dtype=np.float32)
	#total[1] = np.arange(2**(ROACH_SETUP.FFT_POW-1), dtype=np.float32)
	#total[2] = np.arange(2**(ROACH_SETUP.FFT_POW-1), dtype=np.float32)


	x_gpu_dst = cp.empty(x_cpu_src.shape, np.uint8)
	x_gpu_dst_old = cp.empty(x_cpu_src.shape, np.uint8)
	x_gpu_fft = cp.asarray([cp.empty(2**(ROACH_SETUP.FFT_POW-1), np.float32) for x in range(3)])

	x_pinned_cpu_src = _pin_memory(x_cpu_src)
	#x_pinned_cpu_dst = [_pin_memory(x) for x in total]


	windowGPU = cp.array(windowData)
	currentBuf = 0
	totalBuf = len(write_mem)
	#sos = butter(9, 200*10**6, 'lp', fs = 2*10**9, output = 'sos')
	print('TOTAL BUFFERS: ' + str(totalBuf))
	currentFFT = 0
	firstTime = True
	#writeLocs = np.logspace(0, 4.5, num = 10, endpoint = True, base = 10.0, dtype = np.int32)
	fftLength = 2**ROACH_SETUP.FFT_POW
	deltaF = 1000/(fftLength)
	dataX = [x*deltaF for x in range(int(fftLength/2))]
	deltaT = [x/(2.) for x in range(2**ROACH_SETUP.FFT_POW)]
	totalIts = 0
	totalAcqs = 0
	
	totalCounter = [0,0]
	switched = False

	numOn = 1
	ratTotal = 1
	maxExtra = ((ROACH_SETUP.DATA_ITS*(ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.EXTRA_MULT - 2**ROACH_SETUP.FFT_POW)) / ((ROACH_SETUP.PAYLOAD_LEN+4)*4)
	print('MAX EXTRA: ' + str(maxExtra))

	while time.time() - startTime < ROACH_SETUP.TOTAL_TIME:
		while currentBuf < ROACH_SETUP.NUM_BUFFERS:
			fillStart = time.time()
			#toggle = not(toggle) 
			while currentFFT < ROACH_SETUP.NUM_FFT:
				if not(firstTime):
					if writeStatus[currentBuf][currentFFT][0] and writeStatus[currentBuf][currentFFT][1]:
						#timeBlah = time.time()
						#Negative value implies 1 is ahead, postive implies 0 is ahead
						if totalIts >=0:
							sigBytes = (ROACH_SETUP.PAYLOAD_LEN + 4)*4
		
							#Pull from 1
							extra_signed = int.from_bytes(sharedData[currentBuf][currentFFT][sigBytes-8:sigBytes], byteorder = 'big', signed = False) \
							- int.from_bytes(sharedData[currentBuf][currentFFT][sigBytes*2-8:sigBytes*2], byteorder = 'big', signed = False) + 1
							print('EXTRA: ' + str(extra_signed))
							
							extra = np.abs(extra_signed)
							if extra > maxExtra:
								print('THIS IS A SHITTY VALUE NEED TO RE-DO ' + str(totalIts))
								ignore = True
								
							holderArr = (sharedData[currentBuf][currentFFT][:(int(2**ROACH_SETUP.FFT_POW/sigBytes)+extra + 9 )*(sigBytes)]).reshape((-1, sigBytes))[::, :-8:] 
							extraArr = sharedData[currentBuf][currentFFT][:(int(2**ROACH_SETUP.FFT_POW/sigBytes)+extra + 9)*(sigBytes)].reshape((-1, sigBytes))[::, -8::]

							if extra_signed > 0:
								holderArr[1:-extra:2] = holderArr[1+extra::2]
								extraArr[1:-extra:2] = extraArr[1+extra::2]
			
							#Pull from 0
							elif extra_signed < 0:
								holderArr[:-extra:2] = holderArr[extra::2]
								extraArr[:-extra:2] = extraArr[extra::2]

							if totalIts == 10000000:
								aTest = holderArr[::, -8:]
								plotVals = [int.from_bytes(x, byteorder = 'big', signed = False) for x in aTest]

								plotValDiff = [val[0] - val[1] for val in zip(plotVals[1:], plotVals[:-1])]
								with open('NoiseTest_0_EnableOn_QChannel_ClosedTop_2GHz_4-30-21_40dBAmpMarconin127_123p45_30dBAtt_2_Timestamps.txt', 'w') as f:
									for val in plotValDiff:
										f.write(str(val) + '\n')
								plt.plot(plotValDiff, 'r.', label = 'DATA LINE BOTH')
								plt.show()
						
							#stamps = [int.from_bytes(x, byteorder = 'big', signed = False) for x in holderArr[::, -8:]]
							#stampDiff = [int(val[0] - val[1]) for val in zip(stamps[1:], stamps[:-1])]
							#plt.plot(stampDiff[0:8192])
							#plt.show()

							
								
							#Based on example for cupy_memcpy.py
							#print(extraArr)
							#plotVals = [int.from_bytes(x, byteorder = 'big', signed = False) for x in extraArr]
							#plotValDiff = [val[0] - val[1] for val in zip(plotVals[1:], plotVals[:-1])]

							#print(len(plotVals))

							holderArr = np.ravel(holderArr)[:2**ROACH_SETUP.FFT_POW]
							#print('MEAN: ' + str(np.mean(holderArr)))
							#print('STD: ' + str(np.std(holderArr)))
							#n, bins, _ = plt.hist(holderArr, 100, range = [100, 150], linewidth = 2)
							#print(holderArr[0:100])
							#plt.yscale('log', nonposy = 'clip')
							#plt.show()
							#plt.plot(holderArr)
							#plt.show()
							#plt.plot(plotVals)
							#plt.show()
							#holderArr = np.reshape(holderArr[::, 0:-8], (1, -1))

							#x_pinned_cpu_src = holderArr[0][0:2**ROACH_SETUP.FFT_POW]
							#print(len(holderArr))
							x_pinned_cpu_src = holderArr
							#plt.plot(x_pinned_cpu_src)
							#plt.show()
							with copyStream:
								#start = copyStream.record()
								x_gpu_dst.set(x_pinned_cpu_src)
								with processStream:
									if not(ignore):
										if totalIts >= 5:
											aTest = cp.abs(cpx.scipy.fftpack.fft((x_gpu_dst_old.astype(np.float32)-127)*0.940 / 256., overwrite_x = False, n = 2**ROACH_SETUP.FFT_POW, plan = plan)[:int(2**ROACH_SETUP.FFT_POW/2)])**2
																						#aTest = cp.abs(cp.fft.fft((x_gpu_dst_old-127)*0.940 / 256., n = 2**ROACH_SETUP.FFT_POW)[:int(2**ROACH_SETUP.FFT_POW/2)])**2											
											#plt.plot(10*np.log10(aTest.get()))
											#plt.show()
											if totalIts == 5:
												x_gpu_fft[1] = cp.copy(aTest)
												totalCounter[1] += 1
											elif totalIts == 5 + numOn:
												x_gpu_fft[0] = cp.copy(aTest)
												totalCounter[0] += 1
											else:
												if (totalIts-5)%(numOn+1) < numOn:
													#x_gpu_fft[1] = cp.add(x_gpu_fft[1], aTest)
													x_gpu_fft[1] = x_gpu_fft[1] + aTest

													totalCounter[1] += 1
													#print('A')
												else:
													#x_gpu_fft[0] = cp.add(x_gpu_fft[0], aTest)
													x_gpu_fft[0] = x_gpu_fft[0] + aTest

													totalCounter[0] += 1
													#print('B')
												#total[totalIts%2] = cp.add(total[totalIts%2], aTest)	
												#totalCounter[totalIts%2] += 1
											#if (totalIts) > 5:
											#	if totalIts == 6:
													#testAcq = cp.divide(aTest, aTestOld).get()
													#numSmall = 0
													#for rat in testAcq:
													#	if rat < 1:
													#		numSmall += 1
													#print(str(numSmall) + ' LESS THAN ONE OUT OF ' + str(len(testAcq)) + '. THIS IS ' + str(numSmall/len(testAcq)*100) + '%')
													#plt.plot(testAcq)
													#plt.show()
											#		x_gpu_fft[2] = cp.divide(aTest, aTestOld)

											#	elif not(totalIts%2):
											#		x_gpu_fft[2] = cp.add(x_gpu_fft[2], cp.divide(aTest, aTestOld))
											#		print(str(totalCounter))

											
											#aTestOld = cp.copy(aTest)
											#aTestOld = aTest
										#x_gpu_dst_old = cp.copy(x_gpu_dst)
										x_gpu_dst_old = x_gpu_dst
										totalIts = totalIts + 1
											#print(totalIts)
									else:
										print('RUN IS BAD NOT COUNTING IT')
										print(totalIts)
									
									if not(ignore):
										if toggle:
											pass
											#write_read('0')
											#time.sleep(0.1)
											#subprocess.run(['usbrelay', '1_2=0'])
											#dev.ctrl_transfer(requestType, 9, 0x0300, 0, (0xFD, 2, 0, 0, 0, 0, 0, 0))
		
											#sigGen.write('LV -44 DB')
											#subprocess.call(['python2', './setQInput.py'])
											#print('JUST SET TO Q INPUT')

											#print('JUST SET TO 0 dB')
											#print('JUST SET TO -84 DBM\n')
										
										else:
											pass
											#write_read('1')
											#time.sleep(0.2)
											#subprocess.call(['python2', './setIInput.py'])
											#print('JUST SET TO 31 dB')
											#sigGen.write('LV -124 DB')
											#print('JUST SET TO -124 DBM\n')
											#subprocess.run(['usbrelay', '1_2=1'])
											#dev.ctrl_transfer(requestType, 9, 0x0300, 0, (0xFF, 2, 0, 0, 0, 0, 0, 0))
											#print('JUST SET TO I INPUT')
										print('TOGGLED')

										time.sleep(0.3)
										toggle = not(toggle)
									
										
										writeStatus[bool(currentBuf) ^ bool(currentFFT)][(currentFFT+1)%2][0] = 0
										writeStatus[bool(currentBuf) ^ bool(currentFFT)][(currentFFT+1)%2][1] = 0
										#time.sleep(5)
									processStream.synchronize()
								copyStream.synchronize()

								#timeBlah = time.time()
								#if totalCounter[0] + totalCounter[1] > 1:
								#	fftArr[0] = x_gpu_fft[0].get() / totalCounter[0]
								#	fftArr[1] = x_gpu_fft[1].get() / totalCounter[1]
								#print('TOOK ' + str(round((time.time() - timeBlah), 6)) + 's TO COPY DATA FROM GPU')
								#time.sleep(2)
								if totalIts != 4 and totalIts % 100 == 4:
									print('\n\nDONE WITH ' + str(totalIts - 4))
								if totalIts != 1 and totalIts % 2004 == 1: # and (totalIts % 1) == 0:
									print('TOTAL COUNTER: ' + str(totalCounter))
									fftData = x_gpu_fft[1].get() / totalCounter[1]
									#currentTime = datetime.now().strftime("%H-%M-%S")
									#baseFileName = 'SwitchingTest_0_EnableOn_ClosedTop_2GHz_7-21-21_100Avg_SwitchingRelay_Amp50OnZeroANDShortOnOne_20dBAmp_n50dBmSignal_ShortDelay'
									baseFileName = 'TermTest_0_EnableOn_ClosedTop_2GHz_9-7-21_I_1000Avg_NewRelayOn_OldRelayTrigger_Term0ANDTerm1_180degTermFoilCovered_LongerDelay_DoorClosed_FFTTest11'
									with open(baseFileName + '_Sub1.txt', 'w') as f:
									#with open('AttenTest_0And31_EnableOn_QChannel_ClosedTop_2GHz_6-21-21_Term_EveryOther_Sub1.txt', 'w') as f:
										for fftVal in fftData:
											f.write(str(fftVal) + '\n')
									print('WROTE FILE ' + str(totalIts))
								
									fftData = x_gpu_fft[0].get() / totalCounter[0]
									#currentTime = datetime.now().strftime("%H-%M-%S")
									with open(baseFileName + '_Sub2.txt', 'w') as f:
									#with open('AttenTest_0And31_EnableOn_QChannel_ClosedTop_2GHz_6-21-21_Term_EveryOther_Sub2.txt', 'w') as f:
										for fftVal in fftData:
											f.write(str(fftVal) + '\n')
									
								#	fftData = x_gpu_fft[2].get() / totalCounter[0]

								#	with open(baseFileName + '_Ratio.txt', 'w') as f:
										#with open('AttenTest_0And31_EnableOn_QChannel_ClosedTop_2GHz_6-21-21_Term_EveryOther_Sub2.txt', 'w') as f:
								#		for fftVal in fftData:
								#			f.write(str(fftVal) + '\n')
								#end = copyStream.record()

								#print('Asynchronous Device to Host / Host to Device (ms)')
								#cp.cuda.Device().synchronize()
								#print(cp.cuda.get_elapsed_time(start, end))
							
							if ignore:
								print('TRYING TO FIX THE OFFSET')
								if extra_signed > 0:
									writeStatus[currentBuf][currentFFT][0] = 0
									writeStatus[currentBuf][currentFFT][1] = 1
									time.sleep(0.01)
									writeStatus[currentBuf][currentFFT][1] = 0
								else:
									writeStatus[currentBuf][currentFFT][0] = 1
									writeStatus[currentBuf][currentFFT][1] = 0
									time.sleep(0.01)
									writeStatus[currentBuf][currentFFT][0] = 0
								
								time.sleep(5)

							if not(ignore):
								currentFFT = currentFFT + 1
						
							ignore = False		
						#print('TOOK ' + str(round((time.time() - timeBlah), 6)) + 's TO COMPLETE FFT')
				else:
					#prod = 1
					filled = True
					for row in writeStatus:
						for col in row:
							if not(col[0]) or not(col[1]):
								filled = False
					if filled:
						switched = True
						print('\nSTUCK HERE ' + str(totalAcqs) + '\n')
						if startArr[0] and switched:
							totalAcqs = totalAcqs + 1
						if totalAcqs < 3:
							for x in range(ROACH_SETUP.NUM_BUFFERS):
								for y in range(ROACH_SETUP.NUM_FFT):
									writeStatus[x][y][0] = 0
									writeStatus[x][y][1] = 0
						else:
							firstTime = False
							currentFFT = 0
							currentBuf = 0
							print('SLEEPING FOR A SEC...\n\n')
							for x in range(ROACH_SETUP.NUM_BUFFERS):
								for y in range(ROACH_SETUP.NUM_FFT):
									writeStatus[x][y][0] = 1
									writeStatus[x][y][1] = 1
									#time.sleep(5)
							time.sleep(2)
							writeStatus[0][0][0] = 0
							writeStatus[0][0][1] = 0
							#time.sleep(7)
					#currentFFT = currentFFT + 1
			fillEnd = time.time()
			#print('IT TOOK ' + str(round(fillEnd - fillStart, 3)) + 's TO DO THE FFT')
			currentFFT = 0
			currentBuf = currentBuf + 1
		currentBuf = 0
		firstTime = False


		#totalIts = totalIts + 1
		
def create_shared_block():
	#createArr = np.empty(shape=(10000,), dtype=np.dtype('(ROACH_SETUP.PAYLOAD_LEN,)i8'))  # Start with an existing NumPy array 
	sharedMemoryData = shared_memory.SharedMemory(create=True, size=int((ROACH_SETUP.PAYLOAD_LEN+4)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.NUM_FFT*ROACH_SETUP.EXTRA_MULT))
	# # Now create a NumPy array backed by shared memory
	dataArr = np.ndarray((ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN+4)*ROACH_SETUP.DATA_ITS * 8 * ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('B'), buffer=sharedMemoryData.buf)
	#dataArr[:] = createArr[:]  # Copy the original data into shared memory
	return (sharedMemoryData, dataArr)


if __name__ == '__main__':
	try:
		startTime = time.time()
		windowData = []
		windowFile = open('kaiser_2E26_order14', 'r')
		windowData.append(np.fromfile(windowFile, dtype = 'd'))
		windowFile = open('kaiser_2E25-almost-_order14', 'r')
		windowData.append(np.fromfile(windowFile, dtype = 'd'))
		windowFile = open('kaiser_2E24-almost-_order14', 'r')
		windowData.append(np.fromfile(windowFile, dtype = 'd'))
		windowFile = open('kaiser_2E23-almost-_order14', 'r')
		windowData.append(np.fromfile(windowFile, dtype = 'd'))
		windowFile = open('kaiser_2E22-almost-_order14', 'r')
		windowData.append(np.fromfile(windowFile, dtype = 'd'))
		writeStatus = []
		write_mem = []

		for i in range(ROACH_SETUP.NUM_BUFFERS):
			write_mem.append(shared_memory.SharedMemory(create=True, size = ROACH_SETUP.NUM_FFT*2))
			#write_mem_1.append(shared_memory.SharedMemory(create=True, size=int(ROACH_SETUP.NUM_FFT/2)))
			writeStatus.append(np.ndarray((ROACH_SETUP.NUM_FFT,2), dtype='B', buffer=write_mem[i].buf))
			for j in range(ROACH_SETUP.NUM_FFT):
				writeStatus[i][j][0] = 0
				writeStatus[i][j][1] = 0

		endTime = time.time()
		print('IT TOOK ' + str(round(endTime - startTime, 3)) + 'S TO READ WINDOW FILE IN')

		circBuf = []

		#write_buffer_mem = shared_memory.SharedMemory(create = True, size = 8400000*5*6)
		#writeBufferArr = np.ndarray((5, 6, 2**20,), dtype=np.float64, buffer=write_buffer_mem.buf)


		start_mem = shared_memory.SharedMemory(create = True, size = 2)
		startArr = np.ndarray((1,), dtype='B', buffer=start_mem.buf)

		write_index_mem = shared_memory.SharedMemory(create = True, size = 64)
		writeIndexArr = np.ndarray((5,), dtype = np.dtype('int64'), buffer=write_index_mem.buf)

		fft_mem = shared_memory.SharedMemory(create = True, size = 2 * 2**(ROACH_SETUP.FFT_POW-1) * 4)
		fftArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW-1)), dtype = np.dtype('float32'), buffer=fft_mem.buf)
		fftArr[0] = np.zeros(2**(ROACH_SETUP.FFT_POW - 1), dtype = np.dtype('float32'))
		fftArr[1] = np.zeros(2**(ROACH_SETUP.FFT_POW - 1), dtype = np.dtype('float32'))
		#time_stamp_mem = shared_memory.SharedMemory(create = True, size = 2**ROACH_SETUP.FFT_POW*4)
		#timeStampArr = np.ndarray((2**ROACH_SETUP.FFT_POW,), dtype = np.dtype('float32'), buffer = time_stamp_mem.buf)
		
		#handshake_mem = shared_memory.SharedMemory(create = True, size = 2)
		#handshakeArr = np.ndarray((2,), dtype = np.dtype('B'), buffer = handshake_mem.buf)
		#handshakeArr[0] = 0
		#handshakeArr[1] = 0

		for i in range(ROACH_SETUP.NUM_BUFFERS):
			circBuf.append(create_shared_block())
		
		startTime = time.time()
		for j in range(1):
			#ctx = mp.get_context('spawn')
			eth1 = mp.Process(target = gbe0, name = 'gbe0', args = (circBuf, write_mem, start_mem))
			eth2 = mp.Process(target = gbe1, name = 'gbe1', args = (circBuf, write_mem, start_mem))
			fftProc = mp.Process(target = performFFT, name = 'performFFT', args = (circBuf, write_mem, windowData[0], start_mem, fft_mem))
			writeProc = mp.Process(target = writeData, name = 'writeData', args = (fft_mem,))
			print('ON ITERATION ' + str(j))
				
			fftProc.start()
			print('SLEEPING FOR A BIT...')
			time.sleep(10)
			print('DONE SLEEPING')
			subprocess.call(['python2', './enableOutput.py'])
			eth1.start()
			eth2.start()
			#writeProc.start()
			#time.sleep(10)
						#ROACH_funcs.enableROACH(fpga)
			#time.sleep(5)
			startArr[0] = True
		
			eth1.join()
			eth2.join()
			fftProc.join()
			#writeProc.join()
			endTime = time.time()
			print('THAT TOOK ' + str(round(endTime - startTime, 3)) + 'S TO COMPLETE')
			#allocateFFTMem(circBuf[0][0].name, FFTNames, windowDataGPU)
			#fft_proc_1 = mp.Process(target = allocateFFTMem, name = 'fftROACH_SETUP', args = (circBuf[0][0].name, FFTNames, windowData, ([write_mem_0[x].name for x in range(len(write_mem_0))], [write_mem_1[x].name for x in range(len(write_mem_1))])))
			
		for i in range(ROACH_SETUP.NUM_BUFFERS):
			circBuf[i][0].close()
			circBuf[i][0].unlink()
			write_mem[i].close()
			write_mem[i].unlink()
		#write_buffer_mem.close()
		#write_buffer_mem.unlink()
		write_index_mem.close()
		write_index_mem.unlink()
		#time_stamp_mem.unlink()
		#time_stamp_mem.close()
		#handshake_mem.unlink()
		#handshake_mem.close()
		start_mem.unlink()
		start_mem.close()
		fft_mem.close()
		fft_mem.unlink()
		print('THAT TOOK ' + str(round(endTime - startTime, 3)) + 'S TO COMPLETE')
	except Exception as e:
		print(e)


