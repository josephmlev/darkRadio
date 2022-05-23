import numpy as np 
from datetime import datetime
from numpy import loadtxt
import corr
import ctypes
import cupy as cp
import cupyx as cpx
import glob
import h5py
import logging
import numba as nb
import ROACH_funcs
import ROACH_SETUP
import time
import itertools
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
from multiprocessing.managers import BaseManager
import os
import pandas as pd
from pathlib import Path
import pyqtgraph as pg
import pyqtgraph.exporters
import pyvisa as visa
import queue
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt
import ScpiInstrumentWrapper
from screeninfo import get_monitors
import serial
import signal
import socket
import socketserver
import struct
import subprocess
import sys 
import timeit
import threading
import usb.core
import usb.util 



ser = serial.Serial('/dev/ttyACM0', 115200)
ser.write(b'0b0')

#sigGen = visa.ResourceManager().open_resource('USB0::1689::851::1625504::0::INSTR')
#sigGen.write('OUTPUT2:STATE ON')

#sigGen = pyvisa.ResourceManager().open_resource('USB0::1689::836::C020120::0::INSTR')
#sigGen = ScpiInstrumentWrapper.ScpiInstrumentWrapper('PROLOGIX::/dev/ttyUSB4::GPIB::10')
#sigGen.write('CF 123.45 MZ')
#sigGen.write('LV -124 DB')
#sigGen.write('VOLTAGE:AMPLITUDE -20DBM')
#arduino =  serial.Serial(port='/dev/ttyACM0', baudrate = 115200, timeout=.1)
#subprocess.call(['python2', './progROACH.py', '192.168.40.76', '--arp'])
#sys.exit(1)
#idVendor = 0x16c0
#idProduct = 0x05df
#idVendor = 0x0403
#idProduct = 0x6011
#dev = usb.core.find(idVendor = idVendor, idProduct = idProduct)
#print(cfg)
#intf = cfg[(0,0)]
#endPoint = intf[0]
#if dev.is_kernel_driver_active(0):
#	try:
#		dev.detach_kernel_driver(0)
#		print("kernel driver detached")
#	except usb.core.USBError as e:
#		sys.exit("Could not detach kernel driver: %s" % str(e))

#requestType = usb.util.build_request_type(usb.util.CTRL_OUT, usb.util.CTRL_TYPE_CLASS, usb.util.CTRL_RECIPIENT_INTERFACE)
sock0 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock0.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)



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

def is_number(x):
	try:
		float(x)
		return True
	except Exception as e:
		return False


def getTemp():
	totalTries = 0
	maxTries = 20
	totalAvg = 5
	tempCount = 0
	totalTemp = 0
	while tempCount < totalAvg:
		ser.write(b'0b2')
		possibleTemp = ser.readline()
		totalTries = totalTries + 1
		if(is_number(possibleTemp)):
			totalTemp = totalTemp + float(possibleTemp)
			tempCount = tempCount + 1
		if totalTries > maxTries:
			print('TOO MANY READS TO GET TEMPERATURE')
			break
	if tempCount == 0:
		return 0
	else:
		return totalTemp / tempCount + 273.15

def write_read(x):
	arduino.write(bytes(x, 'utf-8'))

def _pin_memory(array):
	mem = cp.cuda.alloc_pinned_memory(array.nbytes)
	ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
	ret[...] = array
	return array


#sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock1.setblocking(1)

def gbe0(memName, writeName, startName, endName):

	startTime = time.time()
	toggle = False
	#existing_shm = [shared_memory.SharedMemory(name=memName[i][0].name) for i in range(len(memName))]
	adc_mem = shared_memory.SharedMemory(name=memName.name)
	sharedData =  np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('i1'), buffer=adc_mem.buf) 
	
	write_mem = shared_memory.SharedMemory(name=writeName.name)
	writeStatus = np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT), dtype=np.dtype('B'), buffer=write_mem.buf)
	
	start_mem = shared_memory.SharedMemory(name=startName.name)
	startArr = np.ndarray((1,), dtype=np.dtype('B'), buffer=start_mem.buf)

	end_mem = shared_memory.SharedMemory(name = endName.name)
	endArr = np.ndarray((1,), dtype = np.dtype('B'), buffer = end_mem.buf)

	firstRun = True
	#sock0.setblocking(1)
	i = 0
	totalClears = 1
	while not(endArr[0]) and (time.time() - startTime < ROACH_SETUP.TOTAL_TIME):
		newStartTime = time.time()
		try:
			while i < ROACH_SETUP.NUM_BUFFERS:
				j = 0
				clearOut = 0				
				while j < ROACH_SETUP.NUM_FFT:
					pos = 0
					if startArr[0] and ((not(writeStatus[i][j]))):
						newStartTime = time.time()
						view = memoryview(sharedData[i][j])
						view = view[pos:]
						if firstRun:
							sock0.bind((ROACH_SETUP.UDP0_IP, ROACH_SETUP.GBE0_PORT))
							sock0.setblocking(1)
							#ROACH_funcs.resetFIFO()
							#print('FIFO PERCENT: ' + str(ROACH_funcs.readFIFOPercent()))
							try:
								pass
								#sock0.bind((ROACH_SETUP.UDP0_IP, ROACH_SETUP.GBE0_PORT))
							except Exception as e:
								print('ALREADY BOUND\n\n\n\n\n\n\n')
							sock0.setblocking(1)
							firstRun = False
						if not(firstRun):
							pass
							#pos = 0
							#timeA = time.time()
							#initTime = timeA
							#holdTime = timeA
							#sock0.setblocking(1)
							#clearPacket = 0
							#currentClears = 0
							#while 1:
							#	try:
									#if holdTime - initTime < 0.2:
									#	sock0.recv(8192)
									#	clearPacket += 1
									#	holdTime = time.time()
									#	if holdTime - timeA > 0.005:
									#		timeA = holdTime
									#		time.sleep(0.01)
									#else:
									#time.sleep(1)
									#print(cr0)

									#print(str(cr0) + '\t' + str(clearPacket))
									#clearPacket += 1	
									#if clearPacket == 8201:
									#	time.sleep(0.04)
									#	break
									#elif clearPacket == 8192*totalClears:
									#	break
							#	except:
							#		print(clearPacket)
									#time.sleep(0.05)
									#if currentClears == totalClears:
										#print(clearPacket)
									#	currentClears = 0
									#	break
									#else:
									#	currentClears += 1

							#print('TOOK ' + str(time.time() - initTime) + 'S TO CLEAR BUFFER 0')
							#sock0.setblocking(1)
						pos = 0
						view = memoryview(sharedData[i][j])
						view = view[pos:]
						#sigGen.write('OUTPUT2:STATE ON')
						while pos < int((ROACH_SETUP.PAYLOAD_LEN)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.EXTRA_MULT):
							#cr0 = sock0.recv_into(memoryview(sharedData[i][j])[pos:])
							cr0 = sock0.recv_into(view, 8192)
							pos += cr0
							view = view[cr0:]
						#print(pos)
						#stamps = [int.from_bytes(x, byteorder = 'big', signed = False) for x in (sharedData[i][j].reshape((-1, 8200)))[::, -8:]]
						#diffs = [z[1] - z[0] for z in zip(stamps[:-1], stamps[1:])]
						#holderArr = np.ravel((sharedData[i][j].reshape((-1, 8200))[::, :-8:])) 
						#plt.plot(holderArr)
						#plt.show()
						#plt.show()
						#print('GBE0 TOOK ' + str(round(time.time() - newStartTime, 3)) + ' SECONDS TO FILL (' + str(i) + ', ' + str(j) + ')')
						writeStatus[i][j] = 1
						j += 1
					elif not(firstRun):
						sock0.recv(8192)
						clearOut = clearOut + 1
						if not(clearOut % 10000):
							print('CLEARING OUT THINGS ' + str(clearOut))
				i += 1 
			i = 0
			testTime = time.time()
			#print('\n\nFINISHED FILLING BUFFER FOR ` ~ 1/2 TOOK ABOUT ' + str(round((time.time() - newStartTime)/2, 3)) + 'S \n\n')
			#print('GBE0 TOOK ' + str(round(testTime - newStartTime, 3)) + ' SECONDS TO READ ' + str((ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT/2*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT) + ' BYTES OF DATA')
			#print('IN THIS TIME '+ str(((testTime - newStartTime) * (ROACH_SETUP.PAYLOAD_LEN + 4)* ROACH_SETUP.CLOCK_SPEED)) + ' BYTES WERE TRANSMITTED') 
			#print('THE EFFICIENCY IS: ' + str(round(((ROACH_SETUP.PAYLOAD_LEN + 4)*8*ROACH_SETUP.DATA_ITS*ROACH_SETUP.NUM_FFT*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT/2) * 100/ ((testTime - newStartTime) * ROACH_SETUP.CLOCK_SPEED), 3)) + '%\n\n')
			#if endThings:
			#	print('GBE0 HANGING SO KILLED')
		except KeyboardInterrupt as e:
			print('KEYBOARD INTERRUPT')
			sys.exit(0)
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
	runInfo.attrs['packet length'] = '8192'
	runInfo.attrs['packet period'] = '8192'
	runInfo.attrs['temperature'] = str(round(getTemp(), 2)) + 'K'
	runInfo.attrs['time format'] = 'PDT'
	runInfo.attrs['version'] = 'ROACH v00.1'
	runInfo.attrs['window'] = 'Rectangular'
	hf.create_group('0-300')
	#hf['0-300'].create_dataset('freqs', dtype=np.float64, data = np.linspace(0, 1000, 2**25))
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
	#firstArr = np.zeros((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'))
	fileArr = np.zeros((4, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'))
	avgArr = np.zeros(2**(ROACH_SETUP.FFT_POW - 1), dtype = np.dtype('float32'))
	#ROACH_SETUP: Root group - file with date-range
	#		Sub-group: YYYY-MM identifying year and month
	#		Sub-sub-group: MM-DD identifying month and day
	#		Sub-sub-sub group: Center frequency range to 

	dataCounter = 0
	fileCounter = 0
	datasetCounter = 0

	fileName = 'data' + '_' + str(dataCounter) + '.h5'
	#hf = h5py.File(fileName, 'w')
	maxFileSize = 1000
	currentTime = datetime(1970, 1, 1, 0, 0)
	earliestTime = datetime(9999, 12, 31, 0, 0)
	firstFile = True
	#groupNames = ['31.25-62.5', '62.5-125', '125-250', '250-500', '500-1000']
	groupNames = ['0-300']

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
				datasetCounter = 0
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
				

				#if not('BACKGROUND') in hf[val][yearGroup][monthGroup][dayGroup]:
				#	hf[val][yearGroup][monthGroup][dayGroup].create_group('BACKGROUND')
				#if not('DATA') in hf[val][yearGroup][monthGroup][dayGroup]:
				#	hf[val][yearGroup][monthGroup][dayGroup].create_group('DATA')
				
				#hf[val][yearGroup][monthGroup][dayGroup]['BACKGROUND'].create_dataset(dataSetName, dtype=np.float32, data = dataArr[0])
				#print(dataArr[0:100])
				datasetCounter = datasetCounter + 1
				fileArr[0] = dataArr[0]
				fileArr[1] = dataArr[1]
				fileArr[2] = fileArr[0] - fileArr[1]
				if datasetCounter < ROACH_SETUP.TOTAL_AVG:
					avgArr = (avgArr*(datasetCounter-1) + fileArr[2])/datasetCounter
				else:
					avgArr = (avgArr*(ROACH_SETUP.TOTAL_AVG-1) + fileArr[2])/ROACH_SETUP.TOTAL_AVG
					fileArr[3] = np.greater(fileArr[2], avgArr + ROACH_SETUP.THRESHOLD)

					#for index, diffVal in enumerate(zip(fileArr[2], avgArr)):
					#	if np.abs(diffVal[1] - diffVal[0]) > ROACH_SETUP.THRESHOLD:
					#		fileArr[3][index] = True
					#	else:
					#		fileArr[3][index] = False

				#col_names = ['SPEC_A', 'SPEC_B', 'DIFF', 'SPURIOUS']
				#ds_dt = np.dtype( { 'names':col_names, 'formats':[np.dtype('float32'), np.dtype('float32'), np.dtype('float32'), np.dtype('float32')]})
				#rec_arr = np.rec.array(fileArr,dtype=ds_dt)        
				ds = hf[val][yearGroup][monthGroup][dayGroup].create_dataset(dataSetName, data = np.transpose(fileArr))
				dataCounter = dataCounter + 1

				#plt.plot(writeData[counter][writeIndex[counter] - 1])
				#plt.show()
				#writeIndex[counter] = writeIndex[counter] - 1

			
			if firstFile:
				earliestTime = currentTime
				firstFile = False

			hf['runInfo'].attrs['date range'] = str(earliestTime.strftime('%H:%M:%S:%f')) + '-' + str(currentTime.strftime('%H:%M:%S:%f'))
			stoppingTime = time.time()
			print('\n\n\n\n\n\n\n\n\nWRITING TOOK ' + str(round(stoppingTime-testTime, 3)) + 'S TO COMPLETE\n\n\n\n\n\n\n\n\n') 
	hf.close()

def takeRigolData(rigolQueue, endName):
	# Initialize the Rigol
	RM = visa.ResourceManager('@py')
	INST = RM.open_resource('TCPIP0::169.254.119.50::INSTR')
	INST.write(':INST SA')
	time.sleep(8)
	INST.write(':DET:POS')
	time.sleep(0.1)
	INST.write(':CORR:SA:GAIN 0')
	time.sleep(0.1)
	INST.write(':UNIT:POW DBM')
	time.sleep(0.1)
	INST.write(':INIT:CONT OFF')
	time.sleep(0.1)
	INST.write(':SENS:POW:RF:ATT 0')
	time.sleep(0.1)
	INST.write(':DISP:WIND:TRAC:Y:SCAL:RLEV 0')
	time.sleep(0.1)
	INST.write(':DISP:WIND:TRAC:Y:PDIV 15')
	time.sleep(0.1)
	INST.write(':BAND:SHAP RECT')
	time.sleep(0.1)
	INST.write(':SWE:TIME:AUTO ON')
	INST.write(':TRAC:MODE: AVER')
	time.sleep(0.1)
	INST.write(':TRAC:TYPE AVER')
	time.sleep(0.1)
	INST.write(':BAND 10000')
	time.sleep(0.1)
	INST.write(':FREQ:SPAN 300000000')
	time.sleep(0.1)
	INST.write(':FREQ:CENT 150000000')
	print(INST.query(':SWE:TIME?'))
	acqTime = float(INST.query(':SWE:TIME?'))
	numRigolAvg = 10 #int(10/acqTime)
	INST.write(':TRAC:AVER:COUN ' + str(numRigolAvg))
	time.sleep(0.1)
	INST.write(':INIT')
	checkRigolTime = time.time()
	#while time.time() - testTime < ROACH_SETUP.TOTAL_TIME:
	oldRigolTime = time.time()
	
	end_mem = shared_memory.SharedMemory(name = endName.name)
	endArr = np.ndarray((1,), dtype = np.dtype('B'), buffer = end_mem.buf)

	while(not(endArr[0])):
		if time.time() - oldRigolTime > 1:
			if int(INST.query(':TRAC:AVER:COUNT:CURR?')) >= numRigolAvg:
				rigolTime = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
				if not(endArr[0]):
					rigolQueue.put((rigolTime, [np.float32(x) for x in (INST.query('TRAC:DATA? TRACE1')).split(',')]))
				INST.write(':INIT')
				print('STARTED A NEW RIGOL SCAN')
			#else:
			#	print('CHECKED BUT NOT READY FOR RIGOL WRITE')
			oldRigolTime = time.time()
	print('WAITING FOR 20s TO CLEAR THE QUEUE')
	for sleepIncrement in range(4):
		time.sleep(5)
		print(str(20 - sleepIncrement*5) + 'S...')
	
	while rigolQueue.qsize() > 0:
		rigolQueue.get()


def plotCurrentAcq(dataMem, endName, plotQueue):
	#label_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
	#		  'verticalalignment':'bottom'} 
	#title_font = {'fontname':'sans-serif', 'size':'16', 'color':'black', 'weight':'bold'}
	#legend_font = {'family':'sans-serif', 'size':'10', 'style':'normal'} 

	title_font = {'color':'white', 'font-size':'16px', 'weight':'bold', 'font':'Arial'}

	#plt.tick_params(axis='both', which='major', labelsize=11)
	#mpl.rcParams['agg.path.chunksize'] = 10000
	data_mem = shared_memory.SharedMemory(name=dataMem.name)
	dataArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'), buffer=data_mem.buf)
	end_mem = shared_memory.SharedMemory(name = endName.name)
	endArr = np.ndarray((1,), dtype = np.dtype('B'), buffer = end_mem.buf) 
	writeDir = ROACH_SETUP.SAVE_DIR

	lastRigol = np.zeros(10000)
	RigolFreqs = np.linspace(0, 300, 10000)
	totalRigolAvgCount = 0
	RigolAvg = np.zeros(10000)

	endRun = False
	#Plotting stuff

	labelFont = pg.Qt.QtGui.QFont()
	labelFont.setFamily('Arial')
	labelFont.setPointSize(16)

	#generate layout
	app = pg.mkQApp("Data Viewer v0.0")
	#win = pg.GraphicsLayoutWidget(show=True)
	#win.setWindowTitle('Data Viewer v0.0')
	#label = pg.LabelItem(justify='right')
	labelPos_ROACH = pg.TextItem(color = 'white')
	labelPos_Rigol = pg.TextItem(color = 'white')
	#labelAnt_ROACH = pg.LabelItem(color = 'white', justify = 'right')

	#labelAnt_Identify = pg.TextItem(color = 'green', anchor = (0, 0))
	#labelTerm_Identify = pg.TextItem(color = 'red')

	#labelTerm_ROACH.setFont(labelFont
	#labelTerm_ROACH = pg.TextItem(color = 'red')

	#win.addItem(label)
	view = pg.GraphicsView()
	#view.addItem(label)
	view.addItem(labelPos_ROACH)
	view.addItem(labelPos_Rigol)

	#view.addItem(labelAnt_ROACH)

	#view.addItem(labelAnt_Identify)
	#view.addItem(labelTerm_Identify)

	#view.addItem(labelTerm_ROACH)

	layout = pg.GraphicsLayout(border=(100,100,100))
	#layout.setContentsMargins(0, 0, 0, 0)
	view.setCentralItem(layout)
	view.show()
	view.setWindowTitle('Data Viewer v0.0')
	view.resize(800,600)


	pROACH_single = layout.addPlot(colspan = 1)

	# customize the averaged curve that can be activated from the context menu:
	#p1.avgPen = pg.mkPen('#FFFFFF')
	#p1.avgShadowPen = pg.mkPen('#8080DD', width=10)

	pRigol_single = layout.addPlot(colspan = 1)
	layout.nextRow()
	pROACH_average = layout.addPlot(colspan = 1)
	pRigol_average = layout.addPlot(colspan = 1)
	layout.nextRow()
	timePlot = layout.addPlot(colspan=2)
	timePlot.setMaximumHeight(150)
	view.resize(700, 800)

	#region = pg.LinearRegionItem()
	#region.setZValue(10)
	# Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
	# item when doing auto-range calculations.
	#p2.addItem(region, ignoreBounds=True)

	#pg.dbg()
	pROACH_single.setAutoVisible(y=True)
	pRigol_single.setAutoVisible(y=True)
	

	#create numpy arrays
	#make the numbers large to show that the range shows data from 10000 to all the way 0
	freqs = np.asarray(range(2**(ROACH_SETUP.FFT_POW-1)))*ROACH_SETUP.CLOCK_SPEED/10**6/(2**ROACH_SETUP.FFT_POW)
	numAvg = 2**3
	removeBins = 1 
	freqsAvg = np.mean(np.reshape(freqs, (-1, numAvg)), axis = 1)

	data1 = np.zeros(int(2**(ROACH_SETUP.FFT_POW - 1)/numAvg))
	data2 = np.zeros(int(2**(ROACH_SETUP.FFT_POW - 1)/numAvg))
	label_font = {'color':'white', 'font-size':'15px', 'font':'Arial'}
	
	pROACH_single.setLabel('bottom', 'Frequency', 'MHz', **label_font)
	pROACH_single.setLabel('left', 'Power', 'dBm', **label_font)

	pRigol_single.setLabel('bottom', 'Frequency', 'MHz', **label_font)
	pRigol_single.setLabel('left', 'Power', 'dBm', **label_font)

	pROACH_average.setLabel('bottom', 'Frequency', 'MHz', **label_font)
	pROACH_average.setLabel('left', 'Power', 'dBm', **label_font)

	pRigol_average.setLabel('bottom', 'Frequency', 'MHz', **label_font)
	pRigol_average.setLabel('left', 'Power', 'dBm', **label_font)

	pROACH_single.addLegend()

	pRigol_single_dataAnt = pRigol_single.plot(RigolFreqs[removeBins:], RigolAvg[removeBins:], pen="g")
	
	pROACH_single_dataAnt = pROACH_single.plot(freqsAvg[removeBins:], data1[removeBins:], pen="r", name = 'Terminator')
	pROACH_single_dataTerm = pROACH_single.plot(freqsAvg[removeBins:], data2[removeBins:], pen="g", name = 'Antenna')
	
	#pROACH_single_labelPlot = pROACH_single.plot([0, 300], [0, 0], pen="black")
	#pROACH_single_labelPlot_Point = pg.CurvePoint(pROACH_single_labelPlot)
	#view.addItem(pROACH_single_labelPlot_Point)

	#pRigol_single_labelPlot = pRigol_single.plot([0, 300], [0, 0], pen="black")
	#pRigol_single_labelPlot_Point = pg.CurvePoint(pRigol_single_labelPlot)
	#view.addItem(pRigol_single_labelPlot_Point)

	pRigol_average_dataAnt = pRigol_average.plot(RigolFreqs[removeBins:], RigolAvg[removeBins:], pen="r")

	pROACH_average_dataAnt = pROACH_average.plot(freqsAvg[removeBins:], data1[removeBins:], pen="r")
	pROACH_average_dataTerm = pROACH_average.plot(freqsAvg[removeBins:], data2[removeBins:], pen="g")


	#labelAnt_Identify.setPos(300, 30)
	#labelAnt_Identify.setText('Antenna')
	#labelTerm_Identify.setPos(300, 55)
	#labelTerm_Identify.setText('Terminator')
	#labelAnt_Identify.setParentItem(pROACH_single)
	#labelAnt_Identify.anchor(itemPos=(1,0), parentPos=(1,0), offset=(-10,10))
	
	#labelAnt_Identify.setText('Antenna')
	#labelAnt_Identify.setPos(500, 100)
	#labelTerm_Identify.setText('Terminator')
	#labelTerm_Identify.setPos(570, 100)
	#p2d = p2.plot(data1, pen="w")
	#p1d = p1.plot(data1, pen = 'w')
	# bound the LinearRegionItem to the plotted data
	#region.setClipItem(p2d)
	#oldFirst = dataArr[0][0]

	#def update():
	#	region.setZValue(10)
	#	minX, maxX = region.getRegion()
	#	p1.setXRange(minX, maxX, padding=0)    
	if not(endRun):
		def updateData(dataAnt, dataTerm):
			#data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
			#data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
			#p1DataA.setData(data1)
			#p1DataB.setData(data2)
			#p2d.setData(data2)
			
			avgFile = ROACH_SETUP.SAVE_DIR + '/CurrentRunInfo/AllAverages.h5'
			avgPath = Path(avgFile)
			
			RigolFile = ROACH_SETUP.SAVE_DIR + '/CurrentRunInfo/AllAverages_RIGOL.txt'				
			avgPathRigol = Path(avgFile)

			updatePlots = True 
			if avgPath.is_file():
				dataBin = h5py.File(avgFile, 'r+')
				oldAnt = int(dataBin.attrs['lastAnt'])
				#print(holder)
				#oldFirst = int(float(holder[holder.index('_')+1:].rstrip()))
				print(oldAnt)
				print(int(dataTerm[0]))
				print(oldAnt == int(dataAnt[0]))
				if oldAnt == int(dataAnt[0]):
					updatePlots = False
					print('NOT UPDATING!!!')
				if updatePlots:
					#dfAvg = pd.read_csv(avgFile)
					allData = pd.read_hdf(avgFile)
					averageAnt = np.asarray(allData['AvgAnt'])
					averageTerm = np.asarray(allData['AvgTerm'])
					#totalAverages = float(dfAvg.columns[1][dfAvg.columns[1].index('_')+1:])
					totalAverages = float(dataBin.attrs['totalAvg'])
					

				dataBin.close()
			else:
				averageAnt = np.zeros(len(freqs))
				averageTerm = np.zeros(len(freqs))
				totalAverages = 0
			if dataAnt[0] == 0 and dataTerm[0] == 0:
				updatePlots = False
			#print(len(dataAnt))
			#print(len(dataTerm))
			#print(len(globalAverage))
			if updatePlots:
				if avgPathRigol.is_file():
					with open(RigolFile, 'r') as f:
						firstLine = f.readline()
						totalRigolAvgCount = int(int(firstLine.split()[-1]))
						nonlocal RigolAvg
						for counter, line in enumerate(f):
								RigolAvg[counter] = float(line)

				else:
					totalRigolAvgCount = 0
					RigolAvg = np.zeros(10000)

				antPower = 10.*np.log10(2. * dataAnt  / ((2**ROACH_SETUP.FFT_POW)**2 * 50. / 1000.))
				termPower = 10.*np.log10(2. * dataTerm / ((2**ROACH_SETUP.FFT_POW)**2 * 50. / 1000.))
				
				# First number is needed because data is divided already in performFFT
				averageAnt = 10*np.log10((10**(averageAnt/10.)*totalAverages + 10**(antPower/10.)*ROACH_SETUP.NUM_BEFORE_TOGGLE) / (totalAverages + ROACH_SETUP.NUM_BEFORE_TOGGLE))
				averageTerm = 10*np.log10((10**(averageTerm/10.)*totalAverages + 10**(termPower/10.)*ROACH_SETUP.NUM_BEFORE_TOGGLE) / (totalAverages + ROACH_SETUP.NUM_BEFORE_TOGGLE))
				totalAverages += ROACH_SETUP.NUM_BEFORE_TOGGLE


				getFirst = True
				currentRigolCount  = 0
				currentPlot = np.zeros(10000)
				while plotQueue.qsize() > 0:
					if getFirst:
						lastRigol = np.asarray(plotQueue.get())
						#print(lastRigol)
						#print(len(lastRigol))
						currentPlot = lastRigol
						getFirst = False
					else:
						currentPlot = 10*np.log10(10**(currentPlot/10.) + 10**(np.asarray(plotQueue.get())/10.))
					currentRigolCount += 1

				if totalRigolAvgCount > 0 and currentPlot[0] != 0:
					RigolAvg = 10*np.log10((10**(RigolAvg/10.)*totalRigolAvgCount + 10**(currentPlot/10.)*currentRigolCount)/(totalRigolAvgCount + currentRigolCount)) 
				elif currentRigolCount > 0:
					RigolAvg = currentPlot - 10*np.log10(currentRigolCount) 
				else:
					lastRigol = np.zeros(10000)
					print('NO RIGOL DATA TO PLOT')
				
				totalRigolAvgCount += currentRigolCount
						
				print('SAVING AND DISPLAYING DATA')
				df = pd.DataFrame(zip(averageAnt, averageTerm), columns=['AvgAnt', 'AvgTerm'])
				df.to_hdf(avgFile, key = 'data', mode='w')
				dataBin = h5py.File(avgFile, 'r+')
				dataBin.attrs['totalAvg'] = totalAverages
				dataBin.attrs['lastAnt'] = dataAnt[0]
				dataBin.attrs['lastTerm'] = dataTerm[0]
				dataBin.close()
				with open(RigolFile, 'w') as f:
						f.write('TOTAL AVG: ' + str(int(totalRigolAvgCount)) + '\n')
						for RigolValue in RigolAvg:
							f.write(str(RigolValue) + '\n')
								
				antPowerReduced = 10*np.log10(np.mean(np.reshape(10**(antPower/10.), (-1, numAvg)), axis = 1))
				termPowerReduced = 10*np.log10(np.mean(np.reshape(10**(termPower/10.), (-1, numAvg)), axis = 1))
				averageAntReduced = 10*np.log10(np.mean(np.reshape(10**(averageAnt/10.), (-1, numAvg)), axis = 1))
				averageTermReduced = 10*np.log10(np.mean(np.reshape(10**(averageTerm/10.), (-1, numAvg)), axis = 1))

				#print(len(antPowerReduced))
				#print(len(termPowerReduced))
				#print(len(averageAntReduced))
				#print(len(averageTermReduced))

				currentTime = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
				
				pROACH_single_dataAnt.setData(freqsAvg[removeBins:], antPowerReduced[removeBins:])
				pROACH_single_dataTerm.setData(freqsAvg[removeBins:], termPowerReduced[removeBins:])
				
				pROACH_average_dataAnt.setData(freqsAvg[removeBins:], averageAntReduced[removeBins:])
				pROACH_average_dataTerm.setData(freqsAvg[removeBins:], averageTermReduced[removeBins:])

				
				pRigol_single_dataAnt.setData(RigolFreqs[removeBins:], lastRigol[removeBins:])
				pRigol_average_dataAnt.setData(RigolFreqs[removeBins:], RigolAvg[removeBins:])

				#pROACH_single.setlabel = pg.LabelItem(justify='right')
				pROACH_single.setLabel('top', str(currentTime), **title_font)
				pROACH_average.setLabel('top', str(int(totalAverages)) + ' AVERAGES', **title_font)
				pRigol_average.setLabel('top', str(int(totalRigolAvgCount*10)) + ' AVERAGES', **title_font)

				#p1.setLabel('top', 'ZOOM', **title_font)

				saveTime = currentTime.replace(' ', 'AT')
				saveTime = saveTime.replace(':', '-')
				saveName = ROACH_SETUP.SAVE_DIR + '/RunImages/EventDisplay_' + str(saveTime) + '.png'
				exporter = pg.exporters.ImageExporter(layout.scene() )
				exporter.export(saveName)
				print('SAVED DISPLAY FILE')
				#plt.savefig(ROACH_SETUP.SAVE_DIR + 'EventDisplay_' + str(saveTime) + '.pdf', bbox_inches='tight')
				#plt.show(block=False)
				#oldFirst = dataAnt[0]
			#if firstRun:
			#	firstRun = False
			if endArr[0]:
				endRun = True
				return 
		#region.sigRegionChanged.connect(update)

	#	def updateRegion(window, viewRange):
	#		rgn = viewRange[0]
	#		region.setRegion(rgn)

	#	p1.sigRangeChanged.connect(updateRegion)

	#	region.setRegion([freqsAvg[1000], freqsAvg[2000]])

		#cross hair
		vLine = pg.InfiniteLine(angle=90, movable=False)
		hLine = pg.InfiniteLine(angle=0, movable=False)
		pROACH_single.addItem(vLine, ignoreBounds=True)
		pROACH_single.addItem(hLine, ignoreBounds=True)
		#labelPos_ROACH.setParentItem(pROACH_single_labelPlot_Point)

		#labelPos_Rigol.setParentItem(pRigol_single_labelPlot_Point)


		vb = pROACH_single.vb
		vRigol = pRigol_single.vb
		def mouseMovedROACH(evt):
			pos = evt[0]  ## using signal proxy turns original arguments into a tuple
			if pROACH_single.sceneBoundingRect().contains(pos):
				mousePoint = vb.mapSceneToView(pos)
				index = int(mousePoint.x())
				if index > 0 and index < len(data1):
					#axX = pROACH_single.getAxis('bottom')
					#axY = pROACH_single.getAxis('left')
					#print(axX.range)

					#labelFreq_ROACH.setText('x: NULL y: NULL')
					#mappedPoint = pROACH_single.mapFromDevice(pg.Point(axX.range[1] - 0.1*(axX.range[1]-axX.range[0]) , axY.range[1] - 0.3*(axY.range[1]-axY.range[0])))
					#print(mappedPoint.x())
					#print(mappedPoint.y())
					#print('\n\n')
					#totalX = axX.range[1] - axX.range[0]
					#curveDataX = np.linspace(axX.range[0] + 0.05*totalX, axX.range[1] - 0.05*totalX, 10)
					
					#curveDataY = np.asarray([0.8*(axY.range[1]- axY.range[0]) + (axY.range[0])]*10)
					#pROACH_single_labelPlot.setData(curveDataX, curveDataY)
					#pROACH_single_labelPlot_Point.setPos(curveDataX[0])
					labelPos_ROACH.setText('x: ' + str(round(mousePoint.x(), 3)) + '\ny: ' + str(round(mousePoint.y(), 3)))
					labelPos_ROACH.setPos(800, 50)
					labelPos_ROACH.setFont(labelFont)

				vLine.setPos(mousePoint.x())
				hLine.setPos(mousePoint.y())
		
		def mouseMovedRigol(evt):
			pos = evt[0]
			if pRigol_single.sceneBoundingRect().contains(pos):
				mousePoint = vRigol.mapSceneToView(pos)
				index = int(mousePoint.x())
				if index > pRigol_single.getAxis('bottom').range[0] and index < pRigol_single.getAxis('bottom').range[1]:
					

					#axX = pROACH_single.getAxis('bottom')
					#axY = pROACH_single.getAxis('left')

					#totalX = axX.range[1] - axX.range[0]
					#curveDataX = np.linspace(axX.range[0] + 0.05*totalX, axX.range[1] - 0.05*totalX, 10)
					#curveDataY = np.asarray([0.8*(axY.range[1]- axY.range[0]) + (axY.range[0])]*10)

					#pRigol_single_labelPlot.setData(curveDataX, curveDataY)
					#pRigol_single_labelPlot_Point.setPos(curveDataX[0])
					labelPos_Rigol.setText('x: ' + str(round(mousePoint.x(), 3)) + '\ny: ' + str(round(mousePoint.y(), 3)))
					labelPos_Rigol.setPos(1700, 50)

					#labelFreq_ROACH.setPos(mappedPoint.x(), mappedPoint.y())
					labelPos_Rigol.setFont(labelFont)
		#pROACH_single.scene().sigMouseMoved.connect(mouseMoved)

		
		#fig = plt.figure(num=3,figsize=((get_monitors()[0].width_mm)/25.4/4, (get_monitors()[0].height_mm)/25.4/4))
		
		#fig, axs = plt.subplot_mosaic([['upper left', 'upper right'], ['lower', 'lower']], figsize=((get_monitors()[0].width_mm)/25.4, (get_monitors()[0].height_mm)/25.4), constrained_layout=True)
		#axs['upper left'].set_xlabel('Frequency (MHz)', **label_font)
		#axs['upper left'].set_ylabel('Power (dBm)', **label_font)
		#axs['upper right'].set_xlabel('Frequency (MHz)', **label_font)
		#axs['upper right'].set_ylabel('Power (dBm)', **label_font)
		#axs['lower'].set_xlabel('Frequency (MHz)', **label_font)
		#axs['lower'].set_ylabel('Power (dBm)', **label_font)

		#gs = mpl.gridspec.GridSpec(2, 2)

		#ax0 = plt.subplot(gs[0, 0]) # row 0, col 0
		#plt.ion()
		#ax1 = plt.subplot(gs[0, 1]) # row 0, col 1
		#plt.ion()
		#ax2 = plt.subplot(gs[1, :]) # row 1, span all columns
		#plt.ion()

		#freqs = np.asarray(range(2**(ROACH_SETUP.FFT_POW-1)))*ROACH_SETUP.CLOCK_SPEED/10**6/(2**ROACH_SETUP.FFT_POW)
		
		#freqsAvg = np.mean(np.reshape(freqs, (-1, numAvg)), axis = 1)

		proxyROACH = pg.SignalProxy(pROACH_single.scene().sigMouseMoved, rateLimit=60, slot=mouseMovedROACH)
		proxyRigol = pg.SignalProxy(pRigol_single.scene().sigMouseMoved, rateLimit=60, slot=mouseMovedRigol)

		timer = pg.QtCore.QTimer()
		timer.timeout.connect(lambda: updateData(dataArr[0], dataArr[1]))
		timer.start(5000)
		pg.Qt.QtGui.QApplication.instance().exec_()
		#axs['upper left'].plot(freqsAvg, np.asarray(range(len(freqsAvg))))
		#axs['upper right'].plot(freqsAvg, np.asarray(range(len(freqsAvg))))
		#axs['lower'].plot(freqsAvg, np.asarray(range(len(freqsAvg))))
		#fig.canvas.draw()
		#fig.canvas.flush_events()
	



	#firstRun = True
	if endRun:
		return
	#while not(endArr[0]):
	
	#	else:
	#		time.sleep(1)



def writeDataNew(dataMem, rigolQueue, endName, plotQueue):
	
	data_mem = shared_memory.SharedMemory(name=dataMem.name)
	dataArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'), buffer=data_mem.buf)
	fileArr = np.zeros((4, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'))
	avgArr = np.zeros(2**(ROACH_SETUP.FFT_POW - 1), dtype = np.dtype('float32'))
	end_mem = shared_memory.SharedMemory(name = endName.name)
	endArr = np.ndarray((1,), dtype = np.dtype('B'), buffer = end_mem.buf) 

	#colNames = np.empty(ROACH_SETUP.NUM_DATASETS_PER_FILE, dtype = 'S25')
	colNames = []
	subColNames = ['SUB A', 'SUB B', 'DIFF', 'BAD']
	#dataframeConvertDict = {'SUB A': np.float32, 'SUB B': np.float32, 'DIFF': np.float32, 'BAD': np.bool}
	#ROACH_SETUP: Root group - file with date-range
	#		Sub-group: YYYY-MM identifying year and month
	#		Sub-sub-group: MM-DD identifying month and day
	#		Sub-sub-sub group: Center frequency range to 
	writeDir = ROACH_SETUP.SAVE_DIR
	dataCounter = 0
	fileCounter = 0
	allFiles = glob.glob(writeDir + '/' + 'data_*.h5')
	if len(allFiles) > 0:
		startFile = max([int(x[x.index('data_') + 5: x.index('.h5')]) for x in allFiles]) + 1
		allKeys = [aKey for  aKey in h5py.File(writeDir + '/' + 'data_' + str(startFile - 1) + '.h5', 'r').keys()]
		if len(allKeys) > 0:
			startDataset = max([int(x[x.index('_')+1:]) for x in allKeys]) + 1
		else:
			startDataset = 0
	else:
		startFile = 0
		startDataset = 0
	
	
	datasetCounter = 0
	
	fileName = writeDir + '/' + 'data' + '_' + str(fileCounter + startFile) + '.h5'
	#hf = h5py.File(fileName, 'w')

	#groupNames = ['31.25-62.5', '62.5-125', '125-250', '250-500', '500-1000']
	#groupNames = ['0-300']

	#hf = createNewFile(fileName)

	# Headers for the columns of the dataframe

	oldFirst = 0


	holderRigol = {}
	while not(endArr[0]):
		if dataArr[0][0] != oldFirst:
			oldFirst = dataArr[0][0]	
			print('WRITING DATA NOW!!!!')
			testTime = time.time()
			#currentFileSize = os.path.getsize(fileName)/1024.**2
			testTime = time.time()
			if (datasetCounter % ROACH_SETUP.NUM_DATASETS_PER_FILE) == 0:
				#allCols = pd.MultiIndex.from_product([[colNames[0]], subColNames])
				#hf['runInfo'].attrs['date range'] = str(earliestTime.strftime('%H:%M:%S:%f')) + '-' + str(currentTime.strftime('%H:%M:%S:%f'))
				#hf.close()
				fileName = writeDir + '/'+ 'data' + '_' + str(fileCounter + startFile) + '.h5'
				hf = h5py.File(fileName, 'w')
				hf.attrs['amplifer'] = 'PE15A1012_OLD-PE15A1014-10dB Attenuator'
				hf.attrs['antenna'] = 'SMA Resistor'
				hf.attrs['averages'] = ROACH_SETUP.NUM_TOGGLES * 1000
				acquisitionLength = 1000*2**ROACH_SETUP.FFT_POW / ROACH_SETUP.CLOCK_SPEED
				hf.attrs['acquisition length'] = str(round(acquisitionLength, 3)) + 'ms' 
				hf.attrs['clock speed'] = str(ROACH_SETUP.CLOCK_SPEED/10**6) + 'MHz'
				hf.attrs['extra description'] = ''
				hf.attrs['packet length'] = str(ROACH_SETUP.PKT_PERIOD - 8)
				hf.attrs['packet period'] = str(ROACH_SETUP.PKT_PERIOD)

				hf.attrs['temperature'] = str(round(getTemp(), 2)) + 'K'
				hf.attrs['time format'] = 'PDT'
				hf.attrs['version'] = 'ROACH v00.1'
				hf.attrs['window'] = 'Rectangular'
				#hf.create_group('RIGOL')
				fileCounter = fileCounter + 1
				
				#newDF = pd.read_hdf('data' + '_' + str(fileCounter) + '.h5')
				#print(newDF)
				#datasetCounter = 0
				#colNames = []
				#datasetCounter = 0
				#hf = createNewFile(fileName)
				#firstFile = True
				#print('ON FILE ' + str(fileCounter))
				print('MADE NEW FILE')
				#del df
			#colNames[(datasetCounter-1)] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
			#colNames.append(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
			colNames = [str(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])]
			if datasetCounter%ROACH_SETUP.NUM_DATASETS_PER_FILE == 0:
				earliestTime = colNames[0]
			elif (datasetCounter + 1)%ROACH_SETUP.NUM_DATASETS_PER_FILE == 0:
				latestTime = colNames[0]
				hf.attrs['date range'] = str(earliestTime) + 'PST' + '-' + str(latestTime) + 'PST'

			#fileArr[4*(datasetCounter)] = dataArr[0]
			#fileArr[4*(datasetCounter) + 1] = dataArr[1]
			#fileArr[4*(datasetCounter) + 2] = dataArr[1] - dataArr[0]
			fileArr[0] = dataArr[0]
			fileArr[1] = dataArr[1]
			fileArr[2] = dataArr[1]-dataArr[0]
			if datasetCounter < ROACH_SETUP.TOTAL_AVG:
				avgArr = (avgArr*(datasetCounter) + fileArr[2])/(datasetCounter + 1)
			else:
				avgArr = (avgArr*(ROACH_SETUP.TOTAL_AVG-1) + fileArr[2])/ROACH_SETUP.TOTAL_AVG
				fileArr[3] = np.greater(fileArr[2], avgArr + ROACH_SETUP.THRESHOLD)
		
			allCols = pd.MultiIndex.from_product([[colNames[0]], subColNames])
			#keyName = 'measdata_' + str(datasetCounter)
			#fileArr = np.transpose(fileArr)
			#df = pd.DataFrame(list(zip(*[fileArr[0].astype(np.float32), fileArr[1].astype(np.float32), fileArr[2].astype(np.float32), fileArr[3].astype(np.bool)])), columns=allCols)
			df = pd.DataFrame(fileArr.transpose(), columns = allCols)
			df.loc[:, pd.IndexSlice[:, 'BAD']] = (df.loc[:, pd.IndexSlice[:, 'BAD']].astype(np.bool))
			#print(df.dtypes)
			#print(df.index.get_level_values(0))
			#print(df.index.levels)
			#.astype(dataframeConvertDict)
			df.to_hdf(fileName, key = 'measdata_' + str(datasetCounter + startDataset), mode='a', format = 'table', complevel = 9)

			#hf['measdata_' + str(datasetCounter)]
			#hf['RIGOL'].create_group('measdata_' + str(datasetCounter))
			RigolCounter = 0
			while rigolQueue.qsize() > 0:
				rigolData = rigolQueue.get()
				plotQueue.put(rigolData[1])
				dset = hf['measdata_'+str(datasetCounter + startDataset)].create_dataset('RIGOL_' + str(RigolCounter), data = rigolData[1], dtype = np.float32) #compression = 'gzip', compression_opts = 9, 
				dset.attrs['time'] = rigolData[0]
				RigolCounter = RigolCounter + 1
			datasetCounter = datasetCounter + 1
			RigolCounter = 0
			print('\n\n\n\n\n\n\n\n\nWRITING TOOK ' + str(round(time.time()-testTime, 3)) + 'S TO COMPLETE\n\n\n\n\n\n\n\n\n') 
			if (datasetCounter % ROACH_SETUP.NUM_DATASETS_PER_FILE) == 0 and fileCounter == 1:
				endArr[0] = True
				print('ENDING RUNNING')	
@cp.fuse
def squareMag(x):
	return cp.square(cp.abs(x))

def performFFT(memName, writeName, startName, fftName, saveName, endName):

	startTime = time.time()
	#write_mem = [shared_memory.SharedMemory(name=writeName[i].name) for i in range(len(writeName))]
	#writeStatus = [np.ndarray((ROACH_SETUP.NUM_FFT, 2), dtype=np.dtype('B'), buffer=write_mem[i].buf) for i in range(len(write_mem))]
	write_mem = shared_memory.SharedMemory(name=writeName.name)
	writeStatus = np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT), dtype = np.dtype('B'), buffer=write_mem.buf)
	toggle = True
	
	start_mem = shared_memory.SharedMemory(name=startName.name)
	startArr = np.ndarray((1,), dtype=np.dtype('B'), buffer=start_mem.buf)

	write_mem = shared_memory.SharedMemory(name=writeName.name)
	writeStatus = np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT), dtype=np.dtype('B'), buffer=write_mem.buf)

	#write_mem = shared_memory.SharedMemory(name=writeName.name)
	#writeStatus = np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT), dtype=np.dtype('B'), buffer=write_mem.buf)


	#existing_shm = [shared_memory.SharedMemory(name=memName[i][0].name) for i in range(len(memName))]
	#sharedData =  [np.ndarray((ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN+4)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('B'), buffer=existing_shm[i].buf) for i in range(len(existing_shm))] 
	adc_mem = shared_memory.SharedMemory(name=memName.name)
	sharedData =  np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('i1'), buffer=adc_mem.buf) 

	fft_mem = shared_memory.SharedMemory(name=fftName.name)
	fftArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'), buffer=fft_mem.buf)

	save_mem = shared_memory.SharedMemory(name=saveName.name)
	saveArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW - 1)), dtype = np.dtype('float32'), buffer=save_mem.buf)

	end_mem = shared_memory.SharedMemory(name = endName.name)
	endArr = np.ndarray((1,), dtype = np.dtype('B'), buffer = end_mem.buf) 
	endArr[0] = False

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


	x_gpu_dst = cp.empty(x_cpu_src.shape, np.dtype('i1'))
	x_gpu_dst_old = cp.empty(x_cpu_src.shape, np.dtype('i1'))
	x_gpu_fft = cp.asarray([cp.empty(2**(ROACH_SETUP.FFT_POW-1), np.float32) for x in range(3)])

	x_pinned_cpu_src = _pin_memory(x_cpu_src)
	#x_pinned_cpu_dst = [_pin_memory(x) for x in total]

	injectedFreq = 10
	# Using a rectangular window for right now
	#windowGPU = cp.array(windowData)
	currentBuf = 0
	totalBuf = ROACH_SETUP.NUM_BUFFERS
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
	totalToggles = 0
	totalCounter = [0,0]
	switched = False
	blah = []
	numOn = ROACH_SETUP.NUM_BEFORE_TOGGLE
	ratTotal = 1
	maxExtra = ((ROACH_SETUP.DATA_ITS*(ROACH_SETUP.PAYLOAD_LEN+4)*8*ROACH_SETUP.EXTRA_MULT - 2**ROACH_SETUP.FFT_POW)) / ((ROACH_SETUP.PAYLOAD_LEN+4)*4)
	print('MAX EXTRA: ' + str(maxExtra))

	while not(endArr[0]) and (time.time() - startTime < ROACH_SETUP.TOTAL_TIME):
		while currentBuf < ROACH_SETUP.NUM_BUFFERS:
			fillStart = time.time()
			#toggle = not(toggle) 
			while currentFFT < ROACH_SETUP.NUM_FFT:
				if not(firstTime):
					if writeStatus[currentBuf][currentFFT]:
						timeBlah = time.time()
						#Negative value implies 1 is ahead, postive implies 0 is ahead
						if totalIts >=0:
							sigBytes = (ROACH_SETUP.PAYLOAD_LEN)*8
								
							holderArr = (sharedData[currentBuf][currentFFT]) #.reshape((-1, sigBytes))#[::, :-8:] 
							

							#plotVals = [int.from_bytes(x, byteorder = 'big', signed = True) for x in sharedData[currentBuf][currentFFT].reshape((-1, 8))]#[::, :-8:]]
							#plotValDiff = [val[0] - val[1] for val in zip(plotVals[1:], plotVals[:-1])]
							#plt.plot(plotVals, 'b-')
							#plt.show()
							#timestampArr = (sharedData[currentBuf][currentFFT]).reshape((-1, sigBytes))[::, -8::] 

							if totalIts == 10000000:
								aTest = holderArr[::, -8:]
								plotVals = [int.from_bytes(x, byteorder = 'big', signed = True) for x in aTest]

								plotValDiff = [val[0] - val[1] for val in zip(plotVals[1:], plotVals[:-1])]
								with open('NoiseTest_0_EnableOn_QChannel_ClosedTop_2GHz_4-30-21_40dBAmpMarconin127_123p45_30dBAtt_2_Timestamps.txt', 'w') as f:
									for val in plotValDiff:
										f.write(str(val) + '\n')
								plt.plot(plotValDiff, 'r.', label = 'DATA LINE BOTH')
								plt.show()
						
							#stamps = [int.from_bytes(x, byteorder = 'big', signed = False) for x in (sharedData[currentBuf][currentFFT]).reshape((-1, sigBytes))[::, -8::]]
							#stampDiff = [int(val[0] - val[1]) for val in zip(stamps[1:], stamps[:-1])]
							#plt.plot(stampDiff, 'r.')
							#plt.show()
							#packetCaptureTest[totalIts*8192:(totalIts+1)*8192] = stamps
							#print(max(stampDiff))
							#if totalIts == 1000:
								#aTest = holderArr[::, -8:]
							#	with open('CaptureTest_9-25-21.txt', 'w') as f:
							#		for val in packetCaptureTest:
							#			f.write(str(val) + '\n')
							
								
							#Based on example for cupy_memcpy.py
							#print(extraArr)
							#plotVals = [int.from_bytes(x, byteorder = 'big', signed = False) for x in extraArr]
							#plotValDiff = [val[0] - val[1] for val in zip(plotVals[1:], plotVals[:-1])]

							#print(len(plotVals))

							#plt.plot(holderArr)
							#plt.show()
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
								#plt.plot(x_pinned_cpu_src)
								#plt.show()
								x_gpu_dst.set(x_pinned_cpu_src)
								with processStream:
									if totalIts >= 10:
										#print(len(holderArr))
										#packetCount = holderArr.reshape((-1, 8))
										#testPlot = np.asarray([int.from_bytes(x, byteorder = 'big', signed = True) for x in packetCount])
										#plt.plot(testPlot[1:] - testPlot[:-1])
										#plt.show()
										#print('LENGTH: ' + str(len(testPlot)))	
										#packetCount = (sharedData[currentBuf][currentFFT]).reshape((-1, sigBytes))[::, -8:]
										#testPlot = np.asarray([int.from_bytes(x, byteorder = 'big', signed = True) for x in packetCount])
										#print('LENGTH: ' + str(len(testPlot)))
										#plt.plot()
										#plt.show()
										#print('DINGO LORRIS')
										if 0: #plot time domain data
											plotData = x_gpu_dst_old.astype(np.float32).get()
											plt.title('MAX VALUE: ' + str(int(max(plotData))) + '\tMIN VALUE: ' + str(int(min(plotData))))
											plt.plot(plotData)
											plt.show()
											plt.hist(plotData, bins = 127*2)
											plt.gca().set_yscale('log')
											plt.show()
										aTest = cp.abs(cpx.scipy.fftpack.fft((x_gpu_dst_old.astype(np.float32)-127)*0.940 / 256., overwrite_x = False, n = 2**ROACH_SETUP.FFT_POW, plan = plan)[:int(2**ROACH_SETUP.FFT_POW/2)])**2
										#sigGen.write('OUTPUT2:STATE OFF')
										
										#time.sleep(0.5)
										#if totalIts%2 == 0:
										#plotVals = x_gpu_dst_old.astype(np.float32).get()
											#if len(blah) == 0:
										#plt.plot(holderArr)
										#plt.show()
										#blahPow = 10.*np.log10(2. * aTest.get() / (2**48 * 50. / 1000.))[1:]
										#blahFreqs = (np.asarray(range(2**23))*600/2**24)[1:]
										#print('MAX OF ' + str(round(np.amax(blahPow[1:]), 3)) + ' dBm AT ' + str(blahFreqs[np.argmax(blahPow)+1]))
										#print(time.time())
										#blahLow = 0 
										#blahHigh = -1
										#blahLow = int(211/600*2**24)
										#blahHigh = int(214/600*2**24)
										#plt.plot(blahFreqs[blahLow:blahHigh], blahPow[blahLow:blahHigh])
										#plt.xlabel('Frequency (MHz)')
										#plt.ylabel('Power (dBm)')
										#plt.show()

										#	print(max(plotVals))
										#	print(min(plotVals))
										#	plt.show()
										#	blah = 2*np.abs(np.fft.fft(plotVals*0.940 / 256., 2**(ROACH_SETUP.FFT_POW))[0:int(2**(ROACH_SETUP.FFT_POW)/2)])**2
											#else:
											#	blah = blah + 2*np.abs(np.fft.fft(plotVals*0.940 / 256., 2**(ROACH_SETUP.FFT_POW))[0:int(2**(ROACH_SETUP.FFT_POW)/2)])**2
											#if totalIts == 100:
										#	blahPow = 10.*np.log10(blah / ((2**(ROACH_SETUP.FFT_POW))**2 * 50. / 1000.))		
										#	freqs = np.asarray(range(int(2**(ROACH_SETUP.FFT_POW)/2)))*6*10**2/2**24
										#	plt.plot(freqs[1:], blahPow[1:])
										#	plt.xlabel('Frequency (MHz)')
										#	plt.ylabel('Power (dBm)')
												
										#	plt.show()
										#	plt.clf()
										#	baseFileName = 'BathtubTest_0_600MHz_BoxClosed_1-3-22_n16dBm_119p9982MHz_20dBGain0dBAtt_FFTPow16_' + str(injectedFreq) + 'MHz.txt'
										#	times = np.asarray(range(2**16))/600
										#	with open(baseFileName, 'w') as f:
										#		for writeRaw in zip(times, plotVals[:2**16]):
										#			f.write(str(writeRaw[0]) + '\t' + str(writeRaw[1]) + '\n')
											
										#plotVals = x_gpu_dst_old.astype(np.float32).get()
										#times = np.asarray(range(len(plotVals)))/(6.*10**2)
											#with open('SampleData_300MHzClock_n20dBmInjectedAt120p0006MHzMarconi_IEE1241_12-13-21.txt', 'w') as f:
											#	f.write('Time (us)\t\tADC Counts\n')
											#	for testVal in zip(times[:2**22], plotVals[:2**22]):
											#		f.write(str(round(testVal[0], 5)) + '\t' + str(int(testVal[1])) + '\n')

										#plt.xlabel('Time (us)')
										#plt.ylabel('ADC Counts')
										#plt.plot(times, plotVals, 'b')
										#plt.show()
											#plt.plot(times[:2**16], plotVals[:2**16])
											#plt.show()
											#plt.clf()
											
										#	injectedFreq = injectedFreq + 10
																					#aTest = cp.abs(cp.fft.fft((x_gpu_dst_old-127)*0.940 / 256., n = 2**ROACH_SETUP.FFT_POW)[:int(2**ROACH_SETUP.FFT_POW/2)])**2											
										#plt.plot(10*np.log10(aTest.get()))
										#plt.show()
										#if totalIts == 10: 
										#	x_gpu_fft[1] = cp.copy(aTest)
										#	totalCounter[1] += 1
										#elif totalIts == 10 + numOn:
										#	x_gpu_fft[0] = cp.copy(aTest)
										#	totalCounter[0] += 1
										#else:
										if (totalIts-10)%(2*numOn) < (numOn):
											#x_gpu_fft[1] = cp.add(x_gpu_fft[1], aTest)
											x_gpu_fft[1] = x_gpu_fft[1] + aTest
											#plt.plot(10*np.log10(x_gpu_fft[1].get()[1000:]/(totalCounter[1]+1)))
											#plt.show()
											totalCounter[1] += 1
											#print('STORED IN A TOOK FFT OF (' + str(currentBuf) + ', ' + str(currentFFT) + '): TOGGLE IS ' + str(toggle))
										else:
											#x_gpu_fft[0] = cp.add(x_gpu_fft[0], aTest)
											x_gpu_fft[0] = x_gpu_fft[0] + aTest

											totalCounter[0] += 1
												#print('STORED IN B TOOK FFT OF (' + str(currentBuf) + ', ' + str(currentFFT) + '): TOGGLE IS ' + str(toggle))

										if (totalIts-10) % numOn == numOn - 1:
											time.sleep(0.1) # Important
											if toggle:
												ser.write(b'0b1')	
											else:
												ser.write(b'0b0')
											
								
											print('TOGGLED ' + str(totalIts))
										
											toggle = not(toggle)
										
											totalToggles += 1
										
											if not(totalToggles % ROACH_SETUP.NUM_TOGGLES):
												timeCopy = time.time()
												saveArr[0] = x_gpu_fft[0].get()/totalCounter[0]
												print('TOTAL COUNTER 0: ' + str(totalCounter[0]))
												x_gpu_fft[0] = cp.zeros(len(x_gpu_fft[0]))
												saveArr[1] = x_gpu_fft[1].get()/totalCounter[1]
												print('TOTAL COUNTER 1: ' + str(totalCounter[1]))
												x_gpu_fft[1] = cp.zeros(len(x_gpu_fft[1]))
												totalCounter[0] = 0 
												totalCounter[1] = 0
												print('TOOK ' + str(round((time.time() - timeCopy), 6)) + 's TO GET OFF GPU')
											time.sleep(0.9)

									x_gpu_dst_old = x_gpu_dst
									totalIts = totalIts + 1
										#print(totalIts)
								
									#if (totalIts-10) % numOn == numOn - 1:
										#print('TOGGLED ' + str(totalIts))
										#pass
									#	time.sleep(0.1) # Important
									#	if toggle:
									#		ser.write(b'0b0')
											#pass
										#write_read('0')
										#subprocess.run(['usbrelay', '1_2=0'])
											#dev.ctrl_transfer(requestType, 9, 0x0300, 0, (0xFD, 2, 0, 0, 0, 0, 0, 0))
	
										#sigGen.write('LV -44 DB')
										#subprocess.call(['python2', './setQInput.py'])
										#print('JUST SET TO Q INPUT')

										#print('JUST SET TO 0 dB')
										#print('JUST SET TO -84 DBM\n')
									
									#	else:
									#		ser.write(b'0b1')
											#pass
										#write_read('1')
									#	print('TOGGLED ' + str(totalIts))
									#	time.sleep(0.9)
										##subprocess.call(['python2', './setIInput.py'])
										#print('JUST SET TO 31 dB')
										#sigGen.write('LV -124 DB')
										#print('JUST SET TO -124 DBM\n')
										#subprocess.run(['usbrelay', '1_2=1'])
											#dev.ctrl_transfer(requestType, 9, 0x0300, 0, (0xFF, 2, 0, 0, 0, 0, 0, 0))
										#print('JUST SET TO I INPUT')
										

									#time.sleep(1)
										#writeStatus[int(not(currentBuf))][currentFFT] = 0
										
									#	toggle = not(toggle)
										
									#	totalToggles += 1
									#	if not(totalToggles % ROACH_SETUP.NUM_TOGGLES):
									#		timeCopy = time.time()
									#		saveArr[0] = x_gpu_fft[0].get()/totalCounter[0]
									#		x_gpu_fft[0] = cp.zeros(len(x_gpu_fft[0]))
									#		saveArr[1] = x_gpu_fft[1].get()/totalCounter[1]
									#		x_gpu_fft[1] = cp.zeros(len(x_gpu_fft[1]))
									#		totalCounter[0] = 0 
									#		totalCounter[1] = 1
									#		print('TOOK ' + str(round((time.time() - timeCopy), 6)) + 's TO GET OFF GPU')

										#time.sleep(1.5)
		
									
									#writeStatus[bool(currentBuf) ^ bool(currentFFT)][(currentFFT+1)%2] = 0
									#time.sleep(5)
									processStream.synchronize()
								copyStream.synchronize()
								writeStatus[currentBuf][currentFFT] = 0
								#timeBlah = time.time()
								#if totalCounter[0] + totalCounter[1] > 1:
								#	fftArr[0] = x_gpu_fft[0].get() / totalCounter[0]
								#	fftArr[1] = x_gpu_fft[1].get() / totalCounter[1]
								#print('TOOK ' + str(round((time.time() - timeBlah), 6)) + 's TO COPY DATA FROM GPU')
								#time.sleep(2)
								#if totalIts != 9 and totalIts % 100 == 9:
								#	print('\n\nDONE WITH ' + str(totalIts - 9))
								#if totalIts != 1 and totalIts % 10009 == 1: # and (totalIts % 1) == 0:
									#print('TOTAL COUNTER: ' + str(totalCounter))
									#fftData = x_gpu_fft[1].get() / totalCounter[1]
									#currentTime = datetime.now().strftime("%H-%M-%S")
									#baseFileName = 'ArduinoInterferenceTest_11-14-21_Off.txt'
									#baseFileName = 'SwitchingTest_0_EnableOn_ClosedTop_2GHz_7-21-21_100Avg_SwitchingRelay_Amp50OnZeroANDShortOnOne_20dBAmp_n50dBmSignal_ShortDelay'
									#baseFileName = 'JosephTest_0_600MHz_100Each_FFTPow24_BoxClosed_10-6-21_1000Avg_SwitchingTermAnd360Term40dBAmp_FFTTest'
								#	baseFileName = 'CalibrationTest_0_600MHz_FFTPow24_BoxClosed_2-5-22_5000Avg_BNCTermAndBNCTerm_10dBAtt80dBGain_FFTTest'
								#	with open(baseFileName + '_Sub1.txt', 'w') as f:
								#with open('AttenTest_0And31_EnableOn_QChannel_ClosedTop_2GHz_6-21-21_Term_EveryOther_Sub1.txt', 'w') as f:
								#		for fftVal in fftData:
								#			f.write(str(fftVal) + '\n')
								#	print('WROTE FILE ' + str(totalIts))
								
								#	fftData = x_gpu_fft[0].get() / totalCounter[0]
									#currentTime = datetime.now().strftime("%H-%M-%S")
								
								#	with open(baseFileName + '_Sub2.txt', 'w') as f:
									#with open('AttenTest_0And31_EnableOn_QChannel_ClosedTop_2GHz_6-21-21_Term_EveryOther_Sub2.txt', 'w') as f:
								#		for fftVal in fftData:
								#			f.write(str(fftVal) + '\n')
									
								#	fftData = x_gpu_fft[2].get() / totalCounter[0]

								#	with open(baseFileName + '_Ratio.txt', 'w') as f:
										#with open('AttenTest_0And31_EnableOn_QChannel_ClosedTop_2GHz_6-21-21_Term_EveryOther_Sub2.txt', 'w') as f:
								#		for fftVal in fftData:
								#			f.write(str(fftVal) + '\n')
								#end = copyStream.record()

								#print('Asynchronous Device to Host / Host to Device (ms)')
								#cp.cuda.Device().synchronize()
								#print(cp.cuda.get_elapsed_time(start, end))
							
						
						#print('TOOK ' + str(round((time.time() - timeBlah), 6)) + 's TO COMPLETE FFT')
					currentFFT = currentFFT + 1
				else:
					#prod = 1
					filled = True
					for row in writeStatus:
						for col in row:
							if not(col):
								filled = False
					if filled:
						switched = True
						print('\nSTUCK HERE ' + str(totalAcqs) + '\n')
						if startArr[0] and switched:
							totalAcqs = totalAcqs + 1
						if totalAcqs < 3:
							for x in range(ROACH_SETUP.NUM_BUFFERS):
								for y in range(ROACH_SETUP.NUM_FFT):
									writeStatus[x][y] = 0
									#writeStatus[x][y][1] = 0
						else:
							firstTime = False
							currentFFT = 0
							currentBuf = 0
							print('SLEEPING FOR A SEC...\n\n')
							for x in range(ROACH_SETUP.NUM_BUFFERS):
								for y in range(ROACH_SETUP.NUM_FFT):
									writeStatus[x][y] = 1
									#writeStatus[x][y][1] = 1
									#time.sleep(5)
							time.sleep(2)
							writeStatus[0][0] = 0
							#writeStatus[0][0][1] = 0
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
	sharedMemoryData = shared_memory.SharedMemory(create=True, size=int((ROACH_SETUP.PAYLOAD_LEN)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.NUM_FFT*ROACH_SETUP.EXTRA_MULT))
	# # Now create a NumPy array backed by shared memory
	dataArr = np.ndarray((ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN)*ROACH_SETUP.DATA_ITS * 8 * ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('i1'), buffer=sharedMemoryData.buf)
	#dataArr[:] = createArr[:]  # Copy the original data into shared memory
	return (sharedMemoryData, dataArr)


if __name__ == '__main__':
	# ps -A | grep python | awk '{print $1}'
	# sudo kill -9 $(ps -A | grep python | awk '{print $1}')
	roach = '192.168.40.76'
	lh=corr.log_handlers.DebugLogHandler()
	logger = logging.getLogger(roach)
	logger.addHandler(lh)
	logger.setLevel(10)
	fpga = corr.katcp_wrapper.FpgaClient(roach, logger=logger)
	time.sleep(1)

	#ROACH_funcs.progROACH(fpga)

	try:
		startTime = time.time()
		#windowData = []
		#windowFile = open('kaiser_2E26_order14', 'r')
		#windowData.append(np.fromfile(windowFile, dtype = 'd'))
		#windowFile = open('kaiser_2E25-almost-_order14', 'r')
		#windowData.append(np.fromfile(windowFile, dtype = 'd'))
		#windowFile = open('kaiser_2E24-almost-_order14', 'r')
		#windowData.append(np.fromfile(windowFile, dtype = 'd'))
		#windowFile = open('kaiser_2E23-almost-_order14', 'r')
		#windowData.append(np.fromfile(windowFile, dtype = 'd'))
		#windowFile = open('kaiser_2E22-almost-_order14', 'r')
		#windowData.append(np.fromfile(windowFile, dtype = 'd'))
		write_mem = shared_memory.SharedMemory(create=True, size = ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.NUM_FFT)
		writeStatus = np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT), dtype='B', buffer=write_mem.buf)
		writeStatus = [[0]*len(inner) for inner in writeStatus]

	
		endTime = time.time()
		#print('IT TOOK ' + str(round(endTime - startTime, 3)) + 'S TO READ WINDOW FILE IN')

		#circBuf = []

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
	
		save_mem = shared_memory.SharedMemory(create = True, size = 2 * 2**(ROACH_SETUP.FFT_POW-1) * 4)
		saveArr = np.ndarray((2, 2**(ROACH_SETUP.FFT_POW-1)), dtype = np.dtype('float32'), buffer=save_mem.buf)
		
		saveArr[0] = np.zeros(2**(ROACH_SETUP.FFT_POW - 1), dtype = np.dtype('float32'))
		saveArr[1] = np.zeros(2**(ROACH_SETUP.FFT_POW - 1), dtype = np.dtype('float32'))

		
		end_mem = shared_memory.SharedMemory(create = True, size = 2)
		killSwitch = np.ndarray((1,), dtype = 'B', buffer = end_mem.buf)
		#time_stamp_mem = shared_memory.SharedMemory(create = True, size = 2**ROACH_SETUP.FFT_POW*4)
		#timeStampArr = np.ndarray((2**ROACH_SETUP.FFT_POW,), dtype = np.dtype('float32'), buffer = time_stamp_mem.buf)
		
		#handshake_mem = shared_memory.SharedMemory(create = True, size = 2)
		#handshakeArr = np.ndarray((2,), dtype = np.dtype('B'), buffer = handshake_mem.buf)
		#handshakeArr[0] = 0
		#handshakeArr[1] = 0
		adc_mem = shared_memory.SharedMemory(create=True, size=int((ROACH_SETUP.PAYLOAD_LEN)*ROACH_SETUP.DATA_ITS*8*ROACH_SETUP.NUM_FFT*ROACH_SETUP.NUM_BUFFERS*ROACH_SETUP.EXTRA_MULT))
		adcArr = np.ndarray((ROACH_SETUP.NUM_BUFFERS, ROACH_SETUP.NUM_FFT, int((ROACH_SETUP.PAYLOAD_LEN)*ROACH_SETUP.DATA_ITS * 8 * ROACH_SETUP.EXTRA_MULT),), dtype=np.dtype('i1'), buffer=adc_mem.buf)
		
	

		rigolQueue = mp.Queue()
		plotQueue = mp.Queue()

		startTime = time.time()
		
		for j in range(1):
			#ctx = mp.get_context('spawn')
			eth1 = mp.Process(target = gbe0, name = 'gbe0', args = (adc_mem, write_mem, start_mem, end_mem))
			fftProc = mp.Process(target = performFFT, name = 'performFFT', args = (adc_mem, write_mem, start_mem, fft_mem, save_mem, end_mem))
			writeProc = mp.Process(target = writeDataNew, name = 'writeDataNew', args = (save_mem, rigolQueue, end_mem, plotQueue))
			rigolProc = mp.Process(target = takeRigolData, name = 'takeRigolData', args = (rigolQueue, end_mem))
			plotProc = mp.Process(target = plotCurrentAcq, name = 'plotCurrentAcq', args = (save_mem, end_mem, plotQueue))
			print('ON ITERATION ' + str(j))
				
			
			#fftProc.start()
			print('SLEEPING FOR A BIT...')
			time.sleep(10)
			print('DONE SLEEPING')			
			ROACH_funcs.enableROACH(fpga)
			
			#writeProc.start()
			eth1.start()
			fftProc.start()
			rigolProc.start()
			writeProc.start()
			plotProc.start()
			#time.sleep(10)
			time.sleep(1)
			#subprocess.call(['python2', './enableOutput.py'])
			#for i in range(10):
				#ROACH_funcs.resetFIFO(fpga)
			#	print(ROACH_funcs.resetFIFO(fpga))
			#	time.sleep(0.01)
			#	print(ROACH_funcs.resetFIFO(fpga))
			#	print(ROACH_funcs.resetFIFO(fpga))



			startArr[0] = True

			eth1.join()
			fftProc.join()
			writeProc.join()
			rigolProc.join()
			plotProc.join()

			endTime = time.time()
			#allocateFFTMem(circBuf[0][0].name, FFTNames, windowDataGPU)
			#fft_proc_1 = mp.Process(target = allocateFFTMem, name = 'fftROACH_SETUP', args = (circBuf[0][0].name, FFTNames, windowData, ([write_mem_0[x].name for x in range(len(write_mem_0))], [write_mem_1[x].name for x in range(len(write_mem_1))])))
			
		#for i in range(ROACH_SETUP.NUM_BUFFERS):
		#	circBuf[i][0].close()
		#	circBuf[i][0].unlink()
		#	write_mem[i].close()
		#	write_mem[i].unlink()
		write_mem.unlink()
		write_mem.close()
		adc_mem.unlink()
		adc_mem.close()

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
		save_mem.unlink()
		save_mem.close()
		end_mem.unlink()
		end_mem.close()
		print('THAT TOOK ' + str(round(endTime - startTime, 3)) + 'S TO COMPLETE')
		print('TOTAL ACTIVE THREADS: ' + str(threading.active_count()))
		sys.exit(1)

	except KeyboardInterrupt:
		print('INTERRUPTED')
		sock0.close()
		eth1.terminate()
		fftProc.terminate()
		writeProc.terminate()
		rigolProc.terminate()
		plotProc.terminate()
		write_mem.unlink()
		write_mem.close()
		adc_mem.unlink()
		adc_mem.close()
		write_index_mem.close()
		write_index_mem.unlink()
		start_mem.unlink()
		start_mem.close()
		fft_mem.close()
		fft_mem.unlink()
		save_mem.unlink()
		save_mem.close()
		end_mem.unlink()
		end_mem.close()
	except SystemExit:
		sys.exit(1)
	except Exception as e:
		print(e)