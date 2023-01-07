import holoviews as hv 
import multiprocessing as mp
import numpy as np 
import pandas as pd
import screeninfo
import panel as pn
import serial
import time
import os
import subprocess
from collections import OrderedDict as odict

from holoviews import opts
from holoviews.streams import Stream, Pipe
from holoviews.operation.datashader import rasterize, ResamplingOperation
from holoviews.operation import decimate
import psutil


def currentCount(path):
	if not(path):
		return 0, False	
	elif os.path.exists(path):
		return 0, False
	else:
		try:
			with open(path, 'r') as f:
				holder = int(f.readline())
		except:
			print('BAD FILE FORMAT NOT SAVING AVERAGES')
			return 0, False 
	return holder, True

def outputToList(x):
	return x.decode('ascii').split('\n')[:-1]

class VisPanel(mp.Process):
	def __init__(self, clockRate = 2E3, #In MHz
				 FFTPow = 20, 
				 avgPerWrite = 32,
				 saveCountFile = None,
				 maxSamples = 10000,
				 tempBufferLength = 100):
		
		hv.extension("bokeh")
		self.plotWidth = int(0.6 * screeninfo.get_monitors()[0].width)
		self.plotHeight = int(0.2 * screeninfo.get_monitors()[0].height)
		self.avgPerWrite = avgPerWrite
		self.clockRate = clockRate 
		self.FFTPow = FFTPow
		self.freqArr = np.asarray(range(2**(self.FFTPow - 1) + 1))*self.clockRate/(2**self.FFTPow)
		self.saveCountFile = saveCountFile
		self.saveExists = currentCount(self.saveCountFile)
		self.currentCount = (self.saveExists)[0]
		self.remoteRoach = mp.Queue()
		self.roachData = np.zeros((2, 2**(self.FFTPow - 1) + 1))
		(self.roachData)[0] = np.random.rand(2**(self.FFTPow - 1) + 1)
		(self.roachData)[0][0] = 0
		(self.roachData)[1] = np.random.rand(2**(self.FFTPow - 1) + 1)
		(self.roachData)[1][0] = 0

		self.maxSamples = maxSamples 
		
		self.tempData = mp.Queue()
		self.tempBufferLength = tempBufferLength

		self.__cpu_stream = hv.streams.Pipe(self.get_cpu_data(), index=False)
		self.__mem_stream = hv.streams.Buffer(self.get_mem_data())
		self.__gpu_stream = hv.streams.Buffer(self.getGPUMemory())
		self.__temp_stream = hv.streams.Buffer(self.getTempData(), length = self.tempBufferLength)
		self.__roach_stream = hv.streams.Pipe(self.getRoachData(), index=False)
		self.__count_stream = hv.streams.Pipe(self.genCount())
		
		# Define DynamicMaps and display plot
		self.__cpu_dmap = hv.DynamicMap(self.cpu_box, streams=[self.__cpu_stream])
		self.__mem_dmap = hv.DynamicMap(self.mem_stack, streams=[self.__mem_stream])
		self.__gpu_dmap = hv.DynamicMap(self.gpu_stack, streams=[self.__gpu_stream])
		self.__roach_dmap = hv.DynamicMap(self.genVis, streams=[self.__roach_stream])
		self.__count_dmap = hv.DynamicMap(self.getCount, streams = [self.__count_stream])
		self.__temp_dmap = hv.DynamicMap(self.temp_stack, streams = [self.__temp_stream])


		#(datashade(roach_dmap, precompute = False, line_width = 2).opts(xlabel = 'Frequency (MHz)', ylabel = 'Amplitude (dBm)', width = 1000, height = 400, xlim = (0, 1000)))
		self.__color_cycle = hv.Cycle(['red', 'green'])

		# 	datashade(lineoverlay, pixel_ratio=2, line_width=4, aggregator=ds.by('freqs', ds.count())).opts(width=800)
		# (rasterize(roach_dmap, precompute = False, line_width = 2).opts(xlabel = 'Frequency (MHz)', ylabel = 'Amplitude (dBm)', width = 1000, height = 400, xlim = (0, 1000))).relabel('ROACH Data')
		self.__callback = pn.io.PeriodicCallback(callback=self.cb, period=500)
		self.__callbackRoach = pn.io.PeriodicCallback(callback=self.cbRoach, period=1000)

		self.__server = None

		self.__firstPlot = True
		self.__minX = 0 
		self.__maxX = (self.clockRate)/2
		self.__minY = -1
		self.__maxY = 1 
		self.__app = pn.Column(pn.Row(self.__gpu_dmap, self.__mem_dmap), pn.Row(self.__cpu_dmap), pn.Row((decimate(self.__roach_dmap) * self.__count_dmap).opts(ylim = (self.__minY, self.__maxY), xlabel = 'Frequency (MHz)', ylabel = 'Amplitude (dBm)', width = 1000, height = 400).relabel('ROACH Data')), pn.Row(self.__temp_dmap))

		decimate.max_samples=10000

	# Define functions to get memory and CPU usage
	
	#def setRoachData(self, data):
	#	print('****************FLOOMFAS!!!!!****************\n\n\n\n\n\n\n\n\n\n')
	#	self.roachData = data
	#	self.currentCount += self.avgPerWrite
	
	#def setTempData(self, data):
	#	print('****************FRESIANS!!!!!****************\n\n\n\n\n\n\n\n\n\n')
	#	print(str(data))
	#	(self.tempData).append(float(data))
	#	print(self.tempData)

	def get_mem_data(self):
		vmem = psutil.virtual_memory()
		df = pd.DataFrame(dict(free=vmem.free/2**30,
							   used=vmem.used/2**30),
						  index=[pd.Timestamp.now()])
		#print(df*100)
		return df

	def getTempData(self):
		print('\n\n\n\n\n\n\nTEMP DATA IS: ' + str(self.tempData) + '\n\n\n\n\n\n\n')
		df = pd.DataFrame(dict(temp=None, indices=[pd.Timestamp.now()]))
		while not((self.tempData).empty()):
			df = pd.concat((df, pd.DataFrame(dict(temp=(self.tempData).get(), indices=[pd.Timestamp.now()]))))			
		#if not(len(df)):
		#	df = pd.DataFrame(dict(temp=None, indices=[None]))
		print(df)
		return df 

	def get_cpu_data(self):
		cpu_percent = psutil.cpu_percent(percpu=True)
		df = pd.DataFrame(list(enumerate(cpu_percent)), columns=['CPU', 'Utilization'])
		df['time'] = pd.Timestamp.now()
		return df


	def getGPUMemory(self):
		usedMemoryCommand = 'nvidia-smi --query-gpu=memory.used --format=csv'
		freeMemoryCommand = 'nvidia-smi --query-gpu=memory.free --format=csv'
		try:
			usedMemory = outputToList(subprocess.check_output(usedMemoryCommand.split(),stderr=subprocess.STDOUT))[1]
			freeMemory = outputToList(subprocess.check_output(freeMemoryCommand.split(),stderr=subprocess.STDOUT))[1]
		except subprocess.CalledProcessError as e:
			raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		
		df = pd.DataFrame(dict(free = float(freeMemory.split()[0]), used = float(usedMemory.split()[0])), index = [pd.Timestamp.now()])

		return df
		
	# Define DynamicMap callbacks returning Elements
	def mem_stack(self, data):
		data = pd.melt(data, 'index', var_name='Type', value_name='Usage')
		areas = hv.Dataset(data).to(hv.Area, 'index', 'Usage')
		return hv.Area.stack(areas.overlay()).opts(ylabel = 'Memory (GB)', xlabel = ('Time'), width=int(self.plotWidth/2), height=self.plotHeight, ylim = (0, 130), framewise = True).relabel('RAM')

	def gpu_stack(self, data):
		#print(data)
		data = pd.melt(data, 'index', var_name='Type', value_name='UsageGPU')
		areas = hv.Dataset(data).to(hv.Area, 'index', 'UsageGPU')
		return (hv.Area.stack(areas.overlay()).opts(ylabel = 'Memory (MB)', xlabel = ('Time'), width=int(self.plotWidth/2), height=self.plotHeight, ylim=(0, 8200), framewise = True)).relabel('GPU Memory')


	def temp_stack(self, data):
		points = list(zip(data.indices, data.temp))
		scatter = hv.Scatter(points)
		return scatter.opts(tools = ['hover'], ylabel = 'Temperature (K)', xlabel = ('Time'), size = 15, width=1000, height=400).relabel('Temperature')

	def cpu_box(self, data):
		sortedData = data.sort_values(by=['CPU'])
		sortedData['CPU'] = sortedData['CPU'].astype(str)
		colorTest = []
		return hv.Bars(sortedData, kdims = ['CPU'], vdims = ['Utilization']).opts(color = 'Utilization', cmap = 'coolwarm', xlabel = 'Core #', ylabel = 'Utilization (%)', width=self.plotWidth, height=self.plotHeight, xlim = (0, 48), ylim=(-1, 100)).relabel('CPU Usage')

	def getRoachData(self):
		if not((self.remoteRoach).empty()):
			self.currentCount = self.currentCount + self.avgPerWrite
			if (self.saveExists)[1]:
				with open(self.saveCountFile, 'w') as f:
					f.write(str(self.currentCount) + '\n')
			while not((self.remoteRoach).empty()):
				self.roachData = (self.remoteRoach).get()			
		
		print((self.roachData)[0][:100])
		
		df = pd.DataFrame({'freqs':self.freqArr, 'data0':((self.roachData)[0]), 'data1':((self.roachData)[1])})
		return df

	def genVis(self, data):
		#points0 = list(zip(self.freqArr, (self.roachData)[0]))
		#points1 = list(zip(self.freqArr, (self.roachData)[1]))
		#curves = (hv.NdOverlay({'Ant': hv.Curve(points0), 'Term':hv.Curve(points1)}))
		print('PLOTTING DATA MAYBE?')
		if self.__firstPlot and (self.roachData)[1][0] != 0:
			self.__maxX = (self.clockRate) / 2
			dataMin = min((self.roachData)[0])
			dataMax = max((self.roachData)[0])
			self.__minY = dataMin - (dataMax - dataMin)*0.2
			self.__maxY = dataMax + (dataMax - dataMin)*0.2
			#opts.defaults(opts.NdOverlay(ylim = (self.__minY, self.__maxY)))
			#opts.defaults(opts.Text(xlim = (0, self.__maxX*1.05), ylim = (self.__minY, self.__maxY)))
			self.__firstPlot = False

		print('X-RANGE: ' + str(self.__minX) + '-' + str(self.__maxX))
		print('Y-RANGE: ' + str(self.__minY) + '-' + str(self.__maxY))

		#curves = (hv.NdOverlay({'Ant': (hv.Curve(data, 'freqs', 'data0').opts(alpha = 0.5)).redim.range(freqs=(self.__minX, self.__maxX), data0 = (self.__minY, self.__maxY)), 'Term':(hv.Curve(data, 'freqs', 'data1').opts(alpha = 0.5)).redim.range(freqs=(self.__minX, self.__maxX), data1 = (self.__minY, self.__maxY))}))
		
		curves = (hv.NdOverlay({'Ant': (hv.Curve(data, 'freqs', 'data0').opts(alpha = 0.5)), 'Term':(hv.Curve(data, 'freqs', 'data1').opts(alpha = 0.5))}))
		#obj.opts.clear().
		p = hv.HoloMap(curves)
		p.opts(opts.NdOverlay(framewise = True))
		p.opts(opts.Curve(framewise = True))

		#if not(self.__firstPlot): 
		#	overlayDim = curves.dimensions()[1]
		#	overlayDim.range = (None, None)
		#	overlayDim = curves.dimensions()[0]
		#	overlayDim.range = (self.__minX, self.__maxX)
		#	return curves.opts(framewise = True)
		
		return curves

	def genCount(self):
		#self.currentCount += self.avgPerWrite
		#if self.saveExists:
		#	with open(self.saveFile, 'w') as f:
		#		f.write(str(self.currentCount) + '\n')
		return self.currentCount

	def getCount(self, data):
		counter = hv.Text(100, (self.__maxY - 0.2*(self.__maxY - self.__minY)), 'Total Avg: ' + str(self.currentCount))
		print('FUCK YOU: ' + str((self.__maxY - 0.2*(self.__maxY - self.__minY))))
		return counter 
	
	def cb(self):
		self.__cpu_stream.send(self.get_cpu_data())
		self.__mem_stream.send(self.get_mem_data())
		self.__gpu_stream.send(self.getGPUMemory())


	def cbRoach(self):
		print('****************GHERKIN!!!!****************\n\n\n\n\n\n\n\n\n\n')
		self.__roach_stream.send(self.getRoachData())
		self.__count_stream.send(self.genCount())
		self.__temp_stream.send(self.getTempData())

	
	def beginPlot(self):	
		self.__callback.start()
		self.__callbackRoach.start()
		self.__server = pn.serve(self.__app, start=True, show=True)



		#self.__server.start()
		#server.show('/')
		#self.__server.stop()









