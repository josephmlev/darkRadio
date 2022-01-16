import serial as ser 
import time 
port = '/dev/cu.usbmodem14101'
ser = ser.Serial(port, 9600, timeout=1)


def getTemp():
	startChar = '<'
	endChar = '>'
	abortChar = '^'
	listenForStart = True
	listenForData = True
	currentTime = time.time()
	MAX_TIME = 3000

	while listenForStart == True:
		holder = ser.readline()
		readChar = (holder.decode("utf-8").strip())
		if time.time() - currentTime > MAX_TIME:
			ser.write(abortChar)
		elif readChar == startChar:
			ser.write(endChar.encode())
			listenForStart = False
		else:
			print('DID NOT HEAR START...')
			#time.sleep(1)
	 

	while listenForData == True:
		holder = ser.readline()
		ser.write(ser.write(endChar.encode()))
		listenForData = False
	return (holder.decode("utf-8").strip())


while True:
	print('THE TEMPERATURE IS: ' + getTemp())
	ser.flushInput()
	ser.flushOutput()
	time.sleep(0.1)