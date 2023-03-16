import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


dataFile = './100KThermistorTable.csv'

data = pd.read_csv(dataFile)

temps = np.flip(np.asarray(data['Temp']))
resArr = np.flip(np.asarray(data['Center']))


while True:
	resVal = float(input(u'Give a resistance value in k' + '\u03A9' + ': '))
	fitVal = np.interp(resVal, resArr, temps)
	print(u'TEMP AT ' + str(resVal)  + 'k\u03A9 : ' + str(round(fitVal, 3)) + u'\N{DEGREE SIGN}C = ' + str(round(fitVal*1.8+32, 3)) + u'\N{DEGREE SIGN}F')

