from arduinoClass import arduino
from libreVNA import libreVNA
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import datetime

#Must be in the same directory as libreVNA
#libreVNA GUI must be open!

# Create the control instance
vna = libreVNA('localhost', 19542)
stepArduino = arduino()



# Quick connection check (should print "LibreVNA-GUI")
print(vna.query("*IDN?"))

# Make sure we are connecting to a device (just to be sure, with default settings the LibreVNA-GUI auto-connects)
vna.cmd(":DEV:CONN")

dev = vna.query(":DEV:CONN?")
if dev == "Not connected":
    print("Not connected to any device, aborting")
    exit(-1)
else:
    print("Connected to "+dev)
# Simple trace data extraction

#init
totalSteps = 100
stepsToTake = 4800/totalSteps
freqPoints = 4501
dwellTime = 5 #[seconds] Time to sleep after moving stirrer
df = pd.DataFrame(columns=["stirPos", "Freq_MHz", "S11", "S12", "S22"])

time.sleep(3) #arduino is unhappy getting a command right after contact is made. 
stirPos = int(0)
print('Sending Arduino `zero` at: ' + str(datetime.datetime.now()))
stepArduino.writeCmd('zero')
time.sleep(dwellTime)

#take data
for i in range(totalSteps):
    #VNA settings and start sweep
    if 1: #turn off for testing motion. Kills VNA and saving as df
        #print("Setting up the VNA sweep")
        vna.cmd(":DEV:MODE VNA")
        vna.cmd(":VNA:SWEEP FREQUENCY")
        vna.cmd(":VNA:STIM:LVL 0")
        vna.cmd(":VNA:ACQ:IFBW 2500") #IF bandwidth. Basically RBW
        vna.cmd(":VNA:ACQ:AVG 1")
        vna.cmd(":VNA:ACQ:SINGLE TRUE")
        vna.cmd(":VNA:ACQ:POINTS " + str(freqPoints)) #data points. max = 4501
        vna.cmd(":VNA:FREQuency:START " + str(int(250e6)))
        vna.cmd(":VNA:FREQuency:STOP " + str(int(1050e6)))

        # wait for the sweep to finish
        print("Waiting for the VNA sweep to finish... Time is: " + str(datetime.datetime.now()))
        while vna.query(":VNA:ACQ:FIN?") == "FALSE":
            time.sleep(.1)

        # Get S data and calculate the frequency axis (in MHz).
        # Retrieve and process S data
        sQuery = vna.query(":VNA:TRACE:DATA? S11")
        S11 = vna.parse_trace_data(sQuery)
        freqs = np.array([data[0] / 1e6 for data in S11])  # Convert frequency to MHz
        sParamComp_S11 = np.array([data[1] for data in S11])
        sQuery = vna.query(":VNA:TRACE:DATA? S12")
        S12 = vna.parse_trace_data(sQuery)
        sParamComp_S12 = np.array([data[1] for data in S12])
        sQuery = vna.query(":VNA:TRACE:DATA? S22")
        S22 = vna.parse_trace_data(sQuery)
        sParamComp_S22 = np.array([data[1] for data in S22])

        if (len(freqs) != freqPoints): #check if VNA shipped wrong number of frequency points. Should clean this up so it's a function.
            print("VNA error! incorrect number of frequency points received. trying again")
            print("Setting up the VNA sweep again")
            vna.cmd(":DEV:MODE VNA")
            vna.cmd(":VNA:SWEEP FREQUENCY")
            vna.cmd(":VNA:STIM:LVL 0")
            vna.cmd(":VNA:ACQ:IFBW 2500") #IF bandwidth. Basically RBW
            vna.cmd(":VNA:ACQ:AVG 1")
            vna.cmd(":VNA:ACQ:SINGLE TRUE")
            vna.cmd(":VNA:ACQ:POINTS " + str(freqPoints)) #data points. max = 4501
            vna.cmd(":VNA:FREQuency:START " + str(int(250e6)))
            vna.cmd(":VNA:FREQuency:STOP " + str(int(1050e6)))

            # wait for the sweep to finish
            print("Waiting for the VNA sweep to finish...")
            while vna.query(":VNA:ACQ:FIN?") == "FALSE":
                time.sleep(.1)

            # Get S data and calculate the frequency axis (in MHz).
            # Retrieve and process S data
            sQuery = vna.query(":VNA:TRACE:DATA? S11")
            S11 = vna.parse_trace_data(sQuery)
            freqs = np.array([data[0] / 1e6 for data in S11])  # Convert frequency to MHz
            sParamComp_S11 = np.array([data[1] for data in S11])
            sQuery = vna.query(":VNA:TRACE:DATA? S12")
            S12 = vna.parse_trace_data(sQuery)
            sParamComp_S12 = np.array([data[1] for data in S12])
            sQuery = vna.query(":VNA:TRACE:DATA? S22")
            S22 = vna.parse_trace_data(sQuery)
            sParamComp_S22 = np.array([data[1] for data in S22])

        # Create a temporary DataFrame for this sweep.
        # We include a "Sweep" column so you know which measurement these points belong to.
        temp_df = pd.DataFrame({
            "stirPos": stirPos,
            "Freq_MHz": freqs,
            "S11": sParamComp_S11,
            "S12": sParamComp_S12,
            "S22": sParamComp_S22
        })
        
        # Append this sweep's results to the overall DataFrame.
        df = pd.concat([df, temp_df], ignore_index=True)

        
    #print('VNA sweep done, moving stirrer')
    time.sleep(.1)
    if i != (totalSteps - 1):
        stirPos = stepArduino.writeCmd('step ' + str(stepsToTake))
    #print(f'Stirrer in position. Sleeping for {dwellTime} seconds')
    time.sleep(dwellTime)

# Save the complete DataFrame to a CSV file.
#df.to_csv("antennaFacingEachOther_200Step_5secDwell_250_1050MHz_4501freq_2p5kHzRBW_avg1_4_16_25.csv", index=False)
path = ""
filename = "2portRCTest_250_1050MHz_200steps_DeltaQ_nopanel_4_28_25.csv"
df.to_csv(path + filename, index=False)
print("Data saved to vna_data.csv")
stepArduino.writeCmd('power off')