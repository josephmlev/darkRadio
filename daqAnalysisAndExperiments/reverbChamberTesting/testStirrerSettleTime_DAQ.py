from arduinoClass import arduino
from libreVNA import libreVNA
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from datetime import datetime

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

print("Setting up the VNA sweep")
vna.cmd(":DEV:MODE VNA")
vna.cmd(":VNA:SWEEP FREQUENCY")
vna.cmd(":VNA:STIM:LVL 0")
vna.cmd(":VNA:ACQ:IFBW 10000") #IF bandwidth. Basically RBW
vna.cmd(":VNA:ACQ:AVG 1")
vna.cmd(":VNA:ACQ:SINGLE TRUE")
vna.cmd(":VNA:ACQ:POINTS 4001") #data points. max = 4501
vna.cmd(":VNA:FREQuency:START " + str(int(900e6)))
vna.cmd(":VNA:FREQuency:STOP " + str(int(920e6)))
# wait for the sweep to finish
print("Waiting for the VNA sweep to finish...")
while vna.query(":VNA:ACQ:FIN?") == "FALSE":
    time.sleep(.1)

df = pd.DataFrame(columns=["Sweep time", "Freq_MHz", "S11", "S12", "S22"])
totalSteps = 50
time.sleep(3) #arduino is unhappy getting a command right after contact is made. 
print('Sending Arduino `zero` now.')
stepArduino.writeCmd('zero')
startTime = time.time()
#take data
for i in range(totalSteps):
    #VNA settings and start sweep
    vna.cmd(":DEV:MODE VNA")
    vna.cmd(":VNA:SWEEP FREQUENCY")
    vna.cmd(":VNA:STIM:LVL 0")
    #vna.cmd(":VNA:ACQ:IFBW 10000") #IF bandwidth. Basically RBW
    #vna.cmd(":VNA:ACQ:AVG 1")
    #vna.cmd(":VNA:ACQ:SINGLE TRUE")
    #vna.cmd(":VNA:ACQ:POINTS 4001") #data points. max = 4501
    vna.cmd(":VNA:FREQuency:START " + str(int(900e6)))
    vna.cmd(":VNA:FREQuency:STOP " + str(int(920e6)))

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
        "Sweep time": (time.time()-startTime),
        "Freq_MHz": freqs,
        "S11": sParamComp_S11,
        "S12": sParamComp_S12,
        "S22": sParamComp_S22
    })
    
    # Append this sweep's results to the overall DataFrame.
    df = pd.concat([df, temp_df], ignore_index=True)
    print(f"time since start = {time.time() - startTime}")



# Save the complete DataFrame to a CSV file.
df.to_csv("stirrerSettleTime_900MHz.csv", index=False)
print("Data saved to vna_data.csv")