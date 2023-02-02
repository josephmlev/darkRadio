import numpy as np
import csv

# read the file
with open("allE_all_64pts.txt") as f:
    data = f.readlines()

num_cols = len(data[0].strip().split())
# get the number of positions
N = (num_cols - 12) // 4

# remove the header row
header = data[5]
data = data[6:]

# create a dictionary to store the means
means = {}



# loop through the rows in the data
for row in data:
    # split the row by whitespaces
    values = row.strip().split()
    freq = float(values[7])
    # create a list of lists to hold the values of E_i at each position
    pos_values = [[] for _ in range(4)]
    for i in range(12, len(values)):
        pos = (i-12) % (N)
        if pos != 0:
            e = (i-12) // (N)
            pos_values[e].append(float(values[i]) if values[i] != 'NAN' else np.nan)
    # calculate the mean for each E_i, ignoring NANs
    mean = [np.nanmean(e_values) for e_values in pos_values]
    # add the mean to the dictionary
    means[freq] = mean



# write the results to a new CSV file
with open('output.csv', 'w') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(['freq', 'E_0', 'E_1', 'E_2', 'E_3'])
    for freq, mean in means.items():
        writer.writerow([freq, *mean])


