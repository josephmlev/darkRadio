import numpy as np

for i in range(48):
    rand = np.random.uniform(0, 1, 10**8)
    np.save('./splitData/1e7RandomFloats0To1_file' + str(i), rand, allow_pickle=True)
    print(i)