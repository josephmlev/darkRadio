import numpy as np

# Generate random values
roll = np.around(np.random.uniform(0, np.pi/2, 18**3), 3)
yaw = np.around(np.random.uniform(0, np.pi/2, 18**3), 3)
pitch = np.around(np.random.uniform(0, np.pi*2, 18**3), 3)

# Convert arrays to string with space-separated values
roll_str = " ".join(map(str, roll))
yaw_str = " ".join(map(str, yaw))
pitch_str = " ".join(map(str, pitch))

# Write data to file
with open('./data.txt', 'w') as f:
    f.write(f"roll {roll_str}\n")
    f.write(f"yaw {yaw_str}\n")
    f.write(f"pitch {pitch_str}\n")

print('done')