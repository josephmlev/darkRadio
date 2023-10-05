fStart  = 40
fStop   = 310
Q       = 500
f       = fStart
fList   = [f]
while f < fStop:
    step = f/Q
    fList.append(round(f+step,3))
    f   += step

import numpy as np
print(fList)
print(len(fList))
