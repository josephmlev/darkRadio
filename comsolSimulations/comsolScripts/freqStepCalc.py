fStart  = 50
fStop   = 400
Q       = 500
f       = fStart
fList   = [f]
while f < fStop:
    step = f/Q
    fList.append(f+step)
    f   += step


print(fList)
print(len(fList))