import serial

ser = serial.Serial('/dev/ttyACM0', 115200)

while True:
    dataIn = input('enter to toggle')
    ser.write(b'0b0')
    print(0)
    dataIn = input('enter to toggle')
    ser.write(b'0b1')
    print(1)
