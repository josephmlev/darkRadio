import numpy

a = numpy.random.rand(10,1000000)
numpy.savetxt("dummy.csv", a, delimiter=",")