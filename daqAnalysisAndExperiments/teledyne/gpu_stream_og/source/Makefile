example: CFLAGS +=\
				$(shell pkg-config --cflags libadq cuda-11.7) \
				-DLINUX -g \
        -I ./gdrdrv \
        -I $(shell pkg-config --variable=cudaroot cudart-11.7)/include \
        -I $(shell pkg-config --variable=cudaroot cudart-11.7)/targets/x86_64-linux/include

example: LIBS +=$(shell pkg-config --libs libadq cuda-11.7 nvrtc-11.7 cudart-11.7)
				

example: streaming_nvidia.o gdrapi.o
	 $(CXX) $(CFLAGS)  -o streaming_nvidia streaming_nvidia.o gdrapi.o -lpthread  $(LIBS)

streaming_nvidia.o: streaming_nvidia.cpp Settings.h
	 $(CXX) $(CFLAGS)   -Wall -c streaming_nvidia.cpp $(LIBS) 

gdrapi.o: gdrapi.c gdrapi.h
	$(CXX) $(CFLAGS)  -c gdrapi.c $(LIBS)
	 

clean:
	rm -f gdrapi streaming_nvidia  *.o
