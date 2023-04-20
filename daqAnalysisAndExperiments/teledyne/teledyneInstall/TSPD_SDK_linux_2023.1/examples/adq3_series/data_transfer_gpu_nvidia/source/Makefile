CFLAGS += $(shell pkg-config --cflags libadq cuda-$(cuda_version)) -DLINUX -g
LIBS += $(shell pkg-config --libs libadq cuda-$(cuda_version) nvrtc-$(cuda_version) \
	     cudart-$(cuda_version)) -lgdrapi


cudacheck:
ifdef cuda_version
	@if [ $(shell pkg-config --cflags cuda-$(cuda_version)) ]; then \
		$(MAKE) example; \
	else \
		echo "Can't find Cuda version $(cuda_version)"; \
	fi
else
	@echo "Please specify Cuda version, example: make cuda_version=11.8"
endif

example: streaming_nvidia.o
	 $(CXX) $(CFLAGS)  -o streaming_nvidia streaming_nvidia.o  $(LIBS)

streaming_nvidia.o: streaming_nvidia.cpp Settings.h helper_cuda.h helpers.h
	 $(CXX) $(CFLAGS)   -Wall -c streaming_nvidia.cpp $(LIBS)


clean:
	rm -f streaming_nvidia  *.o
