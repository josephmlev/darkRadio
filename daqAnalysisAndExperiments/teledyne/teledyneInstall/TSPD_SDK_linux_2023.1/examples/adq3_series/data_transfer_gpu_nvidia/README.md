# README

This example demonstrates streaming to a Nvidia GPU using Nvidia GPUDirect
RDMA. GPUDirect RDMA is only supported by professional GPUs, specified as
GPUDirect in datasheet.

# Default behavior

The *default* behavior of the example application is to receive 1000 buffers on
each channel using an integrated periodic trigger. The input signal is a ramp
generated by a *test pattern generator*. To acquire ADC data, set
`TEST_PATTERN_SOURCE` to `ADQ_TEST_PATTERN_SOURCE_DISABLE` (disabled) in the
header file `Settings.h`.

# Multiple digitizers
The example supports an arbitrary number of ADQs, controlled by `NOF_DIGITIZERS`
in the header file `Settings.h`. The same configuration will be applied to all
active digitizers.

# Structure

The application can be broken down into six stages:

1. GPU identification and initialization
2. ADQ identification
3. ADQ configuration (configuration parameters are located in the header file
   `settings.h`)
4. GPU buffer allocation
5. Data acquisition
6. Summary and cleanup

The source code includes comments to highlight and explain important actions.

# Prerequisites
* Nvidia drivers
* Cuda

# Compilation
1. Compile the included (modified) gdr_copy
```
examples/libs/gdrcopy$ make config driver lib lib_install && ldconfig
```
2. Load gdrdrv, must be done after every system restart
```
examples/libs/gdrcopy$ ./insmod.sh
```
3. Build example, specify your installed Cuda version
```
data_transfer_gpu_nvidia/source$ make cuda_version=11.8
```
4. Run example
```
data_transfer_gpu_nvidia/source$ ./streaming_nvidia
```

# Performance

Sustained throughput above 7,4 GB/s has been measured in our test system with
this demo. Maximum throughput is not guaranteed and will depend on:
* Hardware
* System settings
* Transfer buffer size (bigger is better), usually 1-2 MB is enough to reach
  maximum throughput
* Record length (512 < samples may have some impact)

These dependencies may change in future releases

# Performance troubleshooting
If you are experiencing low or unstable performance start with verifying the GPU
link.
```
$ lspci -vv | grep VGA.*NVIDIA -A 40 | grep LnkSta
```
In the resulting printout you can see the link speed and width. If the width is
lower than expected (usually x16) refer to your motherboard manual. If speed is
lower than expected (8GT/s for PCIe gen3, 16GT/s for PCIe gen4), try disabling
the adaptive link speed with `powermizermode`.

## Setting powermizermode through GUI
Open Nvidia X Server settings go to Powermizermode select Preferred mode
`Prefer Maximum Performance`

## Setting powermizermode through CLI
```
$ nvidia-settings --assign GPUPowerMizerMode=1
```

Verify that the GPU link has the expected speed
```
$ lspci -vv | grep VGA.*NVIDIA -A 40 | grep LnkSta
```