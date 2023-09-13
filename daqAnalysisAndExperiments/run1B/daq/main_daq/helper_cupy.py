# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
import cupy as cp

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaHostRegisterIoMemory = 4


def cudaGetDeviceProperties(gpu):
    return cp.cuda.runtime.getDeviceProperties(gpu)


def cudaGetDeviceCount():
    return cp.cuda.runtime.getDeviceCount()


def cudaSetDevice(idx):
    return cp.cuda.runtime.setDevice(idx)


def cudaMalloc(size):
    return cp.cuda.runtime.malloc(size)


def cudaMemcpy(dst, src, size, kind):
    return cp.cuda.runtime.memcpy(dst, src, size, kind)


def cudaHostRegister(ptr, size, flag):
    return cp.cuda.runtime.hostRegister(ptr, size, flag)


def cudaMemset(ptr, value, size):
    return cp.cuda.runtime.memset(ptr, value, size)


def cudaMemcpyAsync(dst, src, size, kind, stream):
    return cp.cuda.runtime.memcpyAsync(dst, src, size, kind, stream)


def cudaDeviceSyncrhonize():
    return cp.cuda.runtime.deviceSynchronize()


def loop_through_channels(nof_channels, funcs):
    for i in range(nof_channels):
        retvals = [f() for f in funcs]
    if 0 in retvals:
        assert 0


def printout(**kwargs):
    for k, v in kwargs.items():
        print("{0} = {1}".format(k, v))


class GpuBuffers:
    def __init__(self, size, nof_buffers=1):
        self.nof_buffers = nof_buffers
        self._buffers = [cp.zeros(size, dtype=cp.short) for i in range(nof_buffers)]

    def __setitem__(self, key, value):
        self._buffers[key] = value

    def __getitem__(self, idx):
        return self._buffers[idx]


class GpuStreams:
    def __init__(self, nof_streams=1):
        self.nof_streams = nof_streams
        self._streams = [cp.cuda.Stream(non_blocking=True) for i in range(nof_streams)]

    def __setitem__(self, key, value):
        self._streams[key] = cp.cuda.Stream()

    def __getitem__(self, idx):
        return self._streams[idx]
