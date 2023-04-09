import ctypes
import pathlib
import os


if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "streaming_nvidia.o"
    c_lib = ctypes.CDLL(libname)
