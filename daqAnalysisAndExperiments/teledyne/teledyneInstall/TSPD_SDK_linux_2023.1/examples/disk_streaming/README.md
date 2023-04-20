# README

This example in C demonstrates how to configure a digitizer with FWDSU firmware and acquire data using both the data readout interface (digitizer to RAM) and the disk streaming interface (digitizer to NVMe disks).

# Prerequisites
   - ADQAPI and libadnvds installed.
   - Identify serial numbers of NVMe disks and update the fields of `DiskStorage disk_information` in `settings.h` (e.g. `serials`).
   - The NVMe disks have the correct driver bound.
     - (Windows) Use program NVMeBind.exe, part of the libadnvds installation.
     - (Linux) Use the script setup_adnvds.sh, provided in the release archive.
   - Make sure the digitizer is running the intended DSU7-firmware. (e.g. through `adqupdater`).

# Compiling

To compile the example application, please follow the instructions for your
operating system in one of the sections below.

## CMake (Windows and Linux)
1. Ensure `cmake` is installed on the system and run
   ```
   $ mkdir build && cd build
   $ cmake ..
   ```
   on Windows the platform and/or generator may have to be specified when using MSCV, e.g.
   ```
   > cmake -G "Visual Studio 15 2017" -A x64 ..
   ```

2. Run `cmake --build .` to compile the source code into a binary file.


## Bat-script (Windows only)

1. Open a command prompt and run
   ```
   > compile_x64.bat
   ```
   to compile the application into a binary.

   This step requires a valid installation of Microsoft Visual Studio with
   support for C99 or later versions. The `.bat` script assumes Visual Studio
   2017. Please adjust the `VCVARSALL_PATH` appropriately if a different version
   is installed.

## Notes on running the executable
(Linux) On most systems, as configured with the help of `setup_adnvds.sh`, the compiled program must be run with sudo permissions.
