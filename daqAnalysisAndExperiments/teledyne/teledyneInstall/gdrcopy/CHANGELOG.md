# Changelog

## [2.4] - 2023-09-19
- Various bug fixes in the test and benchmark applications.
- Prefix all applications with "gdrcopy\_".
- Introduce more unit tests in gdrcopy\_sanity.
- Introduce gdrcopy\_pplat benchmark application.
- Remove dependency on libcheck and libsubunit
- Introduce gdr\_get\_info\_v2.
- Introduce new copy algorithm for device mappings.
- Add support for NVIDIA BLUEFIELD-3.
- Add support for Linux kernel >= 6.3.
- Add support for SLES and OpenSUSE.
- Add support for systemd service on RHEL9.
- Relicense gdrdrv to Dual MIT/GPL.
- Fix bugs in gdrdrv when pinning two small buffers back-to-back.
- Add support for coherent platforms such as Grace-Hopper.
- Add support for Confidential Computing (CC).

## [2.3.1] - 2023-05-12
- Add a workaround for the GPL-compatibility issue when compile with CONFIG\_ARCH\_HAS\_CC\_PLATFORM on Linux kernel 5.18+.
- Fix error in init.d/gdrcopy due to missing /etc/rc.d/init.d/functions.

## [2.3] - 2021-07-27
- Remove automatically-generated build id links in rpm packages.
- Remove gdrcopy-kmod from the Requires field of the gdrcopy rpm package.
- Remove gdrdrv-dkms dependency enforcement from the gdrcopy deb package.
- Add libsubunit0 to the dependency list of the gdrcopy deb package.
- Add apiperf test.
- Revamp gdrdrv to fix race-condition bugs.
- Add an option to build kmod package.
- Split the gdrcopy deb package into meta, libgdrapi, and tests packages.
- Update the package maintainer.
- Various updates in README.

## [2.2] - 2021-02-01
- Add support for ARM64.
- Update various information on README.
- Improve Makefile.
- Add multi-arch support.
- Handle removal of HAVE\_UNLOCKED\_IOCTL in Linux kernel v5.9 and later.
- Prevent dpkg package creation to unnecessarily compile gdrdrv.
- Improve gdr\_open error message.
- Fix bug that prevents sanity from correctly summarizing failure.
- Add dkms support in kmod package.
- Handle the removal of kzfree in Linux kernel v5.10 and later.
- Improve small-size copy-to-mapping.

## [2.1] - 2020-08-07
- fix build problem on RHL8 kernels
- relax checks in gdrdrv to support multi-threading use cases
- fix fd leak in gdr\_open()
- introduce new copylat test
- remove CUDA RT dependency in tests
- assorted cleanups

## [2.0] - 2019-09-16
- Harden security in gdrdrv.
- Enable cached mappings in POWER9.
- Improve copy performance with unrolling in POWERPC.
- Creates _sanity_ unit test for testing the functionality and security.
- Consolidate _basic_ and _validate_ into _sanity_ unit test.
- Introduce compile time and runtime version checking in _libgdrapi_.
- Improve rpm packaging.
- Introduce deb packaging for the userspace library and the applications.
- Introduce dkms packaging for the _gdrdrv_ driver.
- Rename gdr\_copy\_from/to\_bar to gdr\_copy\_from/to\_mapping.
- Update README

## [1.3] - 2018-07-26
- Add _gdrdrv_ driver for converting cudaMalloc'd addresses to the GPU's BAR1
  addresses and exposing them to CPU-accessible virtual addresses.
- Add _libgdrapi_, a user-space library for communicating with the gdrdrv driver.
- Add _basic_ application as an minimal example on how to use gdrcopy.
- Add _copybw_ application as a complete example on how CPU could read/write to
  cudaMalloc'd memory via BAR1 mappings.
- Add _validate_ unit test to ensure that gdrcopy functions as expected.
- Add a script for packaging gdrcopy in the rpm format.


[2.4]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.4
[2.3.1]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.3.1
[2.3]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.3
[2.2]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.2
[2.1]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.1
[2.0]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.0
[1.3]: https://github.com/NVIDIA/gdrcopy/releases/tag/v1.3