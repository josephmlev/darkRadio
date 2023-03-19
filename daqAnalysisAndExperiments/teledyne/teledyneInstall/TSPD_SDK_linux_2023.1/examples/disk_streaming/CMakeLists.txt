#
# Copyright 2022 Teledyne Signal Processing Devices Sweden AB
#

cmake_minimum_required (VERSION 3.10)

project (disk_streaming LANGUAGES C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_FILES
  disk_streaming.c
  settings.h
  helpers.h
  helpers.c
)

add_executable(disk_streaming ${SOURCE_FILES})

if (UNIX)
  target_compile_options(disk_streaming PRIVATE -Wall -Wextra -pedantic)
  target_compile_definitions(disk_streaming PUBLIC LINUX)

  find_library(ADQ_LIB
    NAMES libadq libadq.so
    HINTS
      "${CMAKE_SOURCE_DIR}/../../source/.libs/"
      "${CMAKE_SOURCE_DIR}/../../build/"
  )

  find_library(ADNVDS_LIB
    NAMES libadnvds libadnvds.so
    HINTS
      "${CMAKE_SOURCE_DIR}/../../source/.libs/"
      "${CMAKE_SOURCE_DIR}/../../build/"
  )
else()
  # Windows
  target_compile_options(disk_streaming PRIVATE /W4 /WX /wd4127)
  target_compile_definitions(disk_streaming PUBLIC _CRT_SECURE_NO_WARNINGS)
  set (ADQAPI_INSTALL_PATH "$ENV{PROGRAMFILES}/SP Devices/ADQAPI_x64/")
  set (LOCAL_API_DIR "${CMAKE_SOURCE_DIR}/../../x64/Debug/")
  set (ADNVDS_INSTALL_PATH "$ENV{PROGRAMFILES}/SP Devices ADQNVMe/")

  find_library(ADQ_LIB
    NAMES ADQAPI.lib
    HINTS ${LOCAL_API_DIR} ${ADQAPI_INSTALL_PATH}
    )
  find_file(ADQAPI_DLL
    NAMES ADQAPI.dll
    HINTS ${LOCAL_API_DIR} ${ADQAPI_INSTALL_PATH}
    )

  if(NOT ADQAPI_DLL)
    message(FATAL_ERROR "ADQAPI.dll not found")
  endif()


  add_custom_command(TARGET disk_streaming POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${ADQAPI_DLL}
    $<TARGET_FILE_DIR:disk_streaming>)

  find_library(ADNVDS_LIB
    NAMES adnvds.lib
    HINTS ${ADNVDS_INSTALL_PATH}/lib/
    )
  find_file(ADNVDS_DLL
    NAMES adnvds.dll
    HINTS ${ADNVDS_INSTALL_PATH}/lib/
    )

  if(NOT ADNVDS_DLL)
    message(FATAL_ERROR "ADNVDS.dll not found")
  endif()

  add_custom_command(TARGET disk_streaming POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${ADNVDS_DLL}
    $<TARGET_FILE_DIR:disk_streaming>)

endif (UNIX)

find_path(ADQAPI_INCLUDE
  NAMES ADQAPI.h
  HINTS
    "${CMAKE_SOURCE_DIR}/../../Release/"
    "${CMAKE_SOURCE_DIR}/../../build/"
    ${ADQAPI_INSTALL_PATH}
)

find_path(ADNVDS_INCLUDE
  NAMES adnvds.h
  HINTS
    "${CMAKE_SOURCE_DIR}/../../Release/"
    "${CMAKE_SOURCE_DIR}/../../build/"
    ${CMAKE_SOURCE_DIR}
    ${ADNVDS_INSTALL_PATH}/include
)


target_link_libraries(disk_streaming ${ADQ_LIB} ${ADNVDS_LIB})
target_include_directories(disk_streaming PUBLIC ${ADQAPI_INCLUDE} ${ADNVDS_INCLUDE})

add_custom_target(build-time-make-directory ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:disk_streaming>/data")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(ADQAPI_DLL)
  message(STATUS "ADQAPI.dll: ${ADQAPI_DLL}")
endif()
message(STATUS "ADQ lib: ${ADQ_LIB}")
message(STATUS "ADQAPI.h: ${ADQAPI_INCLUDE}/ADQAPI.h")
if(ADNVDS_DLL)
  message(STATUS "adnvds.dll: ${ADNVDS_DLL}")
endif()
message(STATUS "ADNVDS lib: ${ADNVDS_LIB}")
message(STATUS "ADNVDS.h: ${ADNVDS_INCLUDE}/adnvds.h")
