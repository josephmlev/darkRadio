#
# Copyright 2021 Teledyne Signal Processing Devices Sweden AB
#

cmake_minimum_required(VERSION 3.10)

project(gen3_streaming_example LANGUAGES C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(gen3_streaming_example gen3_streaming.c)
set_property(TARGET gen3_streaming_example PROPERTY C_STANDARD 99)

if(UNIX)
  target_compile_options(gen3_streaming_example PRIVATE -Wall -Wextra -pedantic)
  target_compile_definitions(gen3_streaming_example PUBLIC LINUX)

  target_link_libraries(gen3_streaming_example PRIVATE adq)
else()
  # Windows
  target_compile_options(gen3_streaming_example PRIVATE /W4 /WX /wd4127)
  target_compile_definitions(gen3_streaming_example PUBLIC _CRT_SECURE_NO_WARNINGS)
  set(ADQAPI_INSTALL_PATH "$ENV{PROGRAMFILES}/SP Devices/ADQAPI_x64/")
  set(LOCAL_API_DIR "${CMAKE_SOURCE_DIR}/../../x64/Debug/")

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

  add_custom_command(TARGET gen3_streaming_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${ADQAPI_DLL}
    $<TARGET_FILE_DIR:gen3_streaming>)

  find_path(ADQAPI_INCLUDE
    NAMES ADQAPI.h
    HINTS
    "${CMAKE_SOURCE_DIR}/../../Release/"
    "${CMAKE_SOURCE_DIR}/../../build/"
    ${ADQAPI_INSTALL_PATH}
  )

  target_link_libraries(gen3_streaming_example ${ADQ_LIB})
  target_include_directories(gen3_streaming_example PUBLIC ${ADQAPI_INCLUDE})

  if(ADQAPI_DLL)
    message(STATUS "ADQAPI.dll: ${ADQAPI_DLL}")
  endif()

  message(STATUS "ADQ lib: ${ADQ_LIB}")
  message(STATUS "ADQAPI.h: ${ADQAPI_INCLUDE}/ADQAPI.h")
endif(UNIX)

add_custom_command(TARGET gen3_streaming_example POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:gen3_streaming_example>/data")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
