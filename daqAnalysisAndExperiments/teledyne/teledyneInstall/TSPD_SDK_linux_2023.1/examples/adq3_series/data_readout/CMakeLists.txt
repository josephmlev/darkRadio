#
# Copyright 2020 Teledyne Signal Processing Devices Sweden AB
#
cmake_minimum_required(VERSION 3.10)

project(adq3_series_data_readout_example LANGUAGES C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(adq3_series_data_readout_example data_readout.c)
set_property(TARGET adq3_series_data_readout_example PROPERTY C_STANDARD 99)

target_compile_definitions(adq3_series_data_readout_example PRIVATE
  $<$<PLATFORM_ID:Linux>:LINUX>
  $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>
)

target_compile_options(adq3_series_data_readout_example PRIVATE
  $<$<C_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
  $<$<C_COMPILER_ID:MSVC>:/W4 /WX /wd4127>
)

if(NOT TARGET adq)
  if(WIN32)
    # Windows
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
      set(ADQAPI_INSTALL_PATH "$ENV{PROGRAMFILES}/SP Devices/ADQAPI_x64")
    else()
      set(ADQAPI_INSTALL_PATH "$ENV{PROGRAMFILES}/SP Devices/ADQAPI")
    endif()

    add_library(adq SHARED IMPORTED)
    set_target_properties(adq PROPERTIES
      IMPORTED_LOCATION ${ADQAPI_INSTALL_PATH}
      INTERFACE_INCLUDE_DIRECTORIES ${ADQAPI_INSTALL_PATH}
      IMPORTED_IMPLIB ${ADQAPI_INSTALL_PATH}/ADQAPI.lib)

    add_custom_command(TARGET adq3_series_data_readout_example POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${ADQAPI_INSTALL_PATH}/ADQAPI.dll
      $<TARGET_FILE_DIR:adq3_series_data_readout_example>)

    message(STATUS "Using ADQAPI library from:")
    message(STATUS "\t${ADQAPI_INSTALL_PATH}/ADQAPI.lib")
    message(STATUS "\t${ADQAPI_INSTALL_PATH}/ADQAPI.h")
    message(STATUS "\t${ADQAPI_INSTALL_PATH}/ADQAPI.dll")
  endif()
else()
  if(WIN32)
    add_custom_command(TARGET adq3_series_data_readout_example POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:adq>
      $<TARGET_FILE_DIR:adq3_series_data_readout_example>)
  endif()
endif()

target_link_libraries(adq3_series_data_readout_example PRIVATE adq)

add_custom_command(TARGET adq3_series_data_readout_example POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  "$<TARGET_FILE_DIR:adq3_series_data_readout_example>/data")
