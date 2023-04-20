
/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */

#include "Settings.h"
#include <chrono>
#include "ADQAPI.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "helper_cuda.h"
#include "helpers.h"
#include "gdrapi.h"
#include <stdint.h>
#include <stdio.h>
#include <iostream>

template <typename T>
int cudacheck(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    printf("CUDA error at %s:%d code=%u(%s) \"%s\" \n", file, line,
           static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
  }
  return result != 0;
}

int AllocateAndPinBuffers(CUdeviceptr &buffer, void *&buffer_pointer, uint64_t &buffer_address,
                          unsigned int &buffer_size, gdr_t gdr, gdr_mh_t &memory_handle,
                          void *&bar_ptr_data)
{
  int gdr_status = 0;
  gdr_info_t info;
  unsigned int flag = 1;
  size_t offset_data;

  /* Round size upwards to GPU page size */
  buffer_size = (buffer_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
  CHECK_ERROR_RETURN(cuMemAlloc(&buffer, buffer_size), 0, "cuMemAlloc");
  /* Always synchronize memory operations */
  CHECK_ERROR_RETURN(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, buffer), 0,
                     "cuPointerSetAttribute");

  /* Map device memory buffer on GPU BAR1, returning an handle. Memory is still not accessible to
     user-space. */
  gdr_status = gdr_pin_buffer(gdr, buffer, buffer_size, 0, 0, &memory_handle);
  CHECK_ERROR_RETURN(gdr_status || (memory_handle.h == 0U), 0, "gdr_pin_buffer");

  /* Create a user-space mapping for the BAR1 info, length is bar1->buffer_size above. WARNING: the
     BAR physical address will be aligned to the page size before being mapped in user-space, so the
     pointer returned might be affected by an offset. gdr_get_info can be used to calculate that
     offset. */
  gdr_status = gdr_map(gdr, memory_handle, &bar_ptr_data, buffer_size);
  CHECK_ERROR_RETURN(gdr_status || (bar_ptr_data == 0U), 0, "gdr_map");
  CHECK_ERROR_RETURN(gdr_get_info(gdr, memory_handle, &info), 0, "gdr_get_info");
  offset_data = info.va - buffer;
  buffer_address = info.physical;
  CHECK_ERROR_RETURN(gdr_validate_phybar(gdr, memory_handle), 0, "gdr_validate_phybar");
  buffer_pointer = (char *)bar_ptr_data + offset_data;

  return 0;
}

void status_printout(double t_duration_count,
                     int nof_buffers_received[NOF_DIGITIZERS][NOF_CHANNELS],
                     struct ADQParameters adq)
{
  printf("Status for last %.02lf s:\n", t_duration_count);
  uint64_t tot_data = 0;
  for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
  {
    printf("  Digitizer %d:\n", adq_num);
    for (int ch = 0; ch < NOF_CHANNELS; ch++)
    {
      printf("    Ch %d: %.2f GB, %.3f GB/s\n", ch,
             (double)nof_buffers_received[adq_num - 1][ch]
               * adq.transfer.channel[ch].record_buffer_size / 1e9,
             (double)nof_buffers_received[adq_num - 1][ch]
               * adq.transfer.channel[ch].record_buffer_size / 1e9 / t_duration_count);
      tot_data += nof_buffers_received[adq_num - 1][ch]
                  * adq.transfer.channel[ch].record_buffer_size;
    }
  }
  printf("Total: %.2f GB, %.3f GB/s\n\n", (double)tot_data / 1e9,
         tot_data / 1e9 / t_duration_count);

  return;
}

void periodic_status_printout(std::chrono::high_resolution_clock::time_point t_start,
                              int nof_buffers_received[NOF_DIGITIZERS][NOF_CHANNELS],
                              struct ADQParameters adq)
{
  if (PRINTOUT_PERIOD <= 0)
    return;

  static std::chrono::high_resolution_clock::time_point t_last_printout =
    std::chrono::high_resolution_clock::now();
  static int nof_buffers_received_last[NOF_DIGITIZERS][NOF_CHANNELS] = {0};

  std::chrono::duration<double> t_duration = std::chrono::high_resolution_clock::now()
                                             - t_last_printout;

  if (t_duration.count() > PRINTOUT_PERIOD)
  {
    t_last_printout = std::chrono::high_resolution_clock::now();
    int nof_buffers_received_since_last[NOF_DIGITIZERS][NOF_CHANNELS];

    for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
      for (int ch = 0; ch < NOF_CHANNELS; ch++)
        nof_buffers_received_since_last[adq_num - 1][ch] =
          nof_buffers_received[adq_num - 1][ch] - nof_buffers_received_last[adq_num - 1][ch];
    std::chrono::duration<double> t_duration_tot = std::chrono::high_resolution_clock::now()
                                                   - t_start;
    printf("%.02lf s elapsed\n", t_duration_tot.count());
    status_printout(t_duration.count(), nof_buffers_received_since_last, adq);

    for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
    {
      for (int ch = 0; ch < NOF_CHANNELS; ch++)
      {
        nof_buffers_received_last[adq_num - 1][ch] = nof_buffers_received[adq_num - 1][ch];
      }
    }
  }
  return;
}

int main()
{
  /* GPU variables */
  gdr_t gdr = NULL;
  gdr_mh_t memory_handle[NOF_DIGITIZERS][NOF_CHANNELS][NOF_GPU_BUFFERS] = {0};
  void *bar_ptr_data[NOF_DIGITIZERS][NOF_CHANNELS][NOF_GPU_BUFFERS] = {NULL};
  int nof_gpu = 0;
  struct cudaDeviceProp prop[8];
  int gpu_num = 0;
  void *gpu_buffer_ptr[NOF_DIGITIZERS][NOF_CHANNELS][NOF_GPU_BUFFERS] = {NULL};
  unsigned int gpu_allocation_size[NOF_CHANNELS];
  CUdeviceptr gpu_buffers[NOF_DIGITIZERS][NOF_CHANNELS][NOF_GPU_BUFFERS] = {0};
  int16_t *buffer_copy = NULL;
  void *dummy;
  bool cuda_open = false;

  /* ADQ variables */
  void *adq_cu = NULL;
  struct ADQInfoListEntry *adq_list = NULL;
  struct ADQParameters adq;
  unsigned int nof_digitizers = 0;
  int result;

  int retval = 0;

  /* Data collection loop variables */
  bool data_transfer_done = false;
  struct ADQP2pStatus status;
  int nof_buffers_received[NOF_DIGITIZERS][NOF_CHANNELS] = {0};

  /* Performance measurement  variables */
  std::chrono::high_resolution_clock::time_point t_start;
  std::chrono::duration<double> t_duration;

  if (NOF_CHANNELS < 1 || NOF_CHANNELS > 2)
  {
    printf("Invalid NOF_CHANNELS %d\n", NOF_CHANNELS);
    goto exit_with_error;
  }

  /* Validate struct definitions. */
  switch (ADQAPI_ValidateVersion(ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR))
  {
  case 0:
    // ADQAPI version is compatible
    break;
  case -1:
    printf("ADQAPI version is incompatible. The application needs to be recompiled and relinked "
           "against the installed ADQAPI.\n");
    return -1;
  case -2:
    printf("ADQAPI version is backwards compatible. It's suggested to recompile and relink the "
           "application against the installed ADQAPI.\n");
    break;
  }

  /* Initialize GPU */
  CHECK_CUDAERROR_EXIT(cudaGetDeviceCount(&nof_gpu));
  for (int i = 0; i < nof_gpu; i++)
  {
    CHECK_CUDAERROR_EXIT(cudaGetDeviceProperties(&prop[i], i));
    printf("GPU %d: Name %s pciBusID %d\n", i, prop[i].name, prop[i].pciBusID);
  }

  if (nof_gpu > 1)
  {
    printf("Select GPU\n");
    scanf("%d", &gpu_num);
  }
  else if (nof_gpu < 1)
  {
    goto exit_with_error;
  }

  CHECK_CUDAERROR_EXIT(cudaSetDevice(gpu_num));
  CHECK_CUDAERROR_EXIT(cudaMalloc(&dummy, 0));
  cuda_open = true;

  /*  Initialize GDR */
  gdr = gdr_open();
  CHECK_ERROR_EXIT((gdr == (void *)0), 0);

  /* Initialize the handle to the ADQ control unit object. */
  adq_cu = CreateADQControlUnit();
  if (adq_cu == NULL)
  {
    printf("Failed to create a handle to an ADQ control unit object.\n");
    goto exit_with_error;
  }

  /* Enable the error trace log. */
  ADQControlUnit_EnableErrorTrace(adq_cu, 0x80000000, ".");

  /* List the available adqs connected to the host computer. */
  CHECK_ERROR_EXIT(ADQControlUnit_ListDevices(adq_cu, &adq_list, &nof_digitizers), 1);

  if (nof_digitizers == 0)
  {
    printf("No ADQ connected.\n");
    goto exit_with_error;
  }
  else if (nof_digitizers < NOF_DIGITIZERS)
  {
    printf("Error: found %u ADQs, but example is set to use %d ADQs\n", nof_digitizers,
           NOF_DIGITIZERS);
    goto exit_with_error;
  }
  else if (nof_digitizers > NOF_DIGITIZERS)
  {
    printf("Warning: found %u ADQs, but example is set to use %d ADQs\n"
           "Adq 1 - %d will be used\n",
           nof_digitizers, NOF_DIGITIZERS, NOF_DIGITIZERS);
  }
  else
  {
    printf("Found %u ADQs\n", nof_digitizers);
  }

  /* Perform setup and buffer allocation for all digitizers. */
  for (int adq_to_open_idx = 0; adq_to_open_idx < NOF_DIGITIZERS; adq_to_open_idx++)
  {
    int adq_num = adq_to_open_idx + 1;
    printf("\nSetting up ADQ %d (idx %d)\n", adq_num, adq_to_open_idx);
    CHECK_ERROR_EXIT(ADQControlUnit_SetupDevice(adq_cu, adq_to_open_idx), 1);

    /* Initialize parameter struct. */
    CHECK_ERROR_EXIT(ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_TOP, &adq),
                     (int)sizeof(adq));

    /* Modify parameters (values from the header file "settings.h"). */
    adq.signal_processing.dbs.channel[0].level = CH0_DBS_LEVEL;
    adq.signal_processing.dbs.channel[0].enabled = CH0_DBS_ENABLED;
    adq.signal_processing.dbs.channel[1].level = CH1_DBS_LEVEL;
    adq.signal_processing.dbs.channel[1].enabled = CH1_DBS_ENABLED;
    adq.signal_processing.sample_skip.channel[0].skip_factor = CH0_SAMPLE_SKIP_FACTOR;
    adq.signal_processing.sample_skip.channel[1].skip_factor = CH1_SAMPLE_SKIP_FACTOR;

    adq.test_pattern.channel[0].source = CH0_TEST_PATTERN_SOURCE;
    adq.test_pattern.channel[1].source = CH1_TEST_PATTERN_SOURCE;

    adq.event_source.level.channel[0].level = CH0_LT_LEVEL;
    adq.event_source.level.channel[0].arm_hysteresis = CH0_LT_ARM_HYSTERESIS;
    adq.event_source.level.channel[1].level = CH1_LT_LEVEL;
    adq.event_source.level.channel[1].arm_hysteresis = CH1_LT_ARM_HYSTERESIS;

    adq.event_source.periodic.period = PERIODIC_EVENT_SOURCE_PERIOD;
    adq.event_source.periodic.frequency = PERIODIC_EVENT_SOURCE_FREQUENCY;
#if ADQAPI_VERSION_MAJOR < 7
    adq.event_source.port[ADQ_PORT_TRIG].threshold = TRIGGER_THRESHOLD_V;
    adq.event_source.port[ADQ_PORT_SYNC].threshold = TRIGGER_THRESHOLD_V;
#else
    adq.event_source.port[ADQ_PORT_TRIG].pin[0].threshold = TRIGGER_THRESHOLD_V;
    adq.event_source.port[ADQ_PORT_SYNC].pin[0].threshold = TRIGGER_THRESHOLD_V;
#endif
    /* Configure data acquisition for channel 0. */
    adq.acquisition.channel[0].nof_records = ADQ_INFINITE_NOF_RECORDS;
    adq.acquisition.channel[0].record_length = CH0_RECORD_LEN;
    adq.acquisition.channel[0].trigger_source = CH0_TRIGGER_SOURCE;
    adq.acquisition.channel[0].trigger_edge = CH0_TRIGGER_EDGE;
    adq.acquisition.channel[0].horizontal_offset = CH0_HORIZONTAL_OFFSET;

    /* Configure data acquisition for channel 1. */
    if (NOF_CHANNELS > 1)
    {
      adq.acquisition.channel[1].nof_records = ADQ_INFINITE_NOF_RECORDS;
      adq.acquisition.channel[1].record_length = CH1_RECORD_LEN;
      adq.acquisition.channel[1].trigger_source = CH1_TRIGGER_SOURCE;
      adq.acquisition.channel[1].trigger_edge = CH1_TRIGGER_EDGE;
      adq.acquisition.channel[1].horizontal_offset = CH1_HORIZONTAL_OFFSET;
    }

    /* Configure common data transfer parameters. */
    adq.transfer.common.write_lock_enabled = 1;
    adq.transfer.common.transfer_records_to_host_enabled = 0;
    adq.transfer.common.marker_mode = ADQ_MARKER_MODE_HOST_MANUAL;

    /* Configure data transfer parameters for channel 0: fixed length, no metadata. */
    adq.transfer.channel[0].record_length_infinite_enabled = 0;
    adq.transfer.channel[0].record_size = adq.acquisition.channel[0].bytes_per_sample
                                          * adq.acquisition.channel[0].record_length;
    adq.transfer.channel[0].record_buffer_size = NOF_RECORDS_PER_BUFFER
                                                 * adq.transfer.channel[0].record_size;
    adq.transfer.channel[0].metadata_enabled = 0;
    adq.transfer.channel[0].nof_buffers = NOF_GPU_BUFFERS;

    /* Configure data transfer parameters for channel 1: fixed length, no metadata. */
    if (NOF_CHANNELS > 1)
    {
      adq.transfer.channel[1].record_length_infinite_enabled = 0;
      adq.transfer.channel[1].record_size = adq.acquisition.channel[1].bytes_per_sample
                                            * adq.acquisition.channel[1].record_length;
      adq.transfer.channel[1].record_buffer_size = NOF_RECORDS_PER_BUFFER
                                                   * adq.transfer.channel[1].record_size;
      adq.transfer.channel[1].metadata_enabled = 0;
      adq.transfer.channel[1].nof_buffers = NOF_GPU_BUFFERS;
    }

    /* Allocate GPU buffers */
    printf("Allocating GPU buffers\n");
    for (int ch = 0; ch < NOF_CHANNELS; ch++)
    {
      printf("Ch %d: %d buffers %.2lf MB each\n", ch, NOF_GPU_BUFFERS,
             (double)adq.transfer.channel[ch].record_buffer_size / 1e6);

      for (int buffer = 0; buffer < NOF_GPU_BUFFERS; buffer++)
      {
        uint64_t gpu_buffer_addr = 0;
        /* NOTE: The GPU allocation size may be rounded upwards by "AllocateAndPinBuffers" */
        gpu_allocation_size[ch] = adq.transfer.channel[ch].record_buffer_size;

        retval = AllocateAndPinBuffers(gpu_buffers[adq_to_open_idx][ch][buffer],
                                       gpu_buffer_ptr[adq_to_open_idx][ch][buffer], gpu_buffer_addr,
                                       gpu_allocation_size[ch], gdr,
                                       memory_handle[adq_to_open_idx][ch][buffer],
                                       bar_ptr_data[adq_to_open_idx][ch][buffer]);

        if (gpu_buffer_ptr[adq_to_open_idx][ch][buffer] && gpu_buffer_addr && retval == 0)
        {
          /* Add buffers to transfer_parameters */
          adq.transfer.channel[ch].record_buffer_bus_address[buffer] = gpu_buffer_addr;
          /* Add host pointer to buffer (optional) */
          adq.transfer.channel[ch].record_buffer[buffer] = gpu_buffer_ptr;
        }
        else
        {
          printf("GPU ch %d buffer %d allocation failed\n", ch, buffer);
          goto exit_with_error;
        }
      }
    }

    printf("Configuring digitizer parameters\n");
    CHECK_ERROR_EXIT(ADQ_SetParameters(adq_cu, adq_num, &adq), (int)sizeof(adq));
  }

  printf("\nStarting data acquisition for ADQ 1 - %d. \n", NOF_DIGITIZERS);
  /* Store timepoint for streaming start */
  t_start = std::chrono::high_resolution_clock::now();

  /* Start the data acquisition. */
  for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
    CHECK_ERROR_EXIT(ADQ_StartDataAcquisition(adq_cu, adq_num), ADQ_EOK);

  printf("Success\n\n");

  /* Enter the data collection loop. */
  data_transfer_done = false;
  while (!data_transfer_done)
  {
    for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
    {
      result = ADQ_WaitForP2pBuffers(adq_cu, adq_num, &status, WAIT_TIMEOUT_MS);

      if (result == ADQ_EAGAIN && WAIT_TIMEOUT_MS == 0)
      { /* Wait timeout is expected frequently when timeout is 0 */
        continue;
      }
      else if (result == ADQ_EAGAIN)
      {
        /* Breaking at timeout, sometimes continue is a better option */
        printf("Timed out while waiting for peer-to-peer buffers.\n");
        retval = -1;
        goto exit_streaming;
      }
      else if (result < 0)
      {
        printf("Waiting for peer-to-peer markers failed with retcode %d.\n", result);
        retval = -1;
        goto exit_streaming;
      }
      else/* result == ADQ_EOK */
      {
        /* Process received buffers */
        for (int buffer = 0; buffer < status.channel[0].nof_completed_buffers
                             || buffer < status.channel[NOF_CHANNELS - 1].nof_completed_buffers;
             buffer++)
        {
          /* Channel in inner loop to maximize throughput.
          Unlock transfer buffer for all active channels ASAP to keep transfer active */
          for (int ch = 0; ch < NOF_CHANNELS; ch++)
          {
            if (buffer < status.channel[ch].nof_completed_buffers)
            {
              int buffer_index = status.channel[ch].completed_buffers[buffer];
              /* Trigger your GPU processing of received buffer:
                 gpu_buffers[adq_num-1][ch][buffer_index] */

              /* Make the buffer available to receive data once again. */
              result = ADQ_UnlockP2pBuffers(adq_cu, adq_num, ch, (1ull << buffer_index));
              if (result != ADQ_EOK)
              {
                retval = -1;
                printf("Error: UnlockP2pBuffers returned %d\n", result);
                goto exit_streaming;
              }
              nof_buffers_received[adq_num - 1][ch]++;
            }
          }
        }
      }
    }

    periodic_status_printout(t_start, nof_buffers_received, adq);

    /* Collection done when all active channels has received specified number of buffers */
    data_transfer_done = 1;
    for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
    {
      for (int ch = 0; ch < NOF_CHANNELS; ch++)
      {
        if (nof_buffers_received[adq_num - 1][ch] < NOF_BUFFERS_TO_RECEIVE)
        {
          data_transfer_done = 0;
        }
      }
    }
  }

exit_streaming:
  t_duration = std::chrono::high_resolution_clock::now() - t_start;

  /* Stop the data acquisition process. */
  printf("Stop acquiring data... ");
  for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
  {
    result = ADQ_StopDataAcquisition(adq_cu, adq_num);
    switch (result)
    {
    case ADQ_EOK:
    case ADQ_EINTERRUPTED:
      break;
    default:
      printf("failed for ADQ %d, code %d.\n\n", adq_num, result);
      break;
    }
  }
  printf("success.\n\n");

  status_printout(t_duration.count(), nof_buffers_received, adq);

  if (PRINT_LAST_BUFFERS == 0)
    goto exit;

  printf("\n******************** GPU buffers printout ********************\n");
  /* Copy buffers from GPU and print a few samples */
  for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
  {
    printf("ADQ %d\n", adq_num);
    for (int ch = 0; ch < NOF_CHANNELS; ch++)
    {
      printf("  Channel %d\n", ch);
      buffer_copy = (int16_t *)malloc(adq.transfer.channel[ch].record_buffer_size);

      if (!buffer_copy)
      {
        printf("Allocation of buffer_copy failed\n");
        goto exit_with_error;
      }

      for (int buffer = nof_buffers_received[adq_num - 1][ch] - NOF_GPU_BUFFERS;
           buffer < nof_buffers_received[adq_num - 1][ch]; buffer++)
      {
        printf("    Buffer %d\n", buffer);
        int buffer_index = buffer % NOF_GPU_BUFFERS;

        if (CHECK_CUDAERROR(
              cudaMemcpy((void *)buffer_copy, (void *)gpu_buffers[adq_num - 1][ch][buffer_index],
                         adq.transfer.channel[ch].record_buffer_size, cudaMemcpyDeviceToHost)))
        {
          goto stop_print;
        }

        for (int record = 0; record < 2; record++)
        {
          printf("      Record %d: ", record);
          for (int sample = 0; sample < 8; sample++)
          {
            printf("%d, ",
                   buffer_copy[record * adq.acquisition.channel[ch].record_length + sample]);
          }
          printf("\n");
        }
        printf("\n");
      }
      free(buffer_copy);
    }
    printf("\n");
  }

  goto exit;

stop_print:
  free(buffer_copy);

exit_with_error:
  retval = -1;

exit:
  /* Unmap & free GPU buffers */
  for (int adq_num = 1; adq_num <= NOF_DIGITIZERS; adq_num++)
  {
    for (int ch = 0; ch < NOF_CHANNELS; ch++)
    {
      for (int buffer = 0; buffer < NOF_GPU_BUFFERS; buffer++)
      {
        if (memory_handle[adq_num - 1][ch][buffer].h)
        {
          if (gdr_unmap(gdr, memory_handle[adq_num - 1][ch][buffer],
                        bar_ptr_data[adq_num - 1][ch][buffer], gpu_allocation_size[ch]))
            printf("gdr_unmap ADQ %d ch %d buffer %d failed\n", adq_num, ch, buffer);
          if (gdr_unpin_buffer(gdr, memory_handle[adq_num - 1][ch][buffer]))
            printf("gdr_unpin_buffer ADQ %d ch %d buffer %d failed\n", adq_num, ch, buffer);
        }

        if (gpu_buffers[adq_num - 1][ch][buffer])
        {
          if (cuMemFree(gpu_buffers[adq_num - 1][ch][buffer]))
            printf("cuMemFree ADQ %d ch %d buffer %d failed\n", adq_num, ch, buffer);
        }
      }
    }
  }

  if (gdr)
  {
    if (gdr_close(gdr))
      printf("gdr_close failed\n");
  }

  /* Delete the control unit object */
  if (adq_cu)
    DeleteADQControlUnit(adq_cu);

  if (cuda_open)
    CHECK_CUDAERROR(cudaDeviceReset());

  printf("Exiting the application.\n");
  fflush(stdout);
  return retval;
}
