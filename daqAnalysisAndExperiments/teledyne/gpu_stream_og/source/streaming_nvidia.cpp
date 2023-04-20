/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */

#include "Settings.h"
#include <chrono>
#include "ADQAPI.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "helper_cuda.h"
#include "cuda_helpers.h"
#include "gdrapi.h"
#include <stdint.h>
#include <stdio.h>

#ifndef __linux__
#include <Windows.h>
#include <conio.h>
#else
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>

int _kbhit()
{
    static const int STDIN = 0;
    static int initialized = 0;

    if (!initialized) {
        // Use termios to turn off line buffering
        struct termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        tcsetattr(STDIN, TCSANOW, &term);
        setbuf(stdin, NULL);
        initialized = 1;
    }

    int bytesWaiting;
    ioctl(STDIN, FIONREAD, &bytesWaiting);
    return bytesWaiting;
}

static struct termios old, current;

/* Initialize new terminal i/o settings */
void initTermios(int echo)
{
    tcgetattr(0, &old); /* grab old terminal i/o settings */
    current = old; /* make new settings same as old settings */
    current.c_lflag &= ~ICANON; /* disable buffered i/o */
    if (echo) {
        current.c_lflag |= ECHO; /* set echo mode */
    }
    else {
        current.c_lflag &= ~ECHO; /* set no echo mode */
    }
    tcsetattr(0, TCSANOW, &current); /* use these new terminal i/o settings now */
}

/* Restore old terminal i/o settings */
void resetTermios(void)
{
    tcsetattr(0, TCSANOW, &old);
}

/* Read 1 character - echo defines echo mode */
char getch_(int echo)
{
    char ch;
    initTermios(echo);
    ch = getchar();
    resetTermios();
    return ch;
}

/* Read 1 character without echo */
char _getch(void)
{
    return getch_(0);
}

/* Read 1 character with echo */
char _getche(void)
{
    return getch_(1);
}
#endif

#define CHECK_CUDAERROR(val) cudacheck((val), #val, __FILE__, __LINE__)
#define CHECK_CUDAERROR_EXIT(val) if(CHECK_CUDAERROR(val))  goto exit

template <typename T>
int cudacheck(T result, char const *const func, const char *const file,
  int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
      static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
  }
  return result != 0;
}


int AllocateAndPinBuffers(CUdeviceptr& buffer, void*& buffer_pointer,
  uint64_t& buffer_address, unsigned int &buffer_size, gdr_t gdr, gdr_mh_t& memory_handle, void* &bar_ptr_data)
{
  int gdr_status = 0;
  gdr_info_t info;
  unsigned int flag = 1;
  size_t offset_data;

  buffer_size = (buffer_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK; // round size upwards to GPU page size
  ASSERTDRV(cuMemAlloc(&buffer, buffer_size));                      //Allocate mem in GPU, buffer pointing to allocated mem
  ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, buffer));  // always synchronize
                                                                                        //memory operations that are synchronous
  gdr_status = gdr_pin_buffer(gdr, buffer, buffer_size, 0, 0, &memory_handle); // Map device memory buffer on GPU BAR1, returning an handle.
                                                                  // Memory is still not accessible to user-space.
  GDR_CHECK_ERROR_EXIT_FUNC(gdr_status || (memory_handle.h == 0U), "gdr_pin_buffer");
  gdr_status = gdr_map(gdr, memory_handle, &bar_ptr_data, buffer_size);  // create a user-space mapping for the BAR1 info, length is bar1->buffer_size above.
                                                          // WARNING: the BAR physical address will be aligned to the page size
                                                          // before being mapped in user-space, so the pointer returned might be
                                                          // affected by an offset. gdr_get_info can be used to calculate that
                                                          // offset.

  GDR_CHECK_ERROR_EXIT_FUNC(gdr_status || (bar_ptr_data == 0U), "gdr_map");
  gdr_status = gdr_get_info(gdr, memory_handle, &info);
  GDR_CHECK_ERROR_EXIT_FUNC(gdr_status, "gdr_info");
  offset_data = info.va - buffer;
  buffer_address = info.physical;
//  gdr_status = gdr_validate_phybar(gdr, memory_handle);
  GDR_CHECK_ERROR_EXIT_FUNC(gdr_status, "gdr_validate_phybar");

  buffer_pointer = (int8_t*) bar_ptr_data + offset_data; // Add offset in bytes
  return gdr_status;
}



int main()
{
  /* GPU variables */
  gdr_t gdr = NULL;
  gdr_mh_t memory_handle[NOF_CHANNELS][NOF_GPU_BUFFERS] = {{ {0} }};
  void* bar_ptr_data[NOF_CHANNELS][NOF_GPU_BUFFERS] = {{ NULL }};
  int nof_gpu = 0;
  struct cudaDeviceProp prop[4];
  int gpu_num = 0;
  void *gpu_buffer_ptr[NOF_CHANNELS][NOF_GPU_BUFFERS] = {{ NULL }};
  unsigned int gpu_allocation_size[NOF_CHANNELS];
  CUdeviceptr gpu_buffers[NOF_CHANNELS][NOF_GPU_BUFFERS] = { {0} };
  int16_t *buffer_copy = NULL;
  void *dummy;
  bool cuda_open = false;
  int hasGPUDirectSupport = 0;

  /* ADQ variables */
  void *adq_cu = NULL;
  struct ADQInfoListEntry *adq_list = NULL;
  int device_to_open_idx = 0;
  int adq_num = 1;
  struct ADQParameters adq;
  unsigned int nof_devices = 0;
  int result;

  /* Data collection loop variables */
  bool data_transfer_done = false;
  struct ADQP2pStatus status;
  int nof_buffers_received[2] = {0,0};
  uint64_t nof_buffers_received_total = 0;

  /* Performance measurement  variables */
  std::chrono::high_resolution_clock::time_point t_average_start;
  std::chrono::high_resolution_clock::time_point t_loop_start;
  std::chrono::duration<double> t_data_average;
  std::chrono::duration<double> t_data_loop;
  uint64_t tot_nof_bytes_received = 0;
  uint64_t tot_nof_bytes_received_cached = 0;

  if (NOF_CHANNELS < 1 || NOF_CHANNELS > 2)
  {
    printf("invalid NOF_CHANNELS\n");
    goto exit;
  }

  /* Validate struct definitions. */
  switch (ADQAPI_ValidateVersion(ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR))
  {
  case 0:
    // ADQAPI version is compatible
    break;
  case -1:
    printf("ADQAPI version is incompatible. The application needs to be recompiled and relinked against the installed ADQAPI.\n");
    return -1;
  case -2:
    printf("ADQAPI version is backwards compatible. It's suggested to recompile and relink the application against the installed ADQAPI.\n");
    break;
  }

  /* Initialize GPU */
  CHECK_CUDAERROR_EXIT(cudaGetDeviceCount(&nof_gpu));
  for (int i = 0; i < nof_gpu; i++)
  {
    hasGPUDirectSupport = 0;
    CHECK_CUDAERROR_EXIT(cudaGetDeviceProperties(&prop[i], i));
    CHECK_CUDAERROR_EXIT(cudaDeviceGetAttribute(&hasGPUDirectSupport, cudaDevAttrGPUDirectRDMASupported, i));

    printf("GPU %d: Name %s pciBusID %d. \ncudaDevAttrGPUDirectRDMASupported = %d\n", i, prop[i].name, prop[i].pciBusID, hasGPUDirectSupport);
  }

  if (nof_gpu > 1)
  {
    printf("Select GPU\n");
    scanf("%d", &gpu_num);

    hasGPUDirectSupport = 0;
    CHECK_CUDAERROR_EXIT(cudaDeviceGetAttribute(&hasGPUDirectSupport, cudaDevAttrGPUDirectRDMASupported, gpu_num));
    if (!hasGPUDirectSupport)
    {
        printf("The selected GPU does not have GPUDirect support which is required to run this example!\n");
        return -1;
    }
  }
  else if (nof_gpu < 1)
  {
    return -1;
  }

  CHECK_CUDAERROR_EXIT(cudaSetDevice(gpu_num));
  CHECK_CUDAERROR_EXIT(cudaMalloc(&dummy, 0));
  cuda_open = true;

  /*  Initialize GDR */
  gdr = gdr_open();
  GDR_CHECK_ERROR_EXIT((gdr == (void*)0), "gdr_open");


  /* Initialize the handle to the ADQ control unit object. */
  adq_cu = CreateADQControlUnit();
  if (adq_cu == NULL)
  {
    printf("Failed to create a handle to an ADQ control unit object.\n");
    return -1;
  }

  /* Enable the error trace log. */
  ADQControlUnit_EnableErrorTrace(adq_cu, 0x80000000, ".");

  /* List the available devices connected to the host computer. */
  if (!ADQControlUnit_ListDevices(adq_cu, &adq_list, &nof_devices))
  {
    printf("ListDevices failed!\n");
    return -1;
  }

  if (nof_devices == 0)
  {
    printf("No device connected.\n");
    goto exit;
  }
  else if (nof_devices != 1)
  {
    printf("Multiple ADQ devices detected, selecting device 0\n");
  }

  /* Since this example only supports one device, we always use the device at
     list index zero. */

  printf("Setting up device... ");
  if (ADQControlUnit_SetupDevice(adq_cu, device_to_open_idx))
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Set ADQ parameters to default values */
  if (ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_TOP, &adq) != sizeof(adq))
  {
    printf("Failed to initialize digitizer parameters.\n");
    goto exit;
  }

  /* Modify parameters (values from the header file "settings.h"). */
  adq.afe.channel[0].dc_offset = CH0_DC_OFFSET;
  adq.afe.channel[1].dc_offset = CH1_DC_OFFSET;

  adq.signal_processing.dbs.channel[0].level = CH0_DBS_LEVEL;
  adq.signal_processing.dbs.channel[0].bypass = CH0_DBS_BYPASS;
  adq.signal_processing.dbs.channel[1].level = CH1_DBS_LEVEL;
  adq.signal_processing.dbs.channel[1].bypass = CH1_DBS_BYPASS;
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

  adq.event_source.port[ADQ_PORT_TRIG].threshold = TRIGGER_THRESHOLD_V;
  adq.event_source.port[ADQ_PORT_SYNC].threshold = TRIGGER_THRESHOLD_V;

  /* Configure data acquisition for channel 0. */
  adq.acquisition.channel[0].nof_records = NOF_RECORDS_PER_BUFFER * NOF_BUFFERS_TO_RECEIVE;
  adq.acquisition.channel[0].record_length = CH0_RECORD_LEN;
  adq.acquisition.channel[0].trigger_source = CH0_TRIGGER_SOURCE;
  adq.acquisition.channel[0].trigger_edge = CH0_TRIGGER_EDGE;
  adq.acquisition.channel[0].horizontal_offset = CH0_HORIZONTAL_OFFSET;

  /* Configure data acquisition for channel 1. */
  if (NOF_CHANNELS > 1)
  {
    adq.acquisition.channel[1].nof_records = NOF_RECORDS_PER_BUFFER * NOF_BUFFERS_TO_RECEIVE;
    adq.acquisition.channel[1].record_length = CH1_RECORD_LEN;
    adq.acquisition.channel[1].trigger_source = CH1_TRIGGER_SOURCE;
    adq.acquisition.channel[0].trigger_edge = CH1_TRIGGER_EDGE;
    adq.acquisition.channel[1].horizontal_offset = CH1_HORIZONTAL_OFFSET;
  }

  /* Configure common data transfer parameters. */
  adq.transfer.common.write_lock_enabled = 1;
  adq.transfer.common.transfer_records_to_host_enabled = 0;
  adq.transfer.common.marker_mode = ADQ_MARKER_MODE_HOST_MANUAL;

  /* Configure data transfer parameters for channel 0: fixed length, no metadata. */
  adq.transfer.channel[0].record_length_infinite_enabled = 0;
  adq.transfer.channel[0].record_size = sizeof(int16_t) * adq.acquisition.channel[0].record_length;
  adq.transfer.channel[0].record_buffer_size = NOF_RECORDS_PER_BUFFER * adq.transfer.channel[0].record_size;
  adq.transfer.channel[0].metadata_enabled = 0;
  adq.transfer.channel[0].nof_buffers = NOF_GPU_BUFFERS;

  /* Configure data transfer parameters for channel 1: fixed length, no metadata. */
  if (NOF_CHANNELS > 1)
  {
    adq.transfer.channel[1].record_length_infinite_enabled = 0;
    adq.transfer.channel[1].record_size = sizeof(int16_t) * adq.acquisition.channel[1].record_length;
    adq.transfer.channel[1].record_buffer_size = NOF_RECORDS_PER_BUFFER * adq.transfer.channel[1].record_size;
    adq.transfer.channel[1].metadata_enabled = 0;
    adq.transfer.channel[1].nof_buffers = NOF_GPU_BUFFERS;
  }


  /* Allocate GPU buffers */
  printf("Allocating GPU buffers\n");
  for (int ch = 0; ch < NOF_CHANNELS; ch++)
  {
    printf("Ch %d: %.2f MB\n", ch, (float)adq.transfer.channel[ch].record_buffer_size / 1e6);
    for (int b = 0; b < NOF_GPU_BUFFERS; b++)
    {
      uint64_t gpu_buffer_addr = 0;
      int retval;
      /* NOTE: The GPU allocation size may be rounded upwards by "AllocateAndPinBuffers" */
      gpu_allocation_size[ch] = adq.transfer.channel[ch].record_buffer_size;

      retval = AllocateAndPinBuffers(gpu_buffers[ch][b], gpu_buffer_ptr[ch][b], gpu_buffer_addr,
        gpu_allocation_size[ch], gdr, memory_handle[ch][b], bar_ptr_data[ch][b]);

      if (gpu_buffer_ptr[ch][b] && gpu_buffer_addr && retval == 0)
      {
        /* Add buffers to transfer_parameters */
        adq.transfer.channel[ch].record_buffer_bus_address[b] = gpu_buffer_addr;
        /* Add pointer to buffer (optional) */
        adq.transfer.channel[ch].record_buffer[b] = gpu_buffer_ptr;
      }
      else
      {
        printf("GPU ch %d buffer %d allocation failed\n", ch, b);
        goto exit;
      }
    }
  }

  printf("Configuring digitizer parameters... ");
  if (ADQ_SetParameters(adq_cu, adq_num, &adq) == sizeof(adq))
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Store timepoint for streaming start */
  t_average_start = std::chrono::high_resolution_clock::now();

  /* Start the data acquisition. */
  printf("Start acquiring data... ");
  if (ADQ_StartDataAcquisition(adq_cu, adq_num) == ADQ_EOK)
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    goto exit;
  }

  /* Send software triggers. This structure sends more triggers than required
     since all channels share a common source. */
  for (int ch = 0; ch < NOF_CHANNELS; ++ch)
  {
    if (adq.acquisition.channel[ch].trigger_source == ADQ_EVENT_SOURCE_SOFTWARE)
    {
      if (adq.acquisition.channel[0].nof_records != 0)
        printf("Generating software events on channel %d.\n", ch);

      for (int i = 0; i < adq.acquisition.channel[0].nof_records; ++i)
      {
        if(ADQ_SWTrig(adq_cu, adq_num) != ADQ_EOK)
        {
          printf("Error: SWTrig failed.\n");
          goto exit;
        }
      }
    }
  }

  t_loop_start = std::chrono::high_resolution_clock::now();

  /* Enter the data collection loop. */
  while (!data_transfer_done)
  {
    result = ADQ_WaitForP2pBuffers(adq_cu, adq_num, &status, WAIT_TIMEOUT_MS);

    if (result == ADQ_EAGAIN)
    {
      printf("Timed out while waiting for peer-to-peer buffers.\n");
      goto exit_streaming;
    }
    else if (result < 0)
    {
      printf("Waiting for peer-to-peer markers failed with retcode %d.\n", result);
      goto exit_streaming;
    }
    else
    {
      /* Process received buffers */
      for (int buf = 0; buf < status.channel[0].nof_completed_buffers ||
            buf < status.channel[NOF_CHANNELS - 1].nof_completed_buffers; buf++)
      {
        /* Channel in inner loop to maximize throughput.
        Unlock transfer buffer for all active channels ASAP to keep transfer active */
        for (int ch = 0; ch < NOF_CHANNELS; ch++)
        {
          if (buf < status.channel[ch].nof_completed_buffers)
          {
            int buffer_index = status.channel[ch].completed_buffers[buf];
            /* Trigger your GPU processing of received buffer: gpu_buffers[ch][buffer_index] */


            /* Make the buffer available to receive data once again. */
            ADQ_UnlockP2pBuffers(adq_cu, adq_num, ch, (1ull << buffer_index));

            nof_buffers_received[ch]++;
            nof_buffers_received_total++;

            tot_nof_bytes_received += (uint64_t)adq.transfer.channel[ch].record_buffer_size;
          }
        }
      }
      /* Collection done when all active channels has received specified number of buffers */
      data_transfer_done = nof_buffers_received[0] >= NOF_BUFFERS_TO_RECEIVE
        && nof_buffers_received[NOF_CHANNELS - 1] >= NOF_BUFFERS_TO_RECEIVE;
    }

    t_data_loop = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - t_loop_start);

    if (t_data_loop.count() >= 1.50)
    {
        t_data_average = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - t_average_start);

        t_data_loop = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - t_loop_start);

        printf("%lu buffers, %.3Lf GB in %.3f s = [Average: %.3Lf GB/s] [Momentary: %.4Lf GB/s]\n", nof_buffers_received_total,
                                                                                                    (long double)tot_nof_bytes_received / 1e9,
                                                                                                    t_data_average.count(),
                                                                                                    ((long double)tot_nof_bytes_received / 1e9) / t_data_average.count(),
                                                                                                    ((long double)(tot_nof_bytes_received - tot_nof_bytes_received_cached) / 1e9) / t_data_loop.count());

        // Restart the counter
        t_loop_start = std::chrono::high_resolution_clock::now();
        tot_nof_bytes_received_cached = tot_nof_bytes_received;
    }

    if (_kbhit())
        break;
  }

  t_data_average = std::chrono::duration_cast<std::chrono::duration<double> >(
    std::chrono::high_resolution_clock::now() - t_average_start);

  // Reset and recount?
  tot_nof_bytes_received = 0;
  for (int ch = 0; ch < NOF_CHANNELS; ch++)
  {
    tot_nof_bytes_received += (uint64_t)adq.transfer.channel[ch].record_buffer_size
                              * nof_buffers_received[ch];
  }

  printf("\n%lu buffers, %.3Lf GB in %.3f s = [Average: %.4Lf GB/s] \n", nof_buffers_received_total,
                                                                          (long double)tot_nof_bytes_received / 1e9,
                                                                          t_data_average.count(),
                                                                          ((long double)tot_nof_bytes_received / 1e9) / (long double)t_data_average.count());

  /* Copy last buffers from GPU and print a few samples */
#if (PRINT_LAST_BUFFERS == 1)

  printf("\nGPU buffers printout\n");
  for (int ch = 0; ch < NOF_CHANNELS; ch++)
  {
    printf("Channel %d\n", ch);
    buffer_copy = (int16_t *)malloc(adq.transfer.channel[ch].record_buffer_size);

    if (!buffer_copy)
    {
      printf("Allocation of buffer_copy failed\n");
      goto exit_streaming;
    }

    for (int buffer = nof_buffers_received[ch]-NOF_GPU_BUFFERS; buffer < nof_buffers_received[ch];
         buffer++)
    {
      printf("Buffer %d\n", buffer);
      int buffer_index = buffer % NOF_GPU_BUFFERS;

      if (CHECK_CUDAERROR(cudaMemcpy((void *)buffer_copy, (void *)gpu_buffers[ch][buffer_index],
          adq.transfer.channel[ch].record_buffer_size, cudaMemcpyDeviceToHost)))
        goto stop_print;

      for (int record = 0; record < 2; record++)
      {
        printf("Record %d: ", record);
        for (int sample = 0; sample < 16; sample++)
        {
          printf("%d, ", buffer_copy[record * adq.acquisition.channel[ch].record_length + sample]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
    free(buffer_copy);
  }
  goto exit_streaming;

stop_print:
  free(buffer_copy);

#endif

exit_streaming:
 /* Stop the data acquisition process. */
  printf("Stop acquiring data... ");
  result = ADQ_StopDataAcquisition(adq_cu, adq_num);
  switch (result)
  {
  case ADQ_EOK:
  case ADQ_EINTERRUPTED:
    printf("success.\n");
    break;
  default:
    printf("failed, code %d.\n", result);
    break;
  }

cleanup_and_exit_with_error:
exit:

  printf("Press key to exit!\n");
  while (_kbhit()) { _getch(); }   //Reset _kbhit()
  while (!_kbhit()) {}

  for (int ch = 0; ch < NOF_CHANNELS; ch++)
  {
    /* Free GPU buffers */
    for (int b = 0; b < NOF_GPU_BUFFERS; b++)
    {
      if (memory_handle[ch][b].h)
      {
        if (gdr_unmap(gdr, memory_handle[ch][b], bar_ptr_data[ch][b], gpu_allocation_size[ch]))
          printf("gdr_unmap ch %d buffer %d failed\n", ch, b);
        if (gdr_unpin_buffer(gdr, memory_handle[ch][b]))
          printf("gdr_unpin_buffer ch %d buffer %d failed\n", ch, b);
      }

      if (gpu_buffers[ch][b])
      {
        if (cuMemFree(gpu_buffers[ch][b]))
          printf("cuMemFree ch %d buffer %d failed\n", ch, b);
      }
    }
  }

  if (gdr)
  {
    if (gdr_close(gdr))
      printf("gdr_close failed\n");
  }

  /* Delete the control unit object and the memory allocated by this application. */
  if (adq_cu)
    DeleteADQControlUnit(adq_cu);

  if (cuda_open)
    CHECK_CUDAERROR(cudaDeviceReset());

  printf("Exiting the application.\n");
  fflush(stdout);
  return 0;
}
