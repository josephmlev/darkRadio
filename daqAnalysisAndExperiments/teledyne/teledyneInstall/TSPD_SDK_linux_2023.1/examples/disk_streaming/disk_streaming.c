/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */

#include "ADQAPI.h"
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>
#include <signal.h>
#include "adnvds.h"

#include "settings.h"
#include "helpers.h"

#ifdef LINUX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
#endif


 /* Handler for CTRL+C interrupts. */
static volatile bool abort_acquisition = false;
void sigint_handler(int dummy)
{
  (void)dummy;
  printf("Caught Ctrl-C. Aborting..\n");
  abort_acquisition = true;
}

static void process_monitor_channel_data(
  const struct ADQRecord *const record,
  const struct ADQDataReadoutStatus *status,
  int channel,
  int64_t bytes_received)
{
  /* Any data processing and analysis for the monitor channel data would
     be performed here. By default, this example does nothing with the data. */
  (void)record;
  (void)status;
  (void)channel;
  (void)bytes_received;
}

static void streaming(void *adq_cu, int adq_num, struct DiskStorage *storage)
{
  /* We need the following parameters to direct the flow in this functions. */
  int nof_channels = (int)ADQ_GetNofChannels(adq_cu, adq_num);
  struct ADQDataAcquisitionParameters acquisition;

  int64_t stored_bytes_data[ADQ_MAX_NOF_CHANNELS] = { 0 };
  int64_t stored_bytes_metadata[ADQ_MAX_NOF_CHANNELS] = { 0 };

  int64_t datarate_tracker_data_bytes[ADQ_MAX_NOF_CHANNELS] = { 0 };
  int64_t datarate_tracker_metadata_bytes[ADQ_MAX_NOF_CHANNELS] = { 0 };
  int64_t datarate_tracker_monitor_bytes[ADQ_MAX_NOF_CHANNELS] = { 0 };


  timer_start(TIMER_NO_DATA_RECEIVED);
  timer_start(TIMER_PERIODIC_PRINTOUT);
  timer_start(TIMER_TOTAL_ACQUISITION);

  int nof_received_records[ADQ_MAX_NOF_CHANNELS] = { 0 };

  int result = ADQ_GetParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_ACQUISITION, &acquisition);
  if (result != sizeof(acquisition))
  {
    printf("Failed to get the data acquisition parameters.\n");
    return;
  }

  /* Start the data acquisition. */
  printf("Start acquiring data... ");
  result = ADQ_StartDataAcquisition(adq_cu, adq_num);
  if (result != ADQ_EOK)
  {
    printf("failed, code %d.\n", result);
    goto exit;
  }
  printf("success.\n");

  /* Send software triggers. This structure sends more triggers than required
     since all channels share a common source. */
  for (int ch = 0; ch < ADQ_MAX_NOF_CHANNELS; ++ch)
  {
    if (acquisition.channel[ch].trigger_source == ADQ_EVENT_SOURCE_SOFTWARE)
    {
      if (acquisition.channel[ch].nof_records != 0)
        printf("Generating software events on channel %d.\n", ch);
      for (int i = 0; i < acquisition.channel[ch].nof_records; ++i)
        ADQ_SWTrig(adq_cu, adq_num);
    }
  }

  /* Data readout loop. */
  bool done = false;
  bool flush_issued = false;

  while (!done && !abort_acquisition)
  {
    /* ========= Handle disk streaming ======== */
    for (int ch = 0; ch < nof_channels; ch++)
    {
      if (!DISK_STREAMING_RECORD_ENABLED[ch])
        continue;

      /* Poll and update I/O queues for the disks */
      if (adnvds_wr_poll(storage->adnvds_handle, storage->ep_group_handle[ch]) < 0)
      {
        printf("Failed to poll ongoing ADNVDS operations.\n");
        goto stop_then_exit;
      }
    }

    bool disk_limit_reached = true;

    int64_t new_bytes_received = 0;

    /* Check for new bytes stored on disk for each channel */
    for (int ch = 0; ch < nof_channels; ch++)
    {
      if (!DISK_STREAMING_RECORD_ENABLED[ch])
        continue;

      int64_t stored_bytes_data_ch = 0;
      int64_t stored_bytes_metadata_ch = 0;

      result = adnvds_wr_status_transfer(
        storage->adnvds_handle,
        storage->ep_group_handle[ch],
        &stored_bytes_data_ch,
        &stored_bytes_metadata_ch
      );
      if (result != ADNVDS_STATUS_OK)
      {
        printf("Failed to retrieve ADNVDS write transfer status.\n");
        goto stop_then_exit;
      }

      datarate_tracker_data_bytes[ch] += (stored_bytes_data_ch - stored_bytes_data[ch]);
      datarate_tracker_metadata_bytes[ch] += (stored_bytes_metadata_ch - stored_bytes_metadata[ch]);

      new_bytes_received += (stored_bytes_data_ch - stored_bytes_data[ch]);
      new_bytes_received += (stored_bytes_metadata_ch - stored_bytes_metadata[ch]);

      stored_bytes_data[ch] = stored_bytes_data_ch;
      stored_bytes_metadata[ch] = stored_bytes_metadata_ch;

      if (stored_bytes_data[ch] < DISK_STORAGE_LIMIT_BYTES[ch])
        disk_limit_reached = false;
    }

    if (disk_limit_reached && !flush_issued)
    {
      printf("Reached disk data limit, issuing flush.\n");
      ADQ_FlushDMA(adq_cu, adq_num);
      flush_issued = true;
      timer_start(TIMER_NO_DATA_RECEIVED);
    }


    /* If no new bytes have been stored for TIMEOUT_DSU_TRANSFER amount of time,
       flush once. If there's still no new bytes after flushing, we're either
       at the end of the data acquisition or something has gone wrong, and
       we should exit. */
    if (new_bytes_received > 0)
    {
      timer_start(TIMER_NO_DATA_RECEIVED);
    }
    else
    {
      if (timer_time_seconds(TIMER_NO_DATA_RECEIVED) > TIMEOUT_DSU_TRANSFER)
      {
        if (!flush_issued)
        {
          printf("Zero new bytes received last %.2f seconds, issuing flush.\n", TIMEOUT_DSU_TRANSFER);
          ADQ_FlushDMA(adq_cu, adq_num);
          flush_issued = true;
          timer_start(TIMER_NO_DATA_RECEIVED);
        }
        else
        {
          printf("Zero new bytes received last %.2f seconds, stopping acquisition.\n", TIMEOUT_DSU_TRANSFER);
          done = true;
        }
      }
    }

    /* ========= Handle monitoring channel data ======== */

    /* Wait for a record buffer on any channel. */
    struct ADQDataReadoutStatus status = { 0 };
    struct ADQRecord *record = NULL;
    int channel = ADQ_ANY_CHANNEL;
    int64_t bytes_received = ADQ_WaitForRecordBuffer(
      adq_cu, adq_num, &channel, (void **)(&record), 0, &status
    );

    /* Negative values are errors. Zero bytes received indicates a successful
       call, but that only the status parameter can be read. */
    if (bytes_received > 0)
    {
      /* Process the data. */
      process_monitor_channel_data(record, &status, channel, bytes_received);

      datarate_tracker_monitor_bytes[channel] += bytes_received;

      /* Return the buffer to the API. */
      result = ADQ_ReturnRecordBuffer(adq_cu, adq_num, channel, record);
      if (result != ADQ_EOK)
      {
        printf("Failed to return a record buffer, code %d.\n", result);
        break;
      }

      /* Check if the acquisition should end. We only increment the counter if the */
      if (!(status.flags & ADQ_DATA_READOUT_STATUS_FLAGS_INCOMPLETE))
        ++nof_received_records[channel];

    }
    else if (bytes_received < 0 && bytes_received != ADQ_EAGAIN)
    {
      printf("Waiting for a record buffer failed, code '%" PRId64 "'.\n", bytes_received);
      goto stop_then_exit;
    }


    /* ========= Periodic status printout ======== */
    double interval_sec = timer_time_seconds(TIMER_PERIODIC_PRINTOUT);
    if (interval_sec > PERIODIC_STATUS_PRINT_TIME || done)
    {
      print_status(
        interval_sec,
        nof_channels,
        ADQ_GetStreamOverflow(adq_cu, adq_num),
        datarate_tracker_data_bytes,
        datarate_tracker_metadata_bytes,
        datarate_tracker_monitor_bytes,
        stored_bytes_data,
        stored_bytes_metadata
      );
      for (int ch = 0; ch < ADQ_MAX_NOF_CHANNELS; ch++)
      {
        if (!DISK_STREAMING_RECORD_ENABLED[ch])
          continue;

        for (int disk = 0; disk < storage->n_drives_per_channel[ch]; disk++)
        {
          print_disk_health_info(storage->adnvds_handle, storage->serials[ch][disk]);
        }
      }
      timer_start(TIMER_PERIODIC_PRINTOUT);
    }
  }

stop_then_exit:

  /* Finalize the disk metadata */
  result = adnvds_wr_finish(storage->adnvds_handle);
  if (result != ADNVDS_STATUS_OK) {
    printf("Failed finalize disk metadata.\n");
  }

  /* Stop the data acquisition process. */
  printf("Stopping data acquisition... ");
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

  for (int ch = 0; ch < nof_channels; ++ch)
  {
    if (!DISK_STREAMING_RECORD_ENABLED[ch])
      continue;

    /* Stop any pending transfers */
    result = adnvds_wr_stop_transfer(
      storage->adnvds_handle, storage->ep_group_handle[ch]
    );

    if (result != ADNVDS_STATUS_OK) {
      printf("Failed to stop pending ADNVDS transfers.\n");
    }

    /* Free the transfer structs */
    result = adnvds_wr_unregister_transfer(
      storage->adnvds_handle, storage->ep_group_handle[ch]
    );

    if (result != ADNVDS_STATUS_OK) {
      printf("Failed to unregister ADNVDS transfers.\n");
    }
  }

  int overflow_status = ADQ_GetStreamOverflow(adq_cu, adq_num);
  if (overflow_status != 0)
    printf("The device reports an overflow condition in the monitoring channels.\n");

exit:
  return;
}

static int configure_acquisition(void *adq_cu, int adq_num)
{
  /* Read out static configuration. */
  int nof_channels = (int)ADQ_GetNofChannels(adq_cu, adq_num);

  /* Periodic event generator. */
  printf("Configuring the periodic event source...");
  if (!ADQ_SetInternalTriggerPeriod(adq_cu, adq_num, PERIODIC_EVENT_SOURCE_PERIOD))
  {
    printf("failed.\n");
    return -1;
  }
  printf("success.\n");

  /* Sample skip */
  for (int ch = 0; ch < nof_channels; ++ch)
  {
    printf("Configuring sample skip for channel %d...", ch);
    if (!ADQ_SetChannelSampleSkip(adq_cu, adq_num, ch + 1, SAMPLE_SKIP_FACTOR[ch]))
    {
      printf("failed.\n");
      return -1;
    }
    printf("success.\n");
  }

  /* Initialize data acquisition parameters. */
  struct ADQDataAcquisitionParameters acquisition;
  int result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_ACQUISITION, &acquisition);
  if (result != sizeof(acquisition))
  {
    printf("Failed to initialize data acquisition parameters.\n");
    return -1;
  }

  /* Initialize data transfer parameters. */
  struct ADQDataTransferParameters transfer;
  result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_TRANSFER, &transfer);
  if (result != sizeof(transfer))
  {
    printf("Failed to initialize data transfer parameters.\n");
    return -1;
  }

  /* Initialize data readout parameters. */
  struct ADQDataReadoutParameters readout = { 0 };
  result = ADQ_InitializeParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_READOUT, &readout);
  if (result != sizeof(readout))
  {
    printf("Failed to initialize data readout parameters.\n");
    return -1;
  }

  /* Adjust the transfer buffer size if needed. These are the default values: */
  transfer.channel[0].nof_buffers = 8;
  transfer.channel[0].record_buffer_size = 512 * 1024;

  /* Configure the acquisition parameters. */
  for (int ch = 0; ch < ADQ_MAX_NOF_CHANNELS; ++ch)
  {
    acquisition.channel[ch].horizontal_offset = HORIZONTAL_OFFSET[ch];
    acquisition.channel[ch].trigger_edge = TRIGGER_EDGE[ch];
    acquisition.channel[ch].trigger_source = TRIGGER_SOURCE[ch];
    acquisition.channel[ch].record_length = RECORD_LENGTH[ch];
    acquisition.channel[ch].nof_records = NOF_RECORDS[ch];
    acquisition.channel[ch].dsu_forced_metadata_interval = FORCED_METADATA_INTERVAL[ch];
  }

  /* Configure the readout parameters. */
  for (int ch = 0; ch < ADQ_MAX_NOF_CHANNELS; ++ch)
  {
    if (acquisition.channel[ch].record_length == ADQ_INFINITE_RECORD_LENGTH)
      readout.channel[ch].incomplete_records_enabled = 1;
  }

  /* Write the parameters to the digitizer. */
  printf("Configuring data acquisition... ");
  result = ADQ_SetParameters(adq_cu, adq_num, &acquisition);
  if (result != sizeof(acquisition))
  {
    printf("failed, code %d. See the log file for more information.\n", result);
    return -1;
  }
  printf("success.\n");

  printf("Configuring data transfer... ");
  result = ADQ_SetParameters(adq_cu, adq_num, &transfer);
  if (result != sizeof(transfer))
  {
    printf("failed, code %d. See the log file for more information.\n", result);
    return -1;
  }
  printf("success.\n");

  printf("Configuring data readout... ");
  result = ADQ_SetParameters(adq_cu, adq_num, &readout);
  if (result != sizeof(readout))
  {
    printf("failed, code %d. See the log file for more information.\n", result);
    return -1;
  }
  printf("success.\n");

  printf("Configuring trigger source for monitoring channels... ");
  if (!ADQ_SetAuxTriggerMode(adq_cu, adq_num, MONITOR_CHANNELS_TRIGGER_SOURCE))
  {
    printf("failed.\n");
    return -1;
  }
  printf("success.\n");

  return 0;
}

static uint64_t calculate_metadata_start_lba(
  struct ADQDataAcquisitionParametersChannel *acquisition_ch,
  const int n_drives_per_channel,
  const int drive_interleaving,
  const uint64_t metadata_interval,
  const uint64_t disk_capacity)
{
  // Check the amount of record samples per metadata transfer
  //uint64_t metadata_interval = acquisition_ch->dsu_forced_metadata_interval;

  uint64_t ns_size = disk_capacity * ADNVDS_LBA_SIZE; // namespace size in bytes
  uint64_t cmd_sz_mask = ~((uint64_t)ADNVDS_COMMAND_SIZE - 1); // for truncating to nearest CMD size
  uint64_t metadata_start_lba = 0;

  // Reserve at least 2 * ADNVDS_COMMAND_SIZE bytes for data and metadata sectors respectively
  // on each disk, for the transfer metadata.
  uint64_t reserved_margin = 2 * ADNVDS_COMMAND_SIZE;

  uint64_t total_data_size = acquisition_ch->nof_records * acquisition_ch->record_length * sizeof(int16_t);
  uint64_t total_metadata_size = acquisition_ch->nof_records * acquisition_ch->record_length / metadata_interval * sizeof(struct DSU7MetadataRaw);

  if (total_data_size + total_metadata_size + 2 * reserved_margin * n_drives_per_channel > ns_size * n_drives_per_channel &&
    acquisition_ch->nof_records != ADQ_INFINITE_NOF_RECORDS &&
    acquisition_ch->record_length != ADQ_INFINITE_RECORD_LENGTH)
  {
    printf("Error: the specified amount of data %.03lf + metadata %.03lf + margin %.03lf = %.03lf > total drive capacity %.03lf [GB]\n",
      (double)total_data_size / 1e9,
      (double)total_metadata_size / 1e9,
      (double)2 * reserved_margin * n_drives_per_channel / 1e9,
      (double)(total_data_size + total_metadata_size + 2 * reserved_margin * n_drives_per_channel) / 1e9,
      (double)ns_size * n_drives_per_channel / 1e9
    );
    return 0;
  }

  // Expect filling whole drive(s), at least first drive(s) in sequence
  if (acquisition_ch->nof_records == ADQ_INFINITE_NOF_RECORDS ||
    acquisition_ch->record_length == ADQ_INFINITE_RECORD_LENGTH ||
    total_data_size > ns_size * drive_interleaving)
  {
    // Set positions, always reserve at least 2 CMDs for data/metdata
    uint64_t metadata_size_per_drive = (uint64_t)(ns_size * sizeof(struct DSU7MetadataRaw)) / (metadata_interval * sizeof(int16_t));
    metadata_size_per_drive = (metadata_size_per_drive + reserved_margin + (ADNVDS_COMMAND_SIZE - 1)) & cmd_sz_mask; // +2 and round up to nearest CMD
    metadata_start_lba = disk_capacity - metadata_size_per_drive / ADNVDS_LBA_SIZE;
  }
  else // Limited acquisition, put metadata right after data
  {
    metadata_start_lba = (total_data_size + reserved_margin + (ADNVDS_COMMAND_SIZE - 1)) & cmd_sz_mask; // +2 and round up to nearest CMD
    metadata_start_lba /= ADNVDS_LBA_SIZE;
  }

  printf(
    "Metadata start at LBA %" PRIu64 " -> %.03lf GB for data and %.03lf GB for metadata per drive\n",
    metadata_start_lba,
    (double)(metadata_start_lba * ADNVDS_LBA_SIZE - ADNVDS_COMMAND_SIZE) / 1e9,
    (double)((disk_capacity - metadata_start_lba) * ADNVDS_LBA_SIZE - ADNVDS_COMMAND_SIZE) / 1e9
  );

  return metadata_start_lba;
}

static int configure_storage(void *adq_cu, int adq_num, struct DiskStorage *storage)
{
  struct adnvds_init_params lib_init;
  struct adnvds_init_params_be_spdk be_lib_init;

  uint64_t min_disk_capacity;
  int status = -1;
  storage->ep_group_handle = calloc(ADQ_MAX_NOF_CHANNELS, sizeof(unsigned int));
  struct ADQDataAcquisitionParameters acquisition;

  status = ADQ_GetParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_ACQUISITION, &acquisition);
  if (status != sizeof(acquisition))
  {
    printf("Failed to get the data acquisition parameters.\n");
    goto error;
  }

  /* Initialize ADNVDS */
  status = adnvds_lib_params_init(&lib_init, (void *)&be_lib_init);
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to initialize ADNVDS parameters.\n");
    goto error;
  }

  be_lib_init.name = "disk_streaming";
  status = adnvds_lib_init(&lib_init, (void *)&be_lib_init, (void **)&(storage->adnvds_handle));
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to init ADNVDS\n");
    goto error;
  }

#ifdef LINUX
  for (unsigned int ch = 0; ch < ADQ_GetNofChannels(adq_cu, adq_num); ch++)
  {
    if (!DISK_STREAMING_RECORD_ENABLED[ch])
      continue;

    /* Attach disk devices */
    for (int attempt = 0; attempt < ADNVDS_ATTACH_ATTEMPTS; attempt++)
    {
      status = adnvds_dev_attach(storage->adnvds_handle, storage->serials[ch]);
      if (status != ADNVDS_STATUS_OK)
      {
        Sleep(ADNVDS_ATTACH_WAIT_TIME);
        if (attempt >= ADNVDS_ATTACH_ATTEMPTS - 1)
        {
          printf("Failed to attach disks for ch %u, Code: %d\n",
            ch, status);
          goto error;
        }
        continue;
      }
      break;
    }
#else
  // Windows single attach call workaround
  char *serials_windows_workaround[10];
  unsigned int cnt = 0;
  for (unsigned int ch = 0; ch < ADQ_GetNofChannels(adq_cu, adq_num); ch++)
    for (int diskno = 0; diskno < storage->n_drives_per_channel[ch]; diskno++) {
      serials_windows_workaround[cnt] = storage->serials[ch][diskno];
      cnt += 1;
    }

  serials_windows_workaround[cnt] = NULL;
  status = adnvds_dev_attach(storage->adnvds_handle, serials_windows_workaround);
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to attach disks\n");
    goto error;
  }
  for (unsigned int ch = 0; ch < ADQ_GetNofChannels(adq_cu, adq_num); ch++)
  {
    if (!DISK_STREAMING_RECORD_ENABLED[ch])
      continue;
#endif

    /* Retrieve storage capacity for each disk */
    min_disk_capacity = ~0u;
    for (int n = 0; n < storage->n_drives_per_channel[ch]; n++)
    {
      uint64_t disk_capacity;

      if (!storage->serials[ch][n])
        continue;

      status = adnvds_dev_get_capacity(storage->adnvds_handle,
        (void *)storage->serials[ch][n],
        &disk_capacity);
      if (status != ADNVDS_STATUS_OK)
      {
        printf("Failed to retrieve disk namespace capacity\n");
        goto error;
      }

      printf("Disk %s - Storage capacity ", storage->serials[ch][n]);
      print_bytes((double)disk_capacity * ADNVDS_LBA_SIZE);
      printf("\n");

      if (disk_capacity < min_disk_capacity)
        min_disk_capacity = disk_capacity;
    }

    /* Calculate appropriate partitioning between data and metadata for the
       disks */
    uint64_t metadata_start_lba = calculate_metadata_start_lba(
      &acquisition.channel[ch],
      storage->n_drives_per_channel[ch],
      storage->drive_interleaving,
      FORCED_METADATA_INTERVAL[ch],
      min_disk_capacity
    );

    status = adnvds_wr_register_transfer(
      storage->adnvds_handle,
      storage->serials[ch],
      storage->n_drives_per_channel[ch],
      metadata_start_lba,
      FORMAT_DRIVE[ch],
      CHANNEL_UUID[ch],
      &(storage->ep_group_handle[ch])
    );

    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed ADNVDS register transfer. Code %d\n", status);
      goto error;
    }

    status = adnvds_wr_set_group_interleave(storage->adnvds_handle, storage->ep_group_handle[ch], storage->drive_interleaving);

    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to set ADNVDS endpoint group interleaving. Code %d\n", status);
      goto error;
    }

    uint64_t bar_address;
    unsigned int bar_size_mib;
    unsigned int read_size_min;
    unsigned int read_size_max;
    unsigned int nof_endpoints_max = 0;
    unsigned int nof_data_and_metadata_dsu_channels;

    if (!ADQ_GetDSUParameters(adq_cu, adq_num, &bar_address, &bar_size_mib, &read_size_min,
      &read_size_max, &nof_endpoints_max, &nof_data_and_metadata_dsu_channels))
    {
      printf("Failed to retrieve FWDSU parameters for digitizer device %d.\n", adq_num);
      status = -1;
      goto error;
    }

    printf("Initializing disk transfer..\n");
    status = adnvds_wr_set_ep_group_source(storage->adnvds_handle, storage->ep_group_handle[ch], adq_cu, adq_num, ch, (uintptr_t)bar_address);
    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to set ADNVDS endpoint group source for channel %u Code: %d\n", ch, status);
      goto error;
    }

    status = adnvds_wr_init_transfer(storage->adnvds_handle, storage->ep_group_handle[ch]);
    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to initialize ADNVDS endpoint group disks for channel %u. Code: %d\n", ch, status);
      goto error;
    }
  }

  /* Retrieve DSU write parameters */
  unsigned int active_channels_in_mask = 0;
  for (unsigned int ch = 0; ch < ADQ_GetNofChannels(adq_cu, adq_num); ch++)
  {
    if (DISK_STREAMING_RECORD_ENABLED[ch])
      active_channels_in_mask |= (0x1u << ch);
  }

  struct adqapi_setup_params *params = NULL;
  status = adnvds_wr_get_ADQ_DSU_setup_params(storage->adnvds_handle, &params, active_channels_in_mask);
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to retrieve DSU setup parameters from ADNVDS.\n");
    goto error;
  }

  /* Update transfer parameters with storage information */
  struct ADQDataTransferParameters transfer;
  status = ADQ_GetParameters(adq_cu, adq_num, ADQ_PARAMETER_ID_DATA_TRANSFER, &transfer);
  if (status != sizeof(transfer))
  {
    printf("Failed to retrieve data transfer parameters.\n");
    goto error;
  }

  transfer.common.dsu_doorbell_value_mask = (unsigned int)((params->doorbell_wrap_depth) - 1);
  transfer.common.dsu_operation_size = ADNVDS_COMMAND_SIZE;

  for (int ch = 0; ch < ADQ_MAX_NOF_CHANNELS; ch++)
  {
    transfer.channel[ch].record_enabled = HOST_STREAMING_RECORD_ENABLED[ch];
    transfer.channel[ch].dsu_record_enabled = DISK_STREAMING_RECORD_ENABLED[ch];
    transfer.channel[ch].dsu_metadata_enabled = DISK_STREAMING_METADATA_ENABLED[ch];

    if (DISK_STREAMING_RECORD_ENABLED[ch])
      transfer.channel[ch].dsu_record_enabled_endpoints_mask = params->ch_masks[ch * 2];

    if (DISK_STREAMING_METADATA_ENABLED[ch])
      transfer.channel[ch].dsu_metadata_enabled_endpoints_mask = params->ch_masks[ch * 2 + 1];
  }

  status = ADQ_SetParameters(adq_cu, adq_num, &transfer);
  if (status != sizeof(transfer))
  {
    printf("Failed to set transfer parameters, code %d. See the log file for more information.\n", status);
    goto error;
  }

  for (int ch = 0; ch < (int)ADQ_GetNofChannels(adq_cu, adq_num); ch++)
  {
    if (!DISK_STREAMING_RECORD_ENABLED[ch])
      continue;

    for (int d = 0; d < storage->drive_interleaving * 2; d++)
    {
      int didx = ch * storage->drive_interleaving * 2 + d;

      if (!ADQ_DSUUpdateDoorbellAddress(adq_cu, adq_num, didx, 0, params->db_addr[didx * 2]))
      {
        printf("Failed to update doorbell address.\n");
        status = -1;
        goto error;
      }
    }
  }

  status = 0;

error:
  fflush(stdout);
  return status;
  }

static int initialize_target_device(void* adq_cu)
{
  /* List the available devices connected to the host computer. */
  struct ADQInfoListEntry *adq_list = NULL;
  unsigned int nof_devices = 0;

  if (!ADQControlUnit_ListDevices(adq_cu, &adq_list, &nof_devices))
  {
    printf("ListDevices failed.\n");
    return -1;
  }

  if (nof_devices == 0)
  {
    printf("No devices found.\n");
    return -1;
  }

  for (unsigned int i = 0; i < nof_devices; ++i)
  {
    printf("Device #%u - ", i);
    switch (adq_list[i].ProductID)
    {
    case PID_ADQ7:
      printf("ADQ7\n");
      break;
    default:
      printf("Unsupported\n");
      break;
    }
  }

  int device_to_open = 0;
  if (nof_devices > 1)
  {
    for (;;)
    {
      printf("\nSelect target device: ");
      scanf("%d", &device_to_open);

      if ((device_to_open < 0) || (device_to_open >= (int)(nof_devices)))
      {
        printf("Invalid device '%d', valid range is [0, %u].\n", device_to_open, nof_devices - 1);
      }
      else
      {
        break;
      }
    }
  }

  printf("Initializing device #%d...", device_to_open);
  if (ADQControlUnit_SetupDevice(adq_cu, device_to_open))
  {
    printf("success.\n");
  }
  else
  {
    printf("failed.\n");
    return -1;
  }

  if (ADQ_HasFeature(adq_cu, 1, "FWDSU") != 1)
  {
    printf("The digitizer is not running a FWDSU firmware, or is missing the FWDSU license.\n");
    return -1;
  }

  return 1;
}


int main()
{
  /* Connect handler for CTRL+C interrupts. */
  signal(SIGINT, sigint_handler);

  int revision = ADQAPI_GetRevision();
  printf("Disk streaming example for ADQ7.\n");
  printf("API Revision: %6d\n", revision);

  uint32_t adnvds_rev[2];
  adnvds_info_get_revision(adnvds_rev, NULL);
  printf("ADNVDS Revision: %d\n", adnvds_rev[1]);

  /* Initialize the handle to the ADQ control unit object. */
  void *adq_cu = CreateADQControlUnit();
  if (!adq_cu)
  {
    printf("Failed to create the ADQ control unit.\n");
    return -1;
  }

  /* Validate ADQAPI version. */
  switch (ADQAPI_ValidateVersion(ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR))
  {
  case 0:
    // ADQAPI is compatible
    break;
  case -1:
    printf(
      "ADQAPI version is incompatible. The application needs to be recompiled "
      "and relinked against the installed ADQAPI.\n"
    );
    return -1;
  case -2:
    printf(
      "ADQAPI version is backwards compatible. It's suggested to recompile and "
      "relink the application against the installed ADQAPI.\n");
    break;
  }

  /* Enable the error trace log. */
  ADQControlUnit_EnableErrorTrace(adq_cu, LOG_LEVEL_INFO, ".");

  /* Initialize the target digitizer device */
  int adq_num = initialize_target_device(adq_cu);
  if (adq_num <= 0)
  {
    printf("Failed to initialize target device.\n");
    goto exit;
  }

  /* Configure data acquisition */
  int result = configure_acquisition(adq_cu, adq_num);
  if (result != 0)
  {
    printf("Configuration (acquisition) failed.\n");
    goto exit;
  }

  /* Configure disk storage */
  result = configure_storage(adq_cu, adq_num, &disk_information);
  if (result != 0)
  {
    printf("Configuration (storage) failed.\n");
    goto exit;
  }

  /* Start data acquisition.  */
  streaming(adq_cu, adq_num, &disk_information);

exit:
  if (disk_information.adnvds_handle != NULL)
    adnvds_lib_shutdown(disk_information.adnvds_handle);

  /* Delete the control unit object and the memory allocated by this application. */
  DeleteADQControlUnit(adq_cu);
  printf("Exiting the application.\n");
  fflush(stdout);
  return 0;
}
