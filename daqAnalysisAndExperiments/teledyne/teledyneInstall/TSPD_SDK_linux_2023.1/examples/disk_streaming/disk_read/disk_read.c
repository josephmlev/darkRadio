/*
 * Copyright 2022 Teledyne Signal Processing Devices Sweden AB
 */

#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>
#include <signal.h>
#include "adnvds.h"

#include "ADQAPI.h"
#include "settings.h"

#ifdef LINUX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define Sleep(interval) usleep(1000 * interval)
#endif

static void *adnvds_handle = NULL;

static int initialize_disks()
{
  struct adnvds_init_params lib_init;
  struct adnvds_init_params_be_spdk be_lib_init;

  int status = -1;

  /* Initialize ADNVDS */
  status = adnvds_lib_params_init(&lib_init, (void *)&be_lib_init);
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to initialize ADNVDS parameters.\n");
    goto error;
  }

  be_lib_init.name = "disk_streaming";
  status = adnvds_lib_init(&lib_init, (void *)&be_lib_init, (void **)&adnvds_handle);
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to init ADNVDS\n");
    goto error;
  }

  /* Attach disk devices */
  for (int attempt = 0; attempt < ADNVDS_ATTACH_ATTEMPTS; attempt++)
  {
    status = adnvds_dev_attach(adnvds_handle, disk_serials);
    if (status != ADNVDS_STATUS_OK)
    {
      Sleep(ADNVDS_ATTACH_WAIT_TIME);
      if (attempt >= ADNVDS_ATTACH_ATTEMPTS - 1)
      {
        printf("Failed to attach disks, Code: %d\n", status);
        goto error;
      }
      continue;
    }
    break;
  }

  status = 0;

error:
  fflush(stdout);
  return status;
}

static int read_disks()
{
  int status = 0;

  unsigned int num_datasets = 0;
  struct adnvds_rd_data_set **data_sets = NULL;
  struct DSU7MetadataRaw *metadata_array = NULL;
  struct DSU7RecordHeader metadata_parsed_array[MAX_METADATA_BLOCK_READS_PER_DATASET];

  status = adnvds_rd_init(adnvds_handle, &data_sets, &num_datasets);
  if (status != ADNVDS_STATUS_OK)
  {
    printf("Failed to initialize ADNVDS read. Code %d\n", status);
    goto error;
  }

  printf("Total number of datasets on disks = %u\n", num_datasets);

  for (unsigned int d = 0; d < num_datasets; d++)
  {
    struct adnvds_rd_data_set *data_set = data_sets[d];
    printf("================================================================\n");
    printf("| DATA SET %u\n", d);
    printf("|\n");
    printf("| Digitizer serial number:   %s\n", data_set->source_device_serial);
    printf("| Digitizer channel number:  %u\n", data_set->channel_no);
    printf("| Amount of data stored:     %" PRIu64 " bytes\n", data_set->data_bytes_stored);
    printf("| Number of metadata blocks: %" PRIu64 "\n", data_set->num_metadata);
    printf("| Unix timestamp:            %" PRIu64 "\n", data_set->timestamp);
    printf("| Interleaved disks:         %u\n", data_set->num_interleaved_disks);
    printf("| Sequenced disks:           %u\n", data_set->num_sequenced_disks);
    printf("================================================================\n\n");


    uint64_t metadata_blocks_to_read;

    if (data_set->num_metadata <= MAX_METADATA_BLOCK_READS_PER_DATASET)
      metadata_blocks_to_read = data_set->num_metadata;
    else
      metadata_blocks_to_read = MAX_METADATA_BLOCK_READS_PER_DATASET;

    printf("Reading %" PRIu64 " metadata blocks from disk.\n\n", metadata_blocks_to_read);
    metadata_array = NULL; // adnvds_rd_get_metadata allocates a new buffer if set to NULL
    status = adnvds_rd_get_metadata(data_set, 0, metadata_blocks_to_read, (adnvds_data_t **)&metadata_array);
    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to retrieve metadata blocks. Code %d\n", status);
      goto error;
    }

    if (PARSE_RAW_METADATA)
    {
      void *adq_data_parser;
      unsigned int metadata_blocks_parsed;

      // Create ADQData object
      if (!ADQData_Create(&adq_data_parser))
      {
        printf("ADQData_Create failed\n");
        goto error;
      }
      ADQData_EnableErrorTrace(adq_data_parser, LOG_LEVEL_INFO, "./ADQData.log", 0);
      // Initialize packet stream with metadata from ADQ device struct
      if (!ADQData_InitPacketStream(adq_data_parser, data_set->adqdata_device_struct, NULL))
      {
        printf("ADQData_InitPacketStream failed\n");
        goto error;
      }
      if (!ADQData_ParseDiskStreamHeaders(adq_data_parser,
                                          (void *)metadata_array,
                                          (unsigned int)metadata_blocks_to_read,
                                          (void *)metadata_parsed_array,
                                          &metadata_blocks_parsed,
                                          data_set->channel_no))
      {
        printf("ADQData_ParseDiskStreamHeaders failed\n");
        goto error;
      }
      printf("Parsed %u raw metadata blocks into complete record headers.\n\n", metadata_blocks_parsed);
    }

    for (uint64_t m = 0; m < metadata_blocks_to_read; m++)
    {
      if (PARSE_RAW_METADATA)
      {
        printf("metadata_parsed[%" PRIu64 "].Timestamp =    %" PRIu64 "\n", m, metadata_parsed_array[m].Timestamp - metadata_parsed_array[0].Timestamp);
        printf("metadata_parsed[%" PRId64 "].RecordStart =  %" PRId64 "\n", m, metadata_parsed_array[m].RecordStart);
        printf("metadata_parsed[%" PRId64 "].RecordLength = %" PRId64 "\n", m, metadata_parsed_array[m].RecordLength);
        printf("metadata_parsed[%" PRId64 "].RecordNumber = %u\n", m, metadata_parsed_array[m].RecordNumber);
        printf("metadata_parsed[%" PRId64 "].RecordStatus = 0x%02X\n", m, metadata_parsed_array[m].RecordStatus);
        printf("metadata_parsed[%" PRId64 "].LostRecords =  %u\n", m, metadata_parsed_array[m].LostRecords);
        printf("metadata_parsed[%" PRId64 "].LostCycles =   %u\n", m, metadata_parsed_array[m].LostCycles);
      }
      else
      {
        printf("metadata_raw[%" PRIu64 "].record_number = %u\n", m, metadata_array[m].record_number);
        printf("metadata_raw[%" PRIu64 "].record_length = %" PRIu64 "\n", m, metadata_array[m].record_length);
        printf("metadata_raw[%" PRIu64 "].lost_cycles =   %u\n", m, metadata_array[m].lost_cycles);
        printf("metadata_raw[%" PRIu64 "].lost_records =  %u\n", m, metadata_array[m].lost_records);
        printf("metadata_raw[%" PRIu64 "].status =        0x%02X\n", m, metadata_array[m].status);
      }
      printf("\n");
    }

    if (data_set->num_metadata > MAX_METADATA_BLOCK_READS_PER_DATASET)
      printf("Skipping printout of remaining metadata blocks (above MAX_METADATA_BLOCK_READS_PER_DATASET)\n\n");

    status = adnvds_rd_free_data_buff(metadata_array);
    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to free metadata array memory. Code %d\n", status);
      goto error;
    }

    uint64_t stored_data_bytes_to_read;
    int16_t *stored_data = NULL;

    if (data_set->data_bytes_stored <= MAX_DATA_READ_BYTES_PER_DATASET)
      stored_data_bytes_to_read = data_set->data_bytes_stored;
    else
      stored_data_bytes_to_read = MAX_DATA_READ_BYTES_PER_DATASET;

    printf("Reading %" PRIu64 " data bytes from disk.\n\n", stored_data_bytes_to_read);
    status = adnvds_rd_get_data(data_set, 0, stored_data_bytes_to_read, (adnvds_data_t **)&stored_data);
    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to retrieve stored data. Code %d\n", status);
      goto error;
    }

    if (PRINT_DATA_BYTES)
    {
      for (uint64_t b = 0; b < (stored_data_bytes_to_read / 2); b++)
      {
        if (b > 0 && ((b % 8) == 0))
          printf("\n");
        printf("%04X ", stored_data[b]);
      }
      printf("\n");

      if (data_set->data_bytes_stored > MAX_DATA_READ_BYTES_PER_DATASET)
        printf("Skipping printout of remaining stored data (above MAX_DATA_READ_BYTES_PER_DATASET)\n");
    }

    if (WRITE_DATA_BYTES_TO_FILE)
    {
      printf("Writing data to %s\n", DATA_FILE_NAME);

      FILE* fp = fopen(DATA_FILE_NAME, "wb");

      if (fp != NULL)
      {
        fwrite(stored_data, sizeof(uint8_t), stored_data_bytes_to_read, fp);
        fclose(fp);
      }
      else
      {
        printf("Failed to open data file %s for write.\n", DATA_FILE_NAME);
      }
    }

    status = adnvds_rd_free_data_buff(stored_data);
    if (status != ADNVDS_STATUS_OK)
    {
      printf("Failed to free metadata array memory. Code %d\n", status);
      goto error;
    }
  }

error:
  if (data_sets != NULL)
    adnvds_rd_free_datasets(&data_sets, num_datasets);

  return status;
}


int main()
{
  int result = 0;

  /* Initialize disks */
  result = initialize_disks();
  if (result != 0)
  {
    printf("Initialization of disks failed.\n");
    goto exit;
  }

  /* Read disk contents */
  result = read_disks();
  if (result != 0)
  {
    printf("Readout of disks failed.\n");
    goto exit;
  }

exit:
  if(adnvds_handle != NULL)
    adnvds_lib_shutdown(adnvds_handle);

  /* Delete the control unit object and the memory allocated by this application. */
  printf("\nExiting the application.\n");
  fflush(stdout);
  return 0;
}
