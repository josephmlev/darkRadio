/*
 *  Copyright 2019 Teledyne Signal Processing Devices Sweden AB
 */

#include <stdio.h>
#include "utils.h"

/* Helper function to write data to file */
int atd_file_writer(enum FileWriterMode write_mode, void *data_buffer,
                      unsigned int samples_to_write,
                      unsigned int bytes_per_sample, FILE *file)
{
  unsigned int i;

  if (!file)
  {
    printf("ERROR: Invalid reference to memory 'file'\n");
    return -1;
  }

  switch (write_mode)
  {
  case FWM_ASCII: /* Write ASCII */
    for (i = 0; i < samples_to_write; ++i)
      switch (bytes_per_sample)
      {
      case sizeof(char):
        fprintf(file, "%d\n", ((char *)data_buffer)[i]);
        break;
      case sizeof(short):
        fprintf(file, "%d\n", ((short *)data_buffer)[i]);
        break;
      case sizeof(int):
        fprintf(file, "%d\n", ((int *)data_buffer)[i]);
        break;
      case sizeof(long long int):
        fprintf(file, "%lld\n", ((long long int *)data_buffer)[i]);
        break;
      default:
        printf("WARNING: Unsupported number of bytes per sample.\n");
        break;
      }
    break;
  case FWM_BINARY: /* Write binary */
    fwrite(data_buffer, bytes_per_sample, samples_to_write, file);
    break;
  case FWM_DISABLE:
    break;
  default:
    printf("WARNING: Unsupported write mode. No file will be written\n");
    return -1;
  }

  return 0;
}
