/*
 *  Copyright 2019 Teledyne Signal Processing Devices Sweden AB
 */

#ifndef UTILS_H_9WCQ7EUD
#define UTILS_H_9WCQ7EUD


enum FileWriterMode {
  FWM_DISABLE = 0,
  FWM_ASCII = 1,
  FWM_BINARY = 2,
};


int atd_file_writer(enum FileWriterMode write_mode, void *data_buffer,
                      unsigned int samples_to_write,
                      unsigned int bytes_per_sample, FILE *file);

#endif /* end of include guard: UTILS_H_9WCQ7EUD */
