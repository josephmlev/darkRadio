/*
 *
 * Copyright Signal Processing Devices Sweden AB. All rights reserved.
 * See document "08-0175 EULA" for specific license terms regarding this file.
 *
 * Description   : Fix-width formatted output for printf
 * Documentation :
 *
 */

#ifndef FORMATTER_H_
#define FORMATTER_H_

int printfw(const char* format, ...);
int printfwra(const char* format, ...);

void printfwrastatus(const char* str, unsigned int color);

#endif
