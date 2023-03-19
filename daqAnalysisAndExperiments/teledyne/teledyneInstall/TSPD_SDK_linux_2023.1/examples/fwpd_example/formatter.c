/*
 *
 * Copyright Signal Processing Devices Sweden AB. All rights reserved.
 * See document "08-0175 EULA" for specific license terms regarding this file.
 *
 * Description   : Fix-width formatted output for printf
 * Documentation :
 *
 */

#include "formatter.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#ifndef LINUX
#include <windows.h>
#endif

static unsigned int nof_chars_on_line = 0;

int printfw(const char *format, ...)
{
  int nof_chars_written = 0;
  va_list args;
  va_start(args, format);

  nof_chars_written = vprintf(format, args);
  va_end(args);

  if (nof_chars_written > 0)
    nof_chars_on_line += nof_chars_written;

  return nof_chars_written;
}

int printfwra(const char *format, ...)
{
  size_t formatter_lw = 80;

  // Determine length of input string
  size_t nof_chars_to_write = strlen(format);

  // Compute padding
  size_t padding = formatter_lw - (size_t)nof_chars_on_line - nof_chars_to_write;

  // Print right-adjusted
  int nof_chars_written = printf("%*s\n", (int)padding, format);

  // Reset counter
  nof_chars_on_line = 0;

  return nof_chars_written;
}

void printfwrastatus(const char *str, unsigned int color)
{
#ifndef LINUX
  HANDLE hConsole;
#endif
  size_t internal_padding = (6 - strlen(str)) / 2;
  size_t padding = 70 - (size_t)nof_chars_on_line - 8;
#ifndef LINUX
  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

  SetConsoleTextAttribute(hConsole, 7);
#endif
  printf("%*s", (int)padding, "[");
#ifndef LINUX
  switch (color)
  {
  case 0:
    SetConsoleTextAttribute(hConsole, 7);
    break; // Normal
  case 1:
    SetConsoleTextAttribute(hConsole, 10);
    break; // Green
  case 2:
    SetConsoleTextAttribute(hConsole, 12);
    break; // Red
  case 3:
    SetConsoleTextAttribute(hConsole, 14);
    break; // Yellow
  default:
    SetConsoleTextAttribute(hConsole, 7);
    break; // Normal
  }
#endif
  printf("%*s", (int)(internal_padding + strlen(str)), str);
#ifndef LINUX
  SetConsoleTextAttribute(hConsole, 7);
#endif
  printf("%*s", (int)internal_padding + 2, "]\n");

  nof_chars_on_line = 0;
}
