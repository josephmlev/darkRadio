/*
 *
 * Description : Definition of ADQ14 streaming header struct
 *
 */

#ifndef LINUX
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef signed __int8  int8_t;
typedef signed __int16 int16_t;
typedef signed __int32 int32_t;
typedef signed __int64 int64_t;
#endif

typedef struct
{
  uint8_t  RecordStatus;  // Record status byte
  uint8_t  UserID;        // UserID byte
  uint8_t  Channel;       // Channel byte
  uint8_t  DataFormat;    // Data format byte
  uint32_t SerialNumber;  // Serial number (32 bits)
  uint32_t RecordNumber;  // Record number (32 bits)
  int32_t  SamplePeriod;  // Sample period (32 bits)
  uint64_t Timestamp;     // Timestamp (64 bits)
  int64_t  RecordStart;   // Record start timestamp (64 bits)
  uint32_t RecordLength;  // Record length (32 bits)
  uint16_t MovingAverage; // Moving average (16 bits)
  uint16_t GateCounter;   // Gate counter (16 bits)
} StreamingHeader_t;
