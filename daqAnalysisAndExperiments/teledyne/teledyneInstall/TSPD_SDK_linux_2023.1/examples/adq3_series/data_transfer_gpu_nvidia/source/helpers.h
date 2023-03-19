#ifndef HELPERS_H_KNWUH872
#define HELPERS_H_KNWUH872

#define CHECK_CUDAERROR(val) cudacheck((val), #val, __FILE__, __LINE__)
#define CHECK_CUDAERROR_EXIT(val) \
  if (CHECK_CUDAERROR(val))       \
  goto exit

#define CHECK_ERROR_EXIT(x, ref_val)                                                               \
  do                                                                                               \
  {                                                                                                \
    int x_val = (x);                                                                               \
    if (x_val != (ref_val))                                                                        \
    {                                                                                              \
      printf("Error: '%s' returned %d, expected %d, at line %d.\n", #x, x_val, ref_val, __LINE__); \
      goto exit_with_error;                                                                        \
    }                                                                                              \
  } while (0)

#define CHECK_ERROR_RETURN(x, ref_val, string)                                                     \
  do                                                                                               \
  {                                                                                                \
    int x_val = (x);                                                                               \
    if (x_val != (ref_val))                                                                        \
    {                                                                                              \
      printf("Error: '%s' returned %d, expected %d, at line %d.\n", #x, x_val, ref_val, __LINE__); \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

#endif // #ifndef HELPERS_H_KNWUH872
