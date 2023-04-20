#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H


#define ASSERT(x)                                                       \
    do                                                                  \
        {                                                               \
            if (!(x))                                                   \
                {                                                       \
                    fprintf(stdout, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
                    /*exit(EXIT_FAILURE);*/                                 \
                }                                                       \
        } while (0)

#define ASSERTDRV(stmt)				\
    do                                          \
        {                                       \
            CUresult result = (stmt);           \
            ASSERT(CUDA_SUCCESS == result);     \
        } while (0)

#define ASSERTRT(stmt)				\
    do                                          \
        {                                       \
            cudaError_t result = (stmt);           \
            ASSERT(cudaSuccess == result);     \
        } while (0)

#endif
