//
// Created by xiehaoina on 10/9/19.
//

#ifndef DECIMAL_SUM_REDUCE_CUH_H
#define DECIMAL_SUM_REDUCE_CUH_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "decimal.h"
#include "stdio.h"


#define RECORD_PER_STREAM 256*1024*1024/sizeof(struct decimal_t)  //256 M per stream
#define BLOCK_SIZE 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void sum_decimal(decimal_t *p_decimals, decimal_t *output, unsigned int len);

#endif //DECIMAL_SUM_REDUCE_CUH_H
