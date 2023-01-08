#ifndef __MATRIX_UTILS_CUH__
#define __MATRIX_UTILS_CUH__

namespace libmatrix {

__host__ __device__ size_t log2(size_t n);

__device__ size_t bitreverse(size_t n, const size_t l);

__host__ __device__ size_t lsqrt(size_t n);

}

#include "utils.cu"

#endif
