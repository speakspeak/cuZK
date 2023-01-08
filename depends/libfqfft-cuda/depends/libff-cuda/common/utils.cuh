#ifndef __UTILS_CUH__
#define __UTILS_CUH__

namespace libff {

__host__ __device__ size_t log2(size_t n);

__device__ size_t bitreverse(size_t n, const size_t l);


}

#endif
