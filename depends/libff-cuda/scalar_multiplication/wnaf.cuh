#ifndef __WNAF_CUH__
#define __WNAF_CUH__

#include "../fields/bigint.cuh"

namespace libff{

template<mp_size_t_ n>
__device__ void find_wnaf(long** wnaf, size_t* wnaf_size, const size_t window_size, const bigint<n> &scalar);

template<typename T, mp_size_t_ n>
__device__ T fixed_window_wnaf_exp(const size_t window_size, const T &base, const bigint<n> &scalar);

template<typename T, mp_size_t_ n>
__device__ T opt_window_wnaf_exp(const T &base, const bigint<n> &scalar, const size_t scalar_bits);

}

#include "wnaf.cu"

#endif