#ifndef __CURVE_UTILS_CUH__
#define __CURVE_UTILS_CUH__


#include "../fields/bigint.cuh"

namespace libff{

template<typename GroupT, mp_size_t_ m>
__device__ GroupT scalar_mul(const GroupT &base, const bigint<m> &scalar);


template<typename GroupT>
__device__ GroupT scalar_mul(const GroupT &base, const unsigned long scalar);

}

#include "curve_utils.cu"


#endif 