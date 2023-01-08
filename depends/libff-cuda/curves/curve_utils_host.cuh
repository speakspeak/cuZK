#ifndef __CURVE_UTILS_HOST_CUH__
#define __CURVE_UTILS_HOST_CUH__


#include "../fields/bigint_host.cuh"

namespace libff{

template<typename GroupT, mp_size_t m>
GroupT scalar_mul_host(const GroupT &base, const bigint_host<m> &scalar);


template<typename GroupT>
GroupT scalar_mul_host(const GroupT &base, const unsigned long scalar);

}

#include "curve_utils_host.cu"


#endif 