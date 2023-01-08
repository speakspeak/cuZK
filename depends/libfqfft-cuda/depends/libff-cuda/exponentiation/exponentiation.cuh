#ifndef __EXPONENTIATION_CUH__
#define __EXPONENTIATION_CUH__

#include "../fields/bigint.cuh"
#include "../fields/bigint_host.cuh"

namespace libff {

template<typename FieldT, mp_size_t_ m>
__device__ FieldT power(const FieldT &base, const bigint<m> &exponent);

template<typename FieldT>
__device__ FieldT power(const FieldT &base, const unsigned long exponent);

template<typename FieldT, mp_size_t m>
__host__ FieldT power_host(const FieldT &base, const bigint_host<m> &exponent);

template<typename FieldT>
__host__ FieldT power_host(const FieldT &base, const unsigned long exponent);

}

#include "exponentiation.cu"

#endif



