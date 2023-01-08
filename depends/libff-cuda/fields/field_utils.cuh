#ifndef __FIELD_UTILS_CUH_
#define __FIELD_UTILS_CUH__

#include "../depends/libstl-cuda/vector.cuh"

namespace libff {

template<typename FieldT>
__device__ FieldT coset_shift(const FieldT& instance);

template<typename FieldT>
__device__ FieldT get_root_of_unity(const size_t n, const FieldT& instance);


template<typename FieldT>
__device__ void batch_invert(libstl::vector<FieldT>& vec, const FieldT& instance);


}

#include "field_utils.cu"

#endif