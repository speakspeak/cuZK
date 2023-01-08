#ifndef __KRONECKER_SUBSTITUTION_CUH__
#define __KRONECKER_SUBSTITUTION_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace libfqfft {

/**
 * Given two polynomial vectors, A and B, the function performs
 * polynomial multiplication and returns the resulting polynomial vector.
 * The implementation makes use of
 * [Harvey 07, Multipoint Kronecker Substitution, Section 2.1] and
 * [Gathen and Gerhard, Modern Computer Algebra 3rd Ed., Section 8.4].
 */
template<typename FieldT>
__device__ void  kronecker_substitution(libstl::vector<FieldT>& v3, const libstl::vector<FieldT>& v1, const libstl::vector<FieldT>& v2);

} // libfqfft

#include "kronecker_substitution.cu"

#endif