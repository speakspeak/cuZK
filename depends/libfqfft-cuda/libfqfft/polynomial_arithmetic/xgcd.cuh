#ifndef __XGCD_CUH__
#define __XGCD_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace libfqfft {

/**
 * Perform the standard Extended Euclidean Division algorithm.
 * Input: Polynomial A, Polynomial B.
 * Output: Polynomial G, Polynomial U, Polynomial V, such that G = (A * U) + (B * V).
 */
template<typename FieldT>
__device__ void _polynomial_xgcd(const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b, libstl::vector<FieldT>& g, libstl::vector<FieldT>& u, libstl::vector<FieldT>& v);

} // libfqfft

#include "xgcd.cu"

#endif
