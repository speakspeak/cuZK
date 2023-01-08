#ifndef __BASIC_RADIX2_DOMAIN_AUX_CUH__
#define __BASIC_RADIX2_DOMAIN_AUX_CUH__

#include "../../../depends/libstl-cuda/vector.cuh"
#include "../../../depends/libstl-cuda/utility.cuh"
#include "../../../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cuh"
//#include "../../../depends/libstl-cuda/memory.cuh"

namespace libfqfft {
/**
 * Compute the radix-2 FFT of the vector a over the set S={omega^{0},...,omega^{m-1}}.
 */
template<typename FieldT>
__device__ void _basic_radix2_FFT(libstl::vector<FieldT>& a, const FieldT& omega);


template<typename FieldT>
__device__ void _basic_parallel_radix2_FFT(libstl::vector<FieldT>& a, const FieldT& omega, size_t gridSize, size_t blockSize);

template<typename FieldT>
__device__ void _pp_basic_radix2_FFT(libstl::vector<FieldT>& a, const FieldT& omega, size_t num);

template<typename FieldT>
__host__ void _pp_basic_radix2_FFT_host(libstl::vector<FieldT>& a, const FieldT& omega, size_t num);

/**
 * Translate the vector a to a coset defined by g.
 */
template<typename FieldT>
__device__ void _multiply_by_coset(libstl::vector<FieldT>& a, const FieldT& g);


template<typename FieldT>
__device__ void _p_multiply_by_coset(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize);

template<typename FieldT>
__device__ void _pp_multiply_by_coset(libstl::vector<FieldT>& a, const FieldT& g, size_t num);

template<typename FieldT>
__host__ void _pp_multiply_by_coset_host(libstl::vector<FieldT>& a, const FieldT& g, size_t num);


/**
 * Compute the m Lagrange coefficients, relative to the set S={omega^{0},...,omega^{m-1}}, at the field element t.
 */
template<typename FieldT>
__device__ libstl::vector<FieldT> _basic_radix2_evaluate_all_lagrange_polynomials(const size_t m, const FieldT &t);


template<typename FieldT>
__device__ libstl::vector<FieldT>& _p_basic_radix2_evaluate_all_lagrange_polynomials(const size_t m, const FieldT &t, size_t gridSize, size_t blockSize);


} // libfqfft

#include "basic_radix2_domain_aux.cu"

#endif