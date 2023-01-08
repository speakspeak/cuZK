#ifndef __MULTIEXP_CUH__
#define __MULTIEXP_CUH__

// #include "wnaf.cuh"
#include "../depends/libstl-cuda/vector.cuh"
#include "../depends/libmatrix-cuda/sparse-matrix/ell.cuh"
#include "../depends/libmatrix-cuda/sparse-matrix/csr.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-vector.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-scalar.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-balanced.cuh"
#include "../depends/libmatrix-cuda/transpose/transpose_ell2csr.cuh"

namespace libff{

enum multi_exp_method {
 /**
  * Naive multi-exponentiation individually multiplies each base by the
  * corresponding scalar and adds up the results.
  * multi_exp_method_naive uses opt_window_wnaf_exp for exponentiation,
  * while multi_exp_method_plain uses operator *.
  */
 multi_exp_method_naive,
 multi_exp_method_naive_plain,
 /**
  * A variant of the Bos-Coster algorithm [1],
  * with implementation suggestions from [2].
  *
  * [1] = Bos and Coster, "Addition chain heuristics", CRYPTO '89
  * [2] = Bernstein, Duif, Lange, Schwabe, and Yang, "High-speed high-security signatures", CHES '11
  */
 multi_exp_method_bos_coster,
 /**
  * A special case of Pippenger's algorithm from Page 15 of
  * Bernstein, Doumen, Lange, Oosterwijk,
  * "Faster batch forgery identification", INDOCRYPT 2012
  * (https://eprint.iacr.org/2012/549.pdf)
  * When compiled with USE_MIXED_ADDITION, assumes input is in special form.
  * Requires that T implements .dbl() (and, if USE_MIXED_ADDITION is defined,
  * .to_special(), .mixed_add(), and batch_to_special()).
  */
 multi_exp_method_BDLO12
};


// T: curve  FieldT: field 
template<typename T, typename FieldT, multi_exp_method Method>
__device__ T multi_exp(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const size_t chunks, const T& t_instance);

template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance, size_t gridSize, size_t blockSize);

template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp_faster(libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const FieldT& instance, const T& t_instance, size_t gridSize, size_t blockSize);

template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp_faster_multi_GPU(libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const FieldT& instance, const T& t_instance, size_t gridSize, size_t blockSize);

template<typename T, typename FieldT, multi_exp_method Method>
__host__ T* p_multi_exp_faster_multi_GPU_host(libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, FieldT& instance, T& t_instance, size_t gridSize, size_t blockSize);

template<typename T, typename FieldT, multi_exp_method Method>
__host__ void p_multi_exp_faster_multi_GPU_host(T* result, libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, FieldT& instance, T& t_instance, size_t gridSize, size_t blockSize);


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T multi_exp_with_mixed_addition(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const size_t chunks, const T& t_instance);

template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp_with_mixed_addition(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const FieldT& instance,const T& t_instance, size_t gridSize, size_t blockSize);


template<typename T>
using window_table = libstl::vector<libstl::vector<T> >;

//
template<typename T>
__device__ size_t get_exp_window_size(const size_t num_scalars, const T& instance);

template<typename T>
__device__ window_table<T> get_window_table(const size_t scalar_size, const size_t window, const T &g);

template<typename T>
__device__ window_table<T>* p_get_window_table(const size_t scalar_size, const size_t window, const T& g, size_t gridSize, size_t blockSize);

template<typename T, typename FieldT>
__device__ T windowed_exp(const size_t scalar_size, const size_t window, const window_table<T>& powers_of_g, const FieldT &pow);

template<typename T, typename FieldT>
__device__ libstl::vector<T> batch_exp(const size_t scalar_size, const size_t window, const window_table<T>& table, const libstl::vector<FieldT>& vec);

template<typename T, typename FieldT>
__device__ libstl::vector<T>* p_batch_exp(const size_t scalar_size, const size_t window, const window_table<T>& table, const libstl::vector<FieldT>& vec, T& t_instance, size_t gridSize, size_t blockSize);

template<typename T, typename FieldT>
__device__ libstl::vector<T> batch_exp_with_coeff(const size_t scalar_size, const size_t window, const window_table<T>& table, const FieldT &coeff, const libstl::vector<FieldT>& vec);

template<typename T, typename FieldT>
__device__ libstl::vector<T>* p_batch_exp_with_coeff(const size_t scalar_size, const size_t window, const window_table<T>& table, const FieldT &coeff, const libstl::vector<FieldT>& vec, T& t_instance, size_t gridSize, size_t blockSize);


}

#include "multiexp.cu"

#endif