#ifndef __CSR_SCALAR_CUH__
#define __CSR_SCALAR_CUH__

#include "../sparse-matrix/csr.cuh"
#include "../depends/libstl-cuda/memory.cuh"

namespace libmatrix
{

template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_scalar(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize);


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_scalar(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize);


template<typename T, typename S>
__host__ libstl::vector<T>* p_spmv_csr_scalar_host(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize);


template<typename T>
__host__ libstl::vector<T>* p_spmv_csr_scalar_host(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize);


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize);


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize);


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize);


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize);


template<typename T, typename S>
__host__ libstl::vector<T>* p_spmv_csr_scalar_vector_one_host(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize);


template<typename T>
__host__ libstl::vector<T>* p_spmv_csr_scalar_vector_one_host(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize);


}

#include "csr-scalar.cu"

#endif