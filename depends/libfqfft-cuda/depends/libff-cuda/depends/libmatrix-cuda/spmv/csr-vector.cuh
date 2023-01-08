#ifndef __CSR_VECTOR_CUH__
#define __CSR_VECTOR_CUH__

#include "../sparse-matrix/csr.cuh"
#include "../depends/libstl-cuda/memory.cuh"

namespace libmatrix
{


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_vector(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize);


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_vector(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize);


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_vector_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize);

template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_vector_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize);

}

#include "csr-vector.cu"

#endif