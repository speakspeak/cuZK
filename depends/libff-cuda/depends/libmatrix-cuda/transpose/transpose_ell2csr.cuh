#ifndef __TRANSPOSE_ELL2CSR_CUH__
#define __TRANSPOSE_ELL2CSR_CUH__

#include "../sparse-matrix/csr.cuh"
#include "../sparse-matrix/ell.cuh"
#include "../depends/libstl-cuda/memory.cuh"

namespace libmatrix
{

template<typename T>
__device__ CSR_matrix<T>* p_transpose_ell2csr(const ELL_matrix<T>& mtx, size_t gridSize, size_t blockSize);


template<typename T>
__device__ CSR_matrix_opt<T>* p_transpose_ell2csr(const ELL_matrix_opt<T>& mtx, size_t gridSize, size_t blockSize);

template<typename T>
__host__ CSR_matrix_opt<T>* p_transpose_ell2csr_host(ELL_matrix_opt<T>& mtx, size_t gridSize, size_t blockSize);



}

#include "transpose_ell2csr.cu"

#endif