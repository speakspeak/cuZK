#ifndef __TRANSPOSE_CSR2CSR_CUH__
#define __TRANSPOSE_CSR2CSR_CUH__


#include "../sparse-matrix/csr.cuh"
#include "../depends/libstl-cuda/memory.cuh"

namespace libmatrix
{

template<typename T>
__device__ CSR_matrix<T>* p_transpose_csr2csr(const CSR_matrix<T>& mtx, size_t gridSize, size_t blockSize);

}

#include "transpose_csr2csr.cu"


#endif