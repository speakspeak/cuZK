#ifndef __ELL_CUH__
#define __ELL_CUH__

#include "../depends/libstl-cuda/vector.cuh"

namespace libmatrix{

template<typename T>
class ELL_matrix
{
public:
    libstl::vector<size_t> row_length;
    libstl::vector<size_t> col_idx;
    libstl::vector<T> data;

    size_t max_row_length;
    size_t row_size;
    size_t col_size;

public:
    __device__ bool init(size_t max_row_length, size_t row_size, size_t col_size);
    __device__ bool p_init(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize);
    __host__ bool p_init_host(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize);
    __device__ bool p_reset(size_t gridSize, size_t blockSize);
    __device__ bool insert(const T& data, size_t row, size_t col);
};



template<typename T>
class ELL_matrix_opt
{
public:
    libstl::vector<size_t> row_length;
    libstl::vector<size_t> col_idx;
    libstl::vector<T*> data_addr;

    size_t max_row_length;
    size_t row_size;
    size_t col_size;
    size_t total;

public:
    __device__ bool init(size_t max_row_length, size_t row_size, size_t col_size);
    __device__ bool p_init(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize);
    __host__ bool p_init_host(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize);
    __device__ bool p_reset(size_t gridSize, size_t blockSize);
    __host__ bool p_reset_host(size_t gridSize, size_t blockSize);
    __device__ __inline__ bool insert(size_t row, size_t col);
};


}

#include "ell.cu"


#endif