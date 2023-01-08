#ifndef __ELL_CU__
#define __ELL_CU__

namespace libmatrix{

template<typename T>
__device__ bool ELL_matrix<T>::init(size_t max_row_length, size_t row_size, size_t col_size)
{
    this->max_row_length = max_row_length;
    this->row_size = row_size;
    this->col_size = col_size;
    this->row_length.resize(row_size, 0);
    this->col_idx.resize(row_size * max_row_length);
    this->data.resize(row_size * max_row_length);
    return true;
}

template<typename T>
__device__ bool ELL_matrix<T>::p_init(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize)
{
    this->max_row_length = max_row_length;
    this->row_size = row_size;
    this->col_size = col_size;
    size_t* izero = libstl::create<size_t>(0);
    this->row_length.presize(row_size, *izero, gridSize, blockSize);
    this->col_idx.presize(row_size * max_row_length, gridSize, blockSize);
    this->data.presize(row_size * max_row_length, gridSize, blockSize);
    return true;
}

template<typename T>
__host__ bool ELL_matrix<T>::p_init_host(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize)
{
    cudaMemcpy(&this->max_row_length, &max_row_length, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&this->row_size, &row_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&this->col_size, &col_size, sizeof(size_t), cudaMemcpyHostToDevice);

    size_t* izero = libstl::create_host<size_t>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *izero = 0;
        }
    );
    cudaDeviceSynchronize();

    this->row_length.presize_host(row_size, izero, gridSize, blockSize);
    this->col_idx.presize_host(row_size * max_row_length, gridSize, blockSize);
    this->data.presize_host(row_size * max_row_length, gridSize, blockSize);
    return true;
}

template<typename T>
__device__ bool ELL_matrix<T>::p_reset(size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = gridDim.x * blockDim.x;
            while(tid < row_length.size())
            {
                this->row_length[tid] = 0;
                tid += tnum;
            }
        }
    );
    cudaDeviceSynchronize();
    return true;
}


template<typename T>
__device__ bool ELL_matrix<T>::insert(const T& data, size_t row, size_t col)
{
    size_t row_ptr = row * this->max_row_length;
    size_t idx = 0;
    bool find = false;
    for(size_t i = row_ptr; i < row_ptr + this->row_length[row]; i++)
    {
        if(this->col_idx[i] == col)
        {
            find = true;
            idx = i;
        }
    }

    if(find == false)
    {
        idx = row_ptr + this->row_length[row];
        this->row_length[row] += 1;
        this->data[idx] = data;
    }
    else
        this->data[idx] = this->data[idx] + data;

    this->col_idx[idx] = col;
    
    return true;
}


template<typename T>
__device__ bool ELL_matrix_opt<T>::init(size_t max_row_length, size_t row_size, size_t col_size)
{
    this->max_row_length = max_row_length;
    this->row_size = row_size;
    this->col_size = col_size;
    this->row_length.resize(row_size, 0);
    this->col_idx.resize(row_size * max_row_length);
    // this->data_addr.resize(row_size * max_row_length);
    return true;
}


template<typename T>
__device__ bool ELL_matrix_opt<T>::p_init(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize)
{
    this->max_row_length = max_row_length;
    this->row_size = row_size;
    this->col_size = col_size;
    size_t* izero = libstl::create<size_t>(0);
    this->row_length.presize(row_size, *izero, gridSize, blockSize);
    this->col_idx.presize(row_size * max_row_length, gridSize, blockSize);
    // this->data_addr.presize(row_size * max_row_length, gridSize, blockSize);
    return true;
}

template<typename T>
__host__ bool ELL_matrix_opt<T>::p_init_host(size_t max_row_length, size_t row_size, size_t col_size, size_t gridSize, size_t blockSize)
{
    cudaMemcpy(&this->max_row_length, &max_row_length, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&this->row_size, &row_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&this->col_size, &col_size, sizeof(size_t), cudaMemcpyHostToDevice);

    size_t* izero = libstl::create_host<size_t>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *izero = 0;
        }
    );
    cudaStreamSynchronize(0);


    this->row_length.presize_host(row_size, izero, gridSize, blockSize);
    this->col_idx.presize_host(row_size * max_row_length, gridSize, blockSize);
    // this->data.presize_host(row_size * max_row_length, gridSize, blockSize);
    return true;
}


template<typename T>
__device__ bool ELL_matrix_opt<T>::p_reset(size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = gridDim.x * blockDim.x;
            while(tid < row_length.size())
            {
                this->row_length[tid] = 0;
                tid += tnum;
            }
        }
    );
    cudaDeviceSynchronize();
    return true;
}

template<typename T>
__host__ bool ELL_matrix_opt<T>::p_reset_host(size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = gridDim.x * blockDim.x;
            while(tid < row_length.size())
            {
                this->row_length[tid] = 0;
                tid += tnum;
            }
        }
    );
    cudaStreamSynchronize(0);
    
    return true;
}


template<typename T>
__device__ __inline__ bool ELL_matrix_opt<T>::insert(size_t row, size_t col)
{
    size_t row_ptr = row * this->max_row_length;
    size_t idx = row_ptr + this->row_length[row];
    this->row_length[row] += 1;
    // this->data_addr[idx] = &data;
    this->col_idx[idx] = col;

    return true;
}

}

#endif
