#ifndef __CSR_SCALAR_CU__
#define __CSR_SCALAR_CU__

namespace libmatrix
{

template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_scalar(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &v, &zero]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < mtx.row_size)
            {
                for(size_t i=mtx.row_ptr[idx]; i < mtx.row_ptr[idx + 1]; i++)
                {
                    (*res)[idx] = (*res)[idx] + v[mtx.col_idx[i]] * mtx.data[i];
                }
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return res;
}


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_scalar(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_scalar<T, T>(mtx, v, zero, gridSize, blockSize);
}



template<typename T, typename S>
__host__ libstl::vector<T>* p_spmv_csr_scalar_host(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize)
{

    libstl::vector<T>* res = libstl::create_host<libstl::vector<T>>();
    size_t mtx_row_size;
    cudaMemcpy(&mtx_row_size, &mtx.row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    res->presize_host(mtx_row_size, &zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero)
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < mtx.row_size)
            {
                for(size_t i=mtx.row_ptr[idx]; i < mtx.row_ptr[idx + 1]; i++)
                {
                    (*res)[idx] = (*res)[idx] + v[mtx.col_idx[i]] * mtx.data[i];
                }
                idx += gridDim.x * blockDim.x;
            } 
        }, mtx, v, zero
    );
    cudaStreamSynchronize(0);

    return res;
}


template<typename T>
__host__ libstl::vector<T>* p_spmv_csr_scalar_host(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_scalar_host<T, T>(mtx, v, zero, gridSize, blockSize);
}



template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < mtx.row_size)
            {
                for(size_t i=mtx.row_ptr[idx]; i < mtx.row_ptr[idx + 1]; i++)
                {
                    (*res)[idx] = (*res)[idx] + mtx.data[i];
                }
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();


    return res;
}


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_scalar_vector_one<T, T>(mtx, zero, gridSize, blockSize);
}


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &vec]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < mtx.row_size)
            {
                for(size_t i=mtx.row_ptr[idx]; i < mtx.row_ptr[idx + 1]; i++)
                {
                    (*res)[idx] = (*res)[idx].mixed_add(vec[mtx.col_data[i].data_addr]);
                }
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();


    return res;
}


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_scalar_vector_one(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_scalar_vector_one<T, T>(mtx, vec, zero, gridSize, blockSize);
}



template<typename T, typename S>
__host__ libstl::vector<T>* p_spmv_csr_scalar_vector_one_host(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create_host<libstl::vector<T>>();
    size_t mtx_row_size;
    cudaMemcpy(&mtx_row_size, &mtx.row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    res->presize_host(mtx_row_size, &zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec)
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < mtx.row_size)
            {
                for(size_t i=mtx.row_ptr[idx]; i < mtx.row_ptr[idx + 1]; i++)
                {
                    (*res)[idx] = (*res)[idx].mixed_add(vec[mtx.col_data[i].data_addr]);
                }
                idx += gridDim.x * blockDim.x;
            }
        }, mtx, vec
    );
    cudaStreamSynchronize(0);


    return res;
}


template<typename T>
__host__ libstl::vector<T>* p_spmv_csr_scalar_vector_one_host(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_scalar_vector_one_host<T, T>(mtx, vec, zero, gridSize, blockSize);
}




}

#endif