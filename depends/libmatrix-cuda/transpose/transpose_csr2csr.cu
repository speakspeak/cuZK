#ifndef __TRANSPOSE_CSR2CSR_CU__
#define __TRANSPOSE_CSR2CSR_CU__


namespace libmatrix
{

template<typename T>
__global__ void p_transpose_csr2csr_serial(CSR_matrix<T>& res, size_t gridSize, size_t blockSize)
{
    size_t t_num = gridSize * blockSize;
    size_t stride = (res.row_ptr.size() + t_num - 1) / t_num;

    size_t current_idx = stride * 2 - 1;
    while(current_idx < res.row_ptr.size())
    {
        res.row_ptr[current_idx] += res.row_ptr[current_idx - stride];
        current_idx += stride; 
    }
}


template<typename T>
__device__ CSR_matrix<T>* p_transpose_csr2csr(const CSR_matrix<T>& mtx, size_t gridSize, size_t blockSize)
{
    CSR_matrix<T>* pres = libstl::create<CSR_matrix<T>>(CSR_matrix<T>());
    CSR_matrix<T>& res = *pres;

    res.col_size = mtx.row_size;
    res.row_size = mtx.col_size;

    size_t* pizero = libstl::create<size_t>(0);
    res.row_ptr.presize(res.row_size + 1, *pizero, gridSize, blockSize);
    res.col_idx.presize(mtx.col_idx.size(), gridSize, blockSize);
    res.data.presize(mtx.data.size(), gridSize, blockSize);

    libstl::vector<size_t>* p_offset = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_offset->presize(mtx.col_idx.size(), gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &res]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            while(tid < mtx.col_idx.size())
            {
                (*p_offset)[tid] = atomicAdd((unsigned long long*)&res.row_ptr[mtx.col_idx[tid] + 1], (unsigned long long)1);
                tid += t_num;
            }
        }
    );
    cudaDeviceSynchronize();


    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &res]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t range_s = (res.row_ptr.size() + t_num - 1) / t_num * tid;
            size_t range_e = (res.row_ptr.size() + t_num - 1) / t_num * (tid + 1);
            for(size_t i=range_s+1; i<range_e && i<res.row_ptr.size(); i++)
            {
                res.row_ptr[i] += res.row_ptr[i-1];
            }
        }
    );
    cudaDeviceSynchronize();

    p_transpose_csr2csr_serial<<<1, 1>>>(res, gridSize, blockSize);
    cudaDeviceSynchronize();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &res]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t range_s = (res.row_ptr.size() + t_num - 1) / t_num * (tid + 1);
            size_t range_e = (res.row_ptr.size() + t_num - 1) / t_num * (tid + 2) - 1;
            for(size_t i=range_s; i<range_e && i<res.row_ptr.size(); i++)
            {
                res.row_ptr[i] += res.row_ptr[range_s - 1];
            }
        }
    );
    cudaDeviceSynchronize();

    // transpose
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &res]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t range_s = (mtx.data.size() + t_num - 1) / t_num * tid;
            size_t range_e = (mtx.data.size() + t_num - 1) / t_num * (tid + 1);
            size_t col = 0;
            for(size_t i=range_s; i<range_e && i<mtx.data.size(); i++)
            {
                size_t ptr = res.row_ptr[mtx.col_idx[i]];
                size_t off = (*p_offset)[i];
                while(i >= mtx.row_ptr[col + 1])
                {
                    col += 1;
                }

                res.col_idx[ptr + off] = col;
                res.data[ptr + off] = mtx.data[i];
            }
        }
    );
    cudaDeviceSynchronize();

    return pres;
}


}



#endif