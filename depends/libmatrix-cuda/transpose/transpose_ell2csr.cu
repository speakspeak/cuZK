#ifndef __TRANSPOSE_ELL2CSR_CU__
#define __TRANSPOSE_ELL2CSR_CU__

#include "../common/utils.cuh"
#include <cub/cub.cuh>

namespace libmatrix
{

template<typename T>
__device__ size_t p_transpose_ell2csr_total_length(const ELL_matrix<T>& mtx, size_t gridSize, size_t blockSize)
{
    size_t total_length = 0;
    for(size_t i=0; i<mtx.row_length.size(); i++) total_length += mtx.row_length[i];

    return total_length;
}

template<typename T>
__global__ void p_transpose_ell2csr_serial(CSR_matrix<T>& res, size_t gridSize, size_t blockSize)
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
__device__ CSR_matrix<T>* p_transpose_ell2csr(const ELL_matrix<T>& mtx, size_t gridSize, size_t blockSize)
{
    CSR_matrix<T>* pres = libstl::create<CSR_matrix<T>>(CSR_matrix<T>());
    CSR_matrix<T>& res = *pres;

    res.col_size = mtx.row_size;
    res.row_size = mtx.col_size;

    size_t* pizero = libstl::create<size_t>(0);
    res.row_ptr.presize(mtx.col_size + 1, *pizero, gridSize, blockSize);
    size_t total_length = p_transpose_ell2csr_total_length(mtx, gridSize, blockSize);

    res.col_idx.presize(total_length, gridSize, blockSize);
    res.data.presize(total_length, gridSize, blockSize);

    libstl::vector<size_t>* p_offset = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_offset->presize(mtx.col_idx.size(), gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &res]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            while(tid < mtx.row_size)
            {
                size_t ptr = tid * mtx.max_row_length;
                for(size_t i=0; i<mtx.row_length[tid]; i++)
                {
                    (*p_offset)[ptr + i] = atomicAdd((unsigned long long*)&res.row_ptr[mtx.col_idx[ptr + i] + 1], (unsigned long long)1);
                }
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


    p_transpose_ell2csr_serial<<<1, 1>>>(res, gridSize, blockSize);
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
            while(tid < mtx.row_size)
            {
                for(size_t i=0; i<mtx.row_length[tid]; i++)
                {
                    size_t idx = tid * mtx.max_row_length + i;
                    size_t ptr = res.row_ptr[mtx.col_idx[idx]];
                    size_t off = (*p_offset)[idx];
                    size_t col = tid;

                    res.col_idx[ptr + off] = col;
                    res.data[ptr + off] = mtx.data[idx];
                }
                tid += t_num;
            }
        }
    );
    cudaDeviceSynchronize();

    return pres;
}


template<typename T>
__device__ size_t p_transpose_ell2csr_total_length(const ELL_matrix_opt<T>& mtx, size_t gridSize, size_t blockSize)
{
    size_t* izero = libstl::create<size_t>(0);
    libstl::vector<size_t>* mid_length = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    mid_length->presize(gridSize * blockSize, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t total = mtx.row_length.size();
            size_t range_s = (total + t_num - 1) / t_num * tid;
            size_t range_e = (total + t_num - 1) / t_num * (tid + 1);
            for(size_t i=range_s; i < range_e && i < total; i++) (*mid_length)[tid] += mtx.row_length[i];
        }
    );
    cudaDeviceSynchronize();

    size_t* p_t_count = libstl::create<size_t>(gridSize * blockSize);
    size_t* p_count = libstl::create<size_t>(1);
    size_t& t_count = *p_t_count;
    size_t& count = *p_count;
    while(t_count != 1)
    {
        if(gridSize / (2 * count) != 1)
        {
            libstl::launch<<<gridSize / (2 * count), blockSize>>>
            (
                [=, &t_count, &count]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    size_t idx = 2 * count * tid;
                    size_t total_size = 2 * count * gridDim.x * blockDim.x;
                    if(idx + count < total_size)
                    {
                        (*mid_length)[idx] = (*mid_length)[idx] + (*mid_length)[idx + count];
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
        else
        {
            libstl::launch<<<1, blockSize>>>
            (
                [=, &t_count, &count]
                __device__ ()
                {
                    size_t tid = threadIdx.x;
                    size_t idx = 2 * count * tid;
                    size_t total_size = 2 * count * blockDim.x;
                    size_t inner_count = 1;
                    
                    while(t_count != 1)
                    {
                        if(tid % inner_count  == 0)
                        {
                            if(idx + count < total_size)
                            {
                                (*mid_length)[idx] = (*mid_length)[idx] + (*mid_length)[idx + count];
                            }
                        }
                        if(tid == 0)
                        {
                            t_count = (t_count + 1) / 2;
                            count *= 2;
                        }
                        __syncthreads();

                        inner_count *= 2;
                    }
                }
            );
            cudaDeviceSynchronize();
        }
    }

    return (*mid_length)[0];
}


template<typename T>
__global__ void p_transpose_ell2csr_serial(CSR_matrix_opt<T>& res, size_t gridSize, size_t blockSize)
{
    size_t t_num = gridSize * blockSize;
    size_t stride = (res.row_ptr.size() + t_num - 1) / t_num;

    for(size_t i=0; i < log2(t_num); i++)
    {
        libstl::launch<<<gridSize / 2, blockSize>>>
        (
            [=, &res]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t idx = tid / (1<<i) * (1<<(i + 1)) + (1 << i) + tid % (1 << i);
                size_t widx = (idx + 1) * stride - 1;
                size_t ridx = (idx / (1<<i) * (1<<i)) * stride - 1;
                res.row_ptr[widx] = res.row_ptr[widx] + res.row_ptr[ridx];
            }
        );
        cudaDeviceSynchronize();
    }

}

template<typename T>
__device__ CSR_matrix_opt<T>* p_transpose_ell2csr(const ELL_matrix_opt<T>& mtx, size_t gridSize, size_t blockSize)
{
    CSR_matrix_opt<T>* pres = libstl::create<CSR_matrix_opt<T>>(CSR_matrix_opt<T>());
    CSR_matrix_opt<T>& res = *pres;

    res.col_size = mtx.row_size;
    res.row_size = mtx.col_size;

    size_t* pizero = libstl::create<size_t>(0);
    res.row_ptr.presize(mtx.col_size + 1, *pizero, gridSize, blockSize);
    size_t total_length = mtx.total;  // p_transpose_ell2csr_total_length(mtx, gridSize, blockSize);

    res.col_data.presize(total_length, gridSize, blockSize);

    libstl::vector<size_t>* p_offset = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_offset->presize(mtx.col_idx.size(), gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &res]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            while(tid < mtx.row_size)
            {
                size_t ptr = tid * mtx.max_row_length;
                for(size_t i=0; i<mtx.row_length[tid]; i++)
                {
                    (*p_offset)[ptr + i] = atomicAdd((unsigned long long*)&res.row_ptr[mtx.col_idx[ptr + i] + 1], (unsigned long long)1);
                }
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

    p_transpose_ell2csr_serial<<<1, 1>>>(res, gridSize, blockSize);
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
            while(tid < mtx.row_size)
            {
                for(size_t i=0; i<mtx.row_length[tid]; i++)
                {
                    size_t ridx = tid * mtx.max_row_length + i;
                    size_t widx = (*p_offset)[ridx] + res.row_ptr[mtx.col_idx[ridx]];
                    // res.col_idx[widx] = tid;
                    // res.data_addr[widx] = mtx.data_addr[ridx];

                    res.col_data[widx] = {tid, ridx};
                }
                tid += t_num;
            }
        }
    );
    cudaDeviceSynchronize();

    return pres;
}


template<typename T>
__host__ CSR_matrix_opt<T>* p_transpose_ell2csr_host(ELL_matrix_opt<T>& mtx, size_t gridSize, size_t blockSize)
{
    CSR_matrix_opt<T>* pres = libstl::create_host<CSR_matrix_opt<T>>();
    size_t* pizero = libstl::create_host<size_t>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (ELL_matrix_opt<T>& mtx)
        {
            pres->col_size = mtx.row_size;
            pres->row_size = mtx.col_size;
            *pizero = 0;
        }, mtx
    );
    cudaStreamSynchronize(0);

    size_t row_size;
    size_t col_size;
    size_t total_length;
    
    cudaMemcpy(&row_size, &mtx.row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&col_size, &mtx.col_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_length, &mtx.total, sizeof(size_t), cudaMemcpyDeviceToHost);

    pres->row_ptr.presize_host(col_size + 1, pizero, gridSize, blockSize);
    pres->col_data.presize_host(total_length, gridSize, blockSize);

    size_t lockMem;
    libstl::lock_host(lockMem);

    libstl::vector<size_t>* p_offset = libstl::create_host<libstl::vector<size_t>>();
    p_offset->presize_host(mtx.col_idx.size_host(), gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (ELL_matrix_opt<T>& mtx)
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            while(tid < mtx.row_size)
            {
                size_t ptr = tid * mtx.max_row_length;
                for(size_t i=0; i<mtx.row_length[tid]; i++)
                {
                    (*p_offset)[ptr + i] = atomicAdd((unsigned long long*)&pres->row_ptr[mtx.col_idx[ptr + i] + 1], (unsigned long long)1);
                }
                tid += t_num;
            }
        }, mtx
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t range_s = (pres->row_ptr.size() + t_num - 1) / t_num * tid;
            size_t range_e = (pres->row_ptr.size() + t_num - 1) / t_num * (tid + 1);
            for(size_t i=range_s+1; i<range_e && i<pres->row_ptr.size(); i++)
            {
                pres->row_ptr[i] += pres->row_ptr[i-1];
            }
        }
    );
    cudaStreamSynchronize(0);

    size_t t_num = gridSize * blockSize;
    size_t stride = (col_size + 1 + t_num - 1) / t_num;

    for(size_t i=0; i < log2(t_num); i++)
    {
        libstl::launch<<<gridSize / 2, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t idx = tid / (1<<i) * (1<<(i + 1)) + (1 << i) + tid % (1 << i);
                size_t widx = (idx + 1) * stride - 1;
                size_t ridx = (idx / (1<<i) * (1<<i)) * stride - 1;
                pres->row_ptr[widx] = pres->row_ptr[widx] + pres->row_ptr[ridx];
            }
        );
        cudaStreamSynchronize(0);
    }

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t range_s = (pres->row_ptr.size() + t_num - 1) / t_num * (tid + 1);
            size_t range_e = (pres->row_ptr.size() + t_num - 1) / t_num * (tid + 2) - 1;
            for(size_t i=range_s; i<range_e && i<pres->row_ptr.size(); i++)
            {
                pres->row_ptr[i] += pres->row_ptr[range_s - 1];
            }
        }
    );
    cudaStreamSynchronize(0);

    // transpose
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (ELL_matrix_opt<T>& mtx)
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            while(tid < mtx.row_size)
            {
                for(size_t i=0; i<mtx.row_length[tid]; i++)
                {
                    size_t ridx = tid * mtx.max_row_length + i;
                    size_t widx = (*p_offset)[ridx] + pres->row_ptr[mtx.col_idx[ridx]];

                    pres->col_data[widx] = {tid, ridx};
                }
                tid += t_num;
            }
        }, mtx
    );
    cudaStreamSynchronize(0);

    libstl::resetlock_host(lockMem);

    return pres;
}


template<typename T>
__host__ CSR_matrix_opt<T>* p_transpose_ell2csr_cub_host(ELL_matrix_opt<T>& mtx, size_t gridSize, size_t blockSize)
{
    CSR_matrix_opt<T>* pres = libstl::create_host<CSR_matrix_opt<T>>();
    size_t* pizero = libstl::create_host<size_t>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (ELL_matrix_opt<T>& mtx)
        {
            pres->col_size = mtx.row_size;
            pres->row_size = mtx.col_size;
            *pizero = 0;
        }, mtx
    );
    cudaStreamSynchronize(0);

    size_t row_size;
    size_t col_size;
    size_t total_length;
    
    cudaMemcpy(&row_size, &mtx.row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&col_size, &mtx.col_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_length, &mtx.total, sizeof(size_t), cudaMemcpyDeviceToHost);

    pres->data_addr.presize_host(total_length, gridSize, blockSize);
    pres->row_ptr.presize_host(col_size + 1, pizero, gridSize, blockSize);

    size_t lockMem;
    libstl::lock_host(lockMem);

    libstl::vector<size_t>* p_key_out = libstl::create_host<libstl::vector<size_t>>();
    p_key_out->presize_host(total_length, gridSize, blockSize);
    libstl::sort_pair_host(&mtx.col_idx, &mtx.data_addr, p_key_out, &pres->data_addr, total_length, 0, 32);

    libstl::vector<size_t>* p_unique = libstl::create_host<libstl::vector<size_t>>();
    libstl::vector<size_t>* p_count = libstl::create_host<libstl::vector<size_t>>();
    libstl::vector<size_t>* p_unique_count = libstl::create_host<libstl::vector<size_t>>();
    p_unique->presize_host(col_size + 1, gridSize, blockSize);
    p_count->presize_host(col_size + 1, gridSize, blockSize);
    p_unique_count->presize_host(1, gridSize, blockSize);
    libstl::run_length_host(p_key_out, p_unique, p_count, p_unique_count, total_length);

    libstl::vector<size_t>* p_row_count = libstl::create_host<libstl::vector<size_t>>();
    p_row_count->presize_host(col_size + 1, pizero, gridSize, blockSize);
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            size_t t_num = blockDim.x * gridDim.x;
            size_t range_s = ((*p_unique_count)[0] + t_num - 1) / t_num * tid;
            size_t range_e = ((*p_unique_count)[0] + t_num - 1) / t_num * (tid + 1);
            for(size_t i=range_s+1; i<range_e && i<(*p_unique_count)[0]; i++)
            {
                (*p_row_count)[(*p_unique)[i] + 1] = (*p_count)[i];
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::inclusive_sum(p_row_count, &pres->row_ptr, col_size + 1);
    libstl::resetlock_host(lockMem);

    return pres;
}


}

#endif