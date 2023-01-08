#ifndef __CSR_BALANCED_CU__
#define __CSR_BALANCED_CU__

#include "../common/utils.cuh"

namespace libmatrix
{

// __device__ size_t log2(size_t n)
// {
//     size_t r = ((n & (n-1)) == 0 ? 0 : 1); // add 1 if n is not power of 2

//     while (n > 1)
//     {
//         n >>= 1;
//         r++;
//     }

//     return r;
// }

// static __device__ size_t lsqrt(size_t n)
// {
//     size_t i = 0;
//     while(i * i <= n) i++;
//     i--;
//     return i;
// }


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_balanced(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    size_t x_total = mtx.row_ptr[mtx.row_size];
    size_t N = mtx.row_size;
    size_t m = (x_total + N - 1) / N;
    size_t t = gridSize * blockSize;
    size_t B = blockSize;
    size_t G = gridSize;

    size_t z = B * lsqrt((2 * m * log2(B) + B - 1) / B);

    // n
    libstl::vector<size_t>* pn = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    size_t* izero = libstl::create<size_t>(0);
    pn->presize(N, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < N)
            {
                (*pn)[idx] = (mtx.row_ptr[idx + 1] - mtx.row_ptr[idx]) / z;
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    // zero max num
    libstl::vector<size_t>* p_n_zero_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_max_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_other_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_n_zero_num->presize(t, *izero, gridSize, blockSize);
    p_n_max_num->presize(t, *izero, gridSize, blockSize);
    p_n_other_num->presize(t, *izero, gridSize, blockSize);

    size_t *p_n_zero_total = libstl::create<size_t>(0);
    size_t *p_n_max_total = libstl::create<size_t>(0);
    size_t *p_n_other_total = libstl::create<size_t>(0);
    size_t& n_zero_total = *p_n_zero_total;
    size_t& n_max_total = *p_n_max_total;
    size_t& n_other_total = *p_n_other_total;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            size_t s = (pn->size() + tnum - 1) / tnum * tid;
            size_t e = (pn->size() + tnum - 1) / tnum * (tid + 1);
            for(size_t i=s; i < e && i < pn->size(); i++)
            {
                if((*pn)[i] == 0) (*p_n_zero_num)[tid]++;
                if((*pn)[i] >= G) (*p_n_max_num)[tid]++;
                if((*pn)[i] > 0 && (*pn)[i] < G) (*p_n_other_num)[tid]++;
            }
        }
    );
    cudaDeviceSynchronize();

    size_t tnum = blockSize * gridSize;

    for(size_t i=0; i < log2(tnum); i++)
    {
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                if(tid / (1<<i) % 2 == 1)
                {
                    (*p_n_zero_num)[tid] = (*p_n_zero_num)[tid] + (*p_n_zero_num)[tid / (1<<i) * (1<<i) - 1];
                    (*p_n_max_num)[tid] = (*p_n_max_num)[tid] + (*p_n_max_num)[tid / (1<<i) * (1<<i) - 1];
                    (*p_n_other_num)[tid] = (*p_n_other_num)[tid] + (*p_n_other_num)[tid / (1<<i) * (1<<i) - 1];
                }
            }
        );
        cudaDeviceSynchronize();
    }
    n_zero_total = (*p_n_zero_num)[tnum - 1];
    n_max_total = (*p_n_max_num)[tnum - 1];
    n_other_total = (*p_n_other_num)[tnum - 1];

    // zero max idx
    libstl::vector<size_t>* p_n_zero_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_max_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_other_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_n_zero_idx->presize(n_zero_total, *izero, gridSize, blockSize);
    p_n_max_idx->presize(n_max_total, *izero, gridSize, blockSize);
    p_n_other_idx->presize(n_other_total, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            size_t s = (pn->size() + tnum - 1) / tnum * tid;
            size_t e = (pn->size() + tnum - 1) / tnum * (tid + 1);

            size_t self_zero_ptr = 0;
            size_t self_max_ptr = 0;
            size_t self_other_ptr = 0;
            if(tid != 0)
            {
                self_zero_ptr = (*p_n_zero_num)[tid - 1];
                self_max_ptr = (*p_n_max_num)[tid - 1];
                self_other_ptr = (*p_n_other_num)[tid - 1];
            }

            for(size_t i=s; i < e && i < pn->size(); i++)
            {
                if((*pn)[i] == 0) (*p_n_zero_idx)[self_zero_ptr++] = i;
                if((*pn)[i] >= G) (*p_n_max_idx)[self_max_ptr++] = i;
                if((*pn)[i] > 0 && (*pn)[i] < G) (*p_n_other_idx)[self_other_ptr++] = i;
            }
        }
    );
    cudaDeviceSynchronize();

    // zero
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &v]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(tid < n_zero_total)
            {
                size_t idx = (*p_n_zero_idx)[tid];
                size_t s = mtx.row_ptr[idx];
                size_t e = mtx.row_ptr[idx + 1];
                for(size_t i=s; i < e; i++)
                {
                    (*res)[idx] = (*res)[idx] + v[mtx.col_idx[i]] * mtx.data[i];
                }
                tid += tnum;
            }
        }
    );
    cudaDeviceSynchronize();

    // max
    for(size_t i=0; i<n_max_total; i++)
    {
        libstl::vector<T>* p_max_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
        p_max_mid_res->presize(t, zero, gridSize, blockSize);
        size_t idx = (*p_n_max_idx)[i];

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=, &mtx, &v]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = blockDim.x * gridDim.x;
                size_t row_s = mtx.row_ptr[idx];
                size_t row_e = mtx.row_ptr[idx + 1];
                size_t t_idx = row_s + tid;
                while(t_idx < row_e)
                {
                    (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + v[mtx.col_idx[t_idx]] * mtx.data[t_idx];
                    t_idx += tnum;
                }
                
            }
        );
        cudaDeviceSynchronize();

        // reduction  to be opt
        // for(size_t i=0; i<tnum; i++)
        // {
        //     (*res)[idx] = (*res)[idx] + (*p_max_mid_res)[i];
        // }

        // reduction
        size_t t_count = tnum;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x )
                        {
                            (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + (*p_max_mid_res)[tid + count];
                        }
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
        (*res)[idx] = (*p_max_mid_res)[0];
    }


    size_t n_other_total_size = 0;
    for(size_t i=0; i<n_other_total; i++)
    {
        n_other_total_size += (*pn)[(*p_n_other_idx)[i]];
    }

    // other
    size_t s_row_idx = 0;
    size_t s_col_idx = 0;
    for(size_t i = 0; i < n_other_total_size; i += G)
    {
        libstl::vector<size_t>* p_block_row_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        libstl::vector<size_t>* p_block_col_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        p_block_row_idx->presize(G, gridSize, blockSize);
        p_block_col_idx->presize(G, gridSize, blockSize);
        libstl::vector<size_t>* p_block_change_row_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        p_block_change_row_idx->presize(G, gridSize, blockSize);
        size_t change_row_length = 0;
        (*p_block_change_row_idx)[0] = 0;
        change_row_length++;

        for(size_t j = 0; j < G && s_row_idx < n_other_total; j++)
        {
            if(s_col_idx < (*pn)[(*p_n_other_idx)[s_row_idx]])
            {
                (*p_block_row_idx)[j] = s_row_idx;
                (*p_block_col_idx)[j] = s_col_idx;
                s_col_idx++;
            }
            else
            {
                s_col_idx = 0;
                s_row_idx++;
                (*p_block_change_row_idx)[change_row_length] = j;
                change_row_length++;
                if(s_row_idx >= n_other_total) break;

                (*p_block_row_idx)[j] = s_row_idx;
                (*p_block_col_idx)[j] = s_col_idx;
                s_col_idx++;
            }
        }

        if(s_row_idx < n_other_total)
        {
            (*p_block_change_row_idx)[change_row_length] = G;
            change_row_length++;
        }

        libstl::vector<T>* p_other_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
        p_other_mid_res->presize(t, zero, gridSize, blockSize);

        size_t launch_gridSize = (*p_block_change_row_idx)[change_row_length-1];

        libstl::launch<<<launch_gridSize, blockSize>>>
        (
            [=, &mtx, &v]
            __device__ ()
            {
                size_t bid = blockIdx.x;
                size_t row_idx = (*p_block_row_idx)[bid];
                size_t col_idx = (*p_block_col_idx)[bid];

                size_t row_s = mtx.row_ptr[(*p_n_other_idx)[row_idx]];
                size_t row_e = mtx.row_ptr[(*p_n_other_idx)[row_idx] + 1];
                size_t row_bnum = (*pn)[(*p_n_other_idx)[row_idx]];
                size_t col_s = ((row_e - row_s) + row_bnum - 1) / row_bnum * col_idx;
                size_t col_e = ((row_e - row_s) + row_bnum - 1) / row_bnum * (col_idx + 1);

                size_t s = row_s + col_s + threadIdx.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

                while(s < row_s + col_e && s < row_e)
                {
                    (*p_other_mid_res)[tid] = (*p_other_mid_res)[tid] + v[mtx.col_idx[s]] * mtx.data[s];
                    s += blockDim.x;
                }
            }
        );
        cudaDeviceSynchronize();

        // for(size_t j = 0; j < change_row_length - 1; j++)
        // {
        //     size_t change_row_s = (*p_block_change_row_idx)[j];
        //     size_t change_row_e = (*p_block_change_row_idx)[j + 1];

        //     size_t res_idx = (*p_n_other_idx)[(*p_block_row_idx)[change_row_s]];
        //     for(size_t k = change_row_s; k < change_row_e; k++)
        //     {
        //         for(size_t kk = 0; kk < B; kk++)
        //             (*res)[res_idx] = (*res)[res_idx] + (*p_other_mid_res)[k * B + kk];
        //     }
        // }

        size_t t_count = tnum;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<launch_gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x && (*p_block_row_idx)[tid / B] == (*p_block_row_idx)[(tid + count) / B])
                        {
                            (*p_other_mid_res)[tid] = (*p_other_mid_res)[tid] + (*p_other_mid_res)[tid + count];
                        }
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }

        for(size_t j = 0; j < change_row_length - 1; j++)
        {
            size_t change_row_s = (*p_block_change_row_idx)[j];
            size_t change_row_e = (*p_block_change_row_idx)[j + 1];
            size_t count = 1;

            while(change_row_s < change_row_e)
            {
                size_t res_idx = (*p_n_other_idx)[(*p_block_row_idx)[change_row_s]];
                (*res)[res_idx] = (*res)[res_idx] + (*p_other_mid_res)[change_row_s * B];

                while(change_row_s % count == 0)
                {
                    count *= 2;
                    if((change_row_s + count / 2) >= change_row_e) break;
                }
                count /= 2;

                change_row_s += count;
            }
        }
    }

    return res;
}


template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_balanced(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_balanced<T, T>(mtx, v, zero, gridSize, blockSize);
}


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_balanced_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    size_t x_total = mtx.row_ptr[mtx.row_size];
    size_t N = mtx.row_size;
    size_t m = (x_total + N - 1) / N;
    size_t t = gridSize * blockSize;
    size_t B = blockSize;
    size_t G = gridSize;

    size_t z = B * lsqrt((2 * m * log2(B) + B - 1) / B);

    // n
    libstl::vector<size_t>* pn = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    size_t* izero = libstl::create<size_t>(0);
    pn->presize(N, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < N)
            {
                (*pn)[idx] = (mtx.row_ptr[idx + 1] - mtx.row_ptr[idx]) / z;
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    // zero max num
    libstl::vector<size_t>* p_n_zero_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_max_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_other_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_n_zero_num->presize(t, *izero, gridSize, blockSize);
    p_n_max_num->presize(t, *izero, gridSize, blockSize);
    p_n_other_num->presize(t, *izero, gridSize, blockSize);

    size_t *p_n_zero_total = libstl::create<size_t>(0);
    size_t *p_n_max_total = libstl::create<size_t>(0);
    size_t *p_n_other_total = libstl::create<size_t>(0);
    size_t& n_zero_total = *p_n_zero_total;
    size_t& n_max_total = *p_n_max_total;
    size_t& n_other_total = *p_n_other_total;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            size_t s = (pn->size() + tnum - 1) / tnum * tid;
            size_t e = (pn->size() + tnum - 1) / tnum * (tid + 1);
            for(size_t i=s; i < e && i < pn->size(); i++)
            {
                if((*pn)[i] == 0) (*p_n_zero_num)[tid]++;
                if((*pn)[i] >= G) (*p_n_max_num)[tid]++;
                if((*pn)[i] > 0 && (*pn)[i] < G) (*p_n_other_num)[tid]++;
            }
        }
    );
    cudaDeviceSynchronize();

    size_t tnum = blockSize * gridSize;

    for(size_t i=0; i < log2(tnum); i++)
    {
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                if(tid / (1<<i) % 2 == 1)
                {
                    (*p_n_zero_num)[tid] = (*p_n_zero_num)[tid] + (*p_n_zero_num)[tid / (1<<i) * (1<<i) - 1];
                    (*p_n_max_num)[tid] = (*p_n_max_num)[tid] + (*p_n_max_num)[tid / (1<<i) * (1<<i) - 1];
                    (*p_n_other_num)[tid] = (*p_n_other_num)[tid] + (*p_n_other_num)[tid / (1<<i) * (1<<i) - 1];
                }
            }
        );
        cudaDeviceSynchronize();
    }
    n_zero_total = (*p_n_zero_num)[tnum - 1];
    n_max_total = (*p_n_max_num)[tnum - 1];
    n_other_total = (*p_n_other_num)[tnum - 1];

    // zero max idx
    libstl::vector<size_t>* p_n_zero_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_max_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_other_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_n_zero_idx->presize(n_zero_total, *izero, gridSize, blockSize);
    p_n_max_idx->presize(n_max_total, *izero, gridSize, blockSize);
    p_n_other_idx->presize(n_other_total, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            size_t s = (pn->size() + tnum - 1) / tnum * tid;
            size_t e = (pn->size() + tnum - 1) / tnum * (tid + 1);

            size_t self_zero_ptr = 0;
            size_t self_max_ptr = 0;
            size_t self_other_ptr = 0;
            if(tid != 0)
            {
                self_zero_ptr = (*p_n_zero_num)[tid - 1];
                self_max_ptr = (*p_n_max_num)[tid - 1];
                self_other_ptr = (*p_n_other_num)[tid - 1];
            }

            for(size_t i=s; i < e && i < pn->size(); i++)
            {
                if((*pn)[i] == 0) (*p_n_zero_idx)[self_zero_ptr++] = i;
                if((*pn)[i] >= G) (*p_n_max_idx)[self_max_ptr++] = i;
                if((*pn)[i] > 0 && (*pn)[i] < G) (*p_n_other_idx)[self_other_ptr++] = i;
            }
        }
    );
    cudaDeviceSynchronize();

    // zero
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(tid < n_zero_total)
            {
                size_t idx = (*p_n_zero_idx)[tid];
                size_t s = mtx.row_ptr[idx];
                size_t e = mtx.row_ptr[idx + 1];
                for(size_t i=s; i < e; i++)
                {
                    (*res)[idx] = (*res)[idx] + mtx.data[i];
                }
                tid += tnum;
            }
        }
    );
    cudaDeviceSynchronize();

    // max
    for(size_t i=0; i<n_max_total; i++)
    {
        libstl::vector<T>* p_max_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
        p_max_mid_res->presize(t, zero, gridSize, blockSize);
        size_t idx = (*p_n_max_idx)[i];

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=, &mtx]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = blockDim.x * gridDim.x;
                size_t row_s = mtx.row_ptr[idx];
                size_t row_e = mtx.row_ptr[idx + 1];
                size_t t_idx = row_s + tid;
                while(t_idx < row_e)
                {
                    (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + mtx.data[t_idx];
                    t_idx += tnum;
                }
                
            }
        );
        cudaDeviceSynchronize();

        // reduction  to be opt
        // for(size_t i=0; i<tnum; i++)
        // {
        //     (*res)[idx] = (*res)[idx] + (*p_max_mid_res)[i];
        // }

        // reduction
        size_t t_count = tnum;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x )
                        {
                            (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + (*p_max_mid_res)[tid + count];
                        }
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
        (*res)[idx] = (*p_max_mid_res)[0];
    }


    size_t n_other_total_size = 0;
    for(size_t i=0; i<n_other_total; i++)
    {
        n_other_total_size += (*pn)[(*p_n_other_idx)[i]];
    }

    // other
    size_t s_row_idx = 0;
    size_t s_col_idx = 0;
    for(size_t i = 0; i < n_other_total_size; i += G)
    {
        libstl::vector<size_t>* p_block_row_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        libstl::vector<size_t>* p_block_col_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        p_block_row_idx->presize(G, gridSize, blockSize);
        p_block_col_idx->presize(G, gridSize, blockSize);
        libstl::vector<size_t>* p_block_change_row_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        p_block_change_row_idx->presize(G, gridSize, blockSize);
        size_t change_row_length = 0;
        (*p_block_change_row_idx)[0] = 0;
        change_row_length++;

        for(size_t j = 0; j < G && s_row_idx < n_other_total; j++)
        {
            if(s_col_idx < (*pn)[(*p_n_other_idx)[s_row_idx]])
            {
                (*p_block_row_idx)[j] = s_row_idx;
                (*p_block_col_idx)[j] = s_col_idx;
                s_col_idx++;
            }
            else
            {
                s_col_idx = 0;
                s_row_idx++;
                (*p_block_change_row_idx)[change_row_length] = j;
                change_row_length++;
                if(s_row_idx >= n_other_total) break;

                (*p_block_row_idx)[j] = s_row_idx;
                (*p_block_col_idx)[j] = s_col_idx;
                s_col_idx++;
            }
        }

        if(s_row_idx < n_other_total)
        {
            (*p_block_change_row_idx)[change_row_length] = G;
            change_row_length++;
        }

        libstl::vector<T>* p_other_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
        p_other_mid_res->presize(t, zero, gridSize, blockSize);

        size_t launch_gridSize = (*p_block_change_row_idx)[change_row_length-1];

        libstl::launch<<<launch_gridSize, blockSize>>>
        (
            [=, &mtx]
            __device__ ()
            {
                size_t bid = blockIdx.x;
                size_t row_idx = (*p_block_row_idx)[bid];
                size_t col_idx = (*p_block_col_idx)[bid];

                size_t row_s = mtx.row_ptr[(*p_n_other_idx)[row_idx]];
                size_t row_e = mtx.row_ptr[(*p_n_other_idx)[row_idx] + 1];
                size_t row_bnum = (*pn)[(*p_n_other_idx)[row_idx]];
                size_t col_s = ((row_e - row_s) + row_bnum - 1) / row_bnum * col_idx;
                size_t col_e = ((row_e - row_s) + row_bnum - 1) / row_bnum * (col_idx + 1);

                size_t s = row_s + col_s + threadIdx.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

                while(s < row_s + col_e && s < row_e)
                {
                    (*p_other_mid_res)[tid] = (*p_other_mid_res)[tid] + mtx.data[s];
                    s += blockDim.x;
                }
            }
        );
        cudaDeviceSynchronize();

        // for(size_t j = 0; j < change_row_length - 1; j++)
        // {
        //     size_t change_row_s = (*p_block_change_row_idx)[j];
        //     size_t change_row_e = (*p_block_change_row_idx)[j + 1];

        //     size_t res_idx = (*p_n_other_idx)[(*p_block_row_idx)[change_row_s]];
        //     for(size_t k = change_row_s; k < change_row_e; k++)
        //     {
        //         for(size_t kk = 0; kk < B; kk++)
        //             (*res)[res_idx] = (*res)[res_idx] + (*p_other_mid_res)[k * B + kk];
        //     }
        // }

        size_t t_count = tnum;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<launch_gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x && (*p_block_row_idx)[tid / B] == (*p_block_row_idx)[(tid + count) / B])
                        {
                            (*p_other_mid_res)[tid] = (*p_other_mid_res)[tid] + (*p_other_mid_res)[tid + count];
                        }
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }

        for(size_t j = 0; j < change_row_length - 1; j++)
        {
            size_t change_row_s = (*p_block_change_row_idx)[j];
            size_t change_row_e = (*p_block_change_row_idx)[j + 1];
            size_t count = 1;

            while(change_row_s < change_row_e)
            {
                size_t res_idx = (*p_n_other_idx)[(*p_block_row_idx)[change_row_s]];
                (*res)[res_idx] = (*res)[res_idx] + (*p_other_mid_res)[change_row_s * B];

                while(change_row_s % count == 0)
                {
                    count *= 2;
                    if((change_row_s + count / 2) >= change_row_e) break;
                }
                count /= 2;

                change_row_s += count;
            }
        }
    }

    return res;
}

template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_balanced_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_balanced_vector_one<T, T>(mtx, zero, gridSize, blockSize);
}


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_balanced_vector_one(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    size_t x_total = mtx.row_ptr[mtx.row_size];
    size_t N = mtx.row_size;
    size_t m = (x_total + N - 1) / N;
    size_t t = gridSize * blockSize;
    size_t B = blockSize;
    size_t G = gridSize;

    size_t z = B * lsqrt((2 * m * log2(B) + B - 1) / B);

    // n
    libstl::vector<size_t>* pn = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    size_t* izero = libstl::create<size_t>(0);
    pn->presize(N, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < N)
            {
                (*pn)[idx] = (mtx.row_ptr[idx + 1] - mtx.row_ptr[idx]) / z;
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    // zero max num
    libstl::vector<size_t>* p_n_zero_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_max_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_other_num = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_n_zero_num->presize(t, *izero, gridSize, blockSize);
    p_n_max_num->presize(t, *izero, gridSize, blockSize);
    p_n_other_num->presize(t, *izero, gridSize, blockSize);

    size_t *p_n_zero_total = libstl::create<size_t>(0);
    size_t *p_n_max_total = libstl::create<size_t>(0);
    size_t *p_n_other_total = libstl::create<size_t>(0);
    size_t& n_zero_total = *p_n_zero_total;
    size_t& n_max_total = *p_n_max_total;
    size_t& n_other_total = *p_n_other_total;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            size_t s = (pn->size() + tnum - 1) / tnum * tid;
            size_t e = (pn->size() + tnum - 1) / tnum * (tid + 1);
            for(size_t i=s; i < e && i < pn->size(); i++)
            {
                if((*pn)[i] == 0) (*p_n_zero_num)[tid]++;
                if((*pn)[i] >= G) (*p_n_max_num)[tid]++;
                if((*pn)[i] > 0 && (*pn)[i] < G) (*p_n_other_num)[tid]++;
            }
        }
    );
    cudaDeviceSynchronize();

    size_t tnum = blockSize * gridSize;

    for(size_t i=0; i < log2(tnum); i++)
    {
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                if(tid / (1<<i) % 2 == 1)
                {
                    (*p_n_zero_num)[tid] = (*p_n_zero_num)[tid] + (*p_n_zero_num)[tid / (1<<i) * (1<<i) - 1];
                    (*p_n_max_num)[tid] = (*p_n_max_num)[tid] + (*p_n_max_num)[tid / (1<<i) * (1<<i) - 1];
                    (*p_n_other_num)[tid] = (*p_n_other_num)[tid] + (*p_n_other_num)[tid / (1<<i) * (1<<i) - 1];
                }
            }
        );
        cudaDeviceSynchronize();
    }
    n_zero_total = (*p_n_zero_num)[tnum - 1];
    n_max_total = (*p_n_max_num)[tnum - 1];
    n_other_total = (*p_n_other_num)[tnum - 1];

    // zero max idx
    libstl::vector<size_t>* p_n_zero_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_max_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    libstl::vector<size_t>* p_n_other_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
    p_n_zero_idx->presize(n_zero_total, *izero, gridSize, blockSize);
    p_n_max_idx->presize(n_max_total, *izero, gridSize, blockSize);
    p_n_other_idx->presize(n_other_total, *izero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            size_t s = (pn->size() + tnum - 1) / tnum * tid;
            size_t e = (pn->size() + tnum - 1) / tnum * (tid + 1);

            size_t self_zero_ptr = 0;
            size_t self_max_ptr = 0;
            size_t self_other_ptr = 0;
            if(tid != 0)
            {
                self_zero_ptr = (*p_n_zero_num)[tid - 1];
                self_max_ptr = (*p_n_max_num)[tid - 1];
                self_other_ptr = (*p_n_other_num)[tid - 1];
            }

            for(size_t i=s; i < e && i < pn->size(); i++)
            {
                if((*pn)[i] == 0) (*p_n_zero_idx)[self_zero_ptr++] = i;
                if((*pn)[i] >= G) (*p_n_max_idx)[self_max_ptr++] = i;
                if((*pn)[i] > 0 && (*pn)[i] < G) (*p_n_other_idx)[self_other_ptr++] = i;
            }
        }
    );
    cudaDeviceSynchronize();

    // zero
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &mtx, &vec]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(tid < n_zero_total)
            {
                size_t idx = (*p_n_zero_idx)[tid];
                size_t s = mtx.row_ptr[idx];
                size_t e = mtx.row_ptr[idx + 1];
                for(size_t i=s; i < e; i++)
                {
                    (*res)[idx] = (*res)[idx].mixed_add(vec[mtx.col_data[i].data_addr]);
                   
                }
                tid += tnum;
            }
        }
    );
    cudaDeviceSynchronize();

    // max
    for(size_t i=0; i<n_max_total; i++)
    {
        libstl::vector<T>* p_max_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
        p_max_mid_res->presize(t, zero, gridSize, blockSize);
        size_t idx = (*p_n_max_idx)[i];

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=, &mtx, &vec]
            __device__ ()
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = blockDim.x * gridDim.x;
                size_t row_s = mtx.row_ptr[idx];
                size_t row_e = mtx.row_ptr[idx + 1];
                size_t t_idx = row_s + tid;
                while(t_idx < row_e)
                {
                    (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid].mixed_add(vec[mtx.col_data[t_idx].data_addr]);
                    t_idx += tnum;
                }
                
            }
        );
        cudaDeviceSynchronize();

        // reduction  to be opt
        // for(size_t i=0; i<tnum; i++)
        // {
        //     (*res)[idx] = (*res)[idx] + (*p_max_mid_res)[i];
        // }

        // reduction
        size_t t_count = tnum;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x )
                        {
                            (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + (*p_max_mid_res)[tid + count];
                        }
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
        (*res)[idx] = (*p_max_mid_res)[0];
    }


    size_t n_other_total_size = 0;
    for(size_t i=0; i<n_other_total; i++)
    {
        n_other_total_size += (*pn)[(*p_n_other_idx)[i]];
    }

    // other
    size_t s_row_idx = 0;
    size_t s_col_idx = 0;
    for(size_t i = 0; i < n_other_total_size; i += G)
    {
        libstl::vector<size_t>* p_block_row_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        libstl::vector<size_t>* p_block_col_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        p_block_row_idx->presize(G, gridSize, blockSize);
        p_block_col_idx->presize(G, gridSize, blockSize);
        libstl::vector<size_t>* p_block_change_row_idx = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>());
        p_block_change_row_idx->presize(G, gridSize, blockSize);
        size_t change_row_length = 0;
        (*p_block_change_row_idx)[0] = 0;
        change_row_length++;

        for(size_t j = 0; j < G && s_row_idx < n_other_total; j++)
        {
            if(s_col_idx < (*pn)[(*p_n_other_idx)[s_row_idx]])
            {
                (*p_block_row_idx)[j] = s_row_idx;
                (*p_block_col_idx)[j] = s_col_idx;
                s_col_idx++;
            }
            else
            {
                s_col_idx = 0;
                s_row_idx++;
                (*p_block_change_row_idx)[change_row_length] = j;
                change_row_length++;
                if(s_row_idx >= n_other_total) break;

                (*p_block_row_idx)[j] = s_row_idx;
                (*p_block_col_idx)[j] = s_col_idx;
                s_col_idx++;
            }
        }

        if(s_row_idx < n_other_total)
        {
            (*p_block_change_row_idx)[change_row_length] = G;
            change_row_length++;
        }

        libstl::vector<T>* p_other_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
        p_other_mid_res->presize(t, zero, gridSize, blockSize);

        size_t launch_gridSize = (*p_block_change_row_idx)[change_row_length-1];

        libstl::launch<<<launch_gridSize, blockSize>>>
        (
            [=, &mtx, &vec]
            __device__ ()
            {
                size_t bid = blockIdx.x;
                size_t row_idx = (*p_block_row_idx)[bid];
                size_t col_idx = (*p_block_col_idx)[bid];

                size_t row_s = mtx.row_ptr[(*p_n_other_idx)[row_idx]];
                size_t row_e = mtx.row_ptr[(*p_n_other_idx)[row_idx] + 1];
                size_t row_bnum = (*pn)[(*p_n_other_idx)[row_idx]];
                size_t col_s = ((row_e - row_s) + row_bnum - 1) / row_bnum * col_idx;
                size_t col_e = ((row_e - row_s) + row_bnum - 1) / row_bnum * (col_idx + 1);

                size_t s = row_s + col_s + threadIdx.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

                while(s < row_s + col_e && s < row_e)
                {
                    (*p_other_mid_res)[tid] = (*p_other_mid_res)[tid].mixed_add(vec[mtx.col_data[s].data_addr]);
                    s += blockDim.x;
                }
            }
        );
        cudaDeviceSynchronize();

        // for(size_t j = 0; j < change_row_length - 1; j++)
        // {
        //     size_t change_row_s = (*p_block_change_row_idx)[j];
        //     size_t change_row_e = (*p_block_change_row_idx)[j + 1];

        //     size_t res_idx = (*p_n_other_idx)[(*p_block_row_idx)[change_row_s]];
        //     for(size_t k = change_row_s; k < change_row_e; k++)
        //     {
        //         for(size_t kk = 0; kk < B; kk++)
        //             (*res)[res_idx] = (*res)[res_idx] + (*p_other_mid_res)[k * B + kk];
        //     }
        // }

        size_t t_count = tnum;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<launch_gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x && (*p_block_row_idx)[tid / B] == (*p_block_row_idx)[(tid + count) / B])
                        {
                            (*p_other_mid_res)[tid] = (*p_other_mid_res)[tid] + (*p_other_mid_res)[tid + count];
                        }
                    }
                }
            );
            cudaDeviceSynchronize();
            t_count = (t_count + 1) / 2;
            count *= 2;
        }

        for(size_t j = 0; j < change_row_length - 1; j++)
        {
            size_t change_row_s = (*p_block_change_row_idx)[j];
            size_t change_row_e = (*p_block_change_row_idx)[j + 1];
            size_t count = 1;

            while(change_row_s < change_row_e)
            {
                size_t res_idx = (*p_n_other_idx)[(*p_block_row_idx)[change_row_s]];
                (*res)[res_idx] = (*res)[res_idx] + (*p_other_mid_res)[change_row_s * B];

                while(change_row_s % count == 0)
                {
                    count *= 2;
                    if((change_row_s + count / 2) >= change_row_e) break;
                }
                count /= 2;

                change_row_s += count;
            }
        }
    }

    return res;
}



template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_balanced_vector_one(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_balanced_vector_one<T, T>(mtx, vec, zero, gridSize, blockSize);
}



template<typename T, typename S>
__host__ libstl::vector<T>* p_spmv_csr_balanced_vector_one_host(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, T& zero, size_t gridSize, size_t blockSize)
{
    cudaStream_t streamT;
    cudaStreamCreateWithFlags(&streamT, cudaStreamNonBlocking);

    libstl::vector<T>* res = libstl::create_host<libstl::vector<T>>();
    size_t mtx_row_size;
    cudaMemcpy(&mtx_row_size, &mtx.row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    res->presize_host(mtx_row_size, &zero, gridSize, blockSize);

    size_t t = gridSize * blockSize;
    size_t B = blockSize;
    size_t G = gridSize;
    size_t z = B * 5 * 2;

    size_t* hrow_ptr = new size_t[mtx_row_size + 1];
    void* mtx_row_addr;
    cudaMemcpy(&mtx_row_addr, &mtx.row_ptr._data, sizeof(void *), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(hrow_ptr, (void *)mtx_row_addr, (mtx_row_size + 1) * sizeof(size_t), cudaMemcpyDeviceToHost, streamT);

    // libstl::vector<size_t> hrow_ptr;
    // libstl::vector_device2host(&hrow_ptr, &mtx.row_ptr, streams[G]);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec)
        {
            size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(idx < mtx.row_size)
            {
                size_t s = mtx.row_ptr[idx];
                size_t e = mtx.row_ptr[idx + 1];
                if(e - s < z)
                {
                    for(size_t i=s; i < e; i++)
                    {
                        (*res)[idx] = (*res)[idx].mixed_add(vec[mtx.col_data[i].data_addr]);
                    }
                }
                idx += tnum;
            }
        }, mtx, vec
    );
    cudaStreamSynchronize(0);

    cudaStreamSynchronize(streamT);

    // max
    size_t Gz = G * z;
    for(size_t i=0; i<mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if(e - s < Gz) continue;

        libstl::vector<T>* p_max_mid_res = libstl::create_host<libstl::vector<T>>();
        p_max_mid_res->presize_host(t, &zero, gridSize, blockSize);
        
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec)
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t idx = s + tid;
                size_t tnum = blockDim.x * gridDim.x;
                while(idx < e)
                {
                    (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid].mixed_add(vec[mtx.col_data[idx].data_addr]);
                    idx += tnum;
                }
            }, mtx, vec
        );
        cudaStreamSynchronize(0);

        // reduction
        size_t t_count = t;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x )
                        {
                            (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + (*p_max_mid_res)[tid + count];
                        }
                    }
                    if(tid == 0) (*res)[i] = (*p_max_mid_res)[0];
                }
            );
            cudaStreamSynchronize(0);
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
    }

    size_t n_other_total = 0;
    for(size_t i=0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if((e - s > Gz) || (e - s < z)) continue;
        n_other_total += 1;
    }
    if(n_other_total == 0)
    {
        // for (int i = 0; i < gridSize + 1; i++) cudaStreamDestroy(streams[i]);
        delete[] hrow_ptr;
        cudaStreamDestroy(streamT);
        return res;
    }

    cudaStream_t streams[G];
    for (int i = 0; i < G; i++) {
        cudaStreamCreate(&streams[i]);
    }
    

    // libstl::vector<size_t> row_ptr;
    // row_ptr.resize_host(n_other_total + 1);
    size_t* row_ptr = new size_t[n_other_total + 1];
    row_ptr[0] = 0;

    size_t n_other_total_idx = 0;
    for(size_t i=0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if((e - s >= Gz) || (e - s < z)) continue;

        size_t n = (e - s) / z;
        row_ptr[n_other_total_idx + 1] = row_ptr[n_other_total_idx] + n;

        n_other_total_idx += 1;
    }

    libstl::vector<T>* c_mid_data = libstl::create_host<libstl::vector<T>>();
    libstl::vector<size_t>* c_mid_row = libstl::create_host<libstl::vector<size_t>>();
    libstl::vector<size_t>* c_mid_idx = libstl::create_host<libstl::vector<size_t>>();
    c_mid_data->presize_host(row_ptr[n_other_total], gridSize, blockSize);
    c_mid_row->presize_host(n_other_total + 1, gridSize, blockSize);
    c_mid_idx->presize_host(n_other_total, gridSize, blockSize);

    void* c_mid_row_addr;
    cudaMemcpy(&c_mid_row_addr, &c_mid_row->_data, sizeof(void *), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync((void *)c_mid_row_addr, row_ptr, (n_other_total + 1) * sizeof(size_t), cudaMemcpyHostToDevice, streamT);

    // vector_host2device(c_mid_row, &row_ptr, streams[gridSize]);

    size_t stream_id = 0;
    n_other_total_idx = 0;
    for(size_t i=0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if((e - s > G * z) || (e - s < z)) continue;
        
        size_t stream_G = (e - s) / z;
        size_t ptr = row_ptr[n_other_total_idx];
        libstl::launch_with_shared<<<stream_G, blockSize, blockSize * sizeof(T), streams[stream_id]>>>
        (
            [=]
            __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, T& zero, unsigned char* smem)
            {
                // extern __shared__ T s_mid_res[];
                T* s_mid_res = (T*)smem;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t idx = s + tid;
                size_t bid = threadIdx.x;
                size_t gid = blockIdx.x;
                size_t tnum = blockDim.x * gridDim.x;
                s_mid_res[bid] = zero;

                while(idx < e)
                {    
                    s_mid_res[bid] = s_mid_res[bid].mixed_add(vec[mtx.col_data[idx].data_addr]);
                    idx += tnum;
                }
                __syncthreads();

                // reduce block
                size_t b_count = blockDim.x;
                size_t count = 1;
                while(b_count != 1)
                {
                    if(bid % (count * 2) == 0)
                    {
                        if(bid + count < blockDim.x )
                        {
                            s_mid_res[bid] = s_mid_res[bid] + s_mid_res[bid + count];
                        }
                    }
                    __syncthreads();
                    b_count = (b_count + 1) / 2;
                    count *= 2;
                }
                if(bid == 0)
                {
                    (*c_mid_data)[ptr + gid] = s_mid_res[0];
                }
                if(tid == 0)
                {
                    (*c_mid_idx)[n_other_total_idx] = i;
                }
            }, mtx, vec, zero
        );
        stream_id = (stream_id + 1) % G;
        n_other_total_idx += 1;
    }

    for(size_t i=0; i < G; i++) cudaStreamSynchronize(streams[i]);
    cudaStreamSynchronize(streamT);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(idx < n_other_total)
            {
                size_t s = (*c_mid_row)[idx];
                size_t e = (*c_mid_row)[idx + 1];
                size_t addr = (*c_mid_idx)[idx];
                for(size_t i = s; i < e; i++)
                {
                    (*res)[addr] = (*res)[addr] + (*c_mid_data)[i];
                }
                idx += tnum;
            }
        }
    );
    cudaStreamSynchronize(0);

    for (int i = 0; i < G; i++) cudaStreamDestroy(streams[i]);
    cudaStreamDestroy(streamT);

    delete[] hrow_ptr;
    delete[] row_ptr;
    return res;
}


template<typename T>
__host__ libstl::vector<T>* p_spmv_csr_balanced_vector_one_host(const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_balanced_vector_one_host<T, T>(mtx, vec, zero, gridSize, blockSize);
}


template<typename T, typename S>
__host__ void p_spmv_csr_balanced_vector_one_host(libstl::vector<T>* res, const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, T& zero, size_t gridSize, size_t blockSize)
{
    cudaStream_t streamT;
    cudaStreamCreateWithFlags(&streamT, cudaStreamNonBlocking);

    // libstl::vector<T>* res = libstl::create_host<libstl::vector<T>>();
    size_t mtx_row_size;
    cudaMemcpy(&mtx_row_size, &mtx.row_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    // res->presize_host(mtx_row_size, &zero, gridSize, blockSize);

    size_t t = gridSize * blockSize;
    size_t B = blockSize;
    size_t G = gridSize;
    size_t z = B * 5 * 2;

    size_t* hrow_ptr = new size_t[mtx_row_size + 1];
    void* mtx_row_addr;
    cudaMemcpy(&mtx_row_addr, &mtx.row_ptr._data, sizeof(void *), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(hrow_ptr, (void *)mtx_row_addr, (mtx_row_size + 1) * sizeof(size_t), cudaMemcpyDeviceToHost, streamT);

    // libstl::vector<size_t> hrow_ptr;
    // libstl::vector_device2host(&hrow_ptr, &mtx.row_ptr, streams[G]);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec)
        {
            size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(idx < mtx.row_size)
            {
                size_t s = mtx.row_ptr[idx];
                size_t e = mtx.row_ptr[idx + 1];
                if(e - s < z)
                {
                    for(size_t i=s; i < e; i++)
                    {
                        (*res)[idx] = (*res)[idx].mixed_add(vec[mtx.col_data[i].data_addr]);
                    }
                }
                idx += tnum;
            }
        }, mtx, vec
    );
    cudaStreamSynchronize(0);

    cudaStreamSynchronize(streamT);

    // max
    size_t Gz = G * z;
    for(size_t i=0; i<mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if(e - s < Gz) continue;

        libstl::vector<T>* p_max_mid_res = libstl::create_host<libstl::vector<T>>();
        p_max_mid_res->presize_host(t, &zero, gridSize, blockSize);
        
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec)
            {
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t idx = s + tid;
                size_t tnum = blockDim.x * gridDim.x;
                while(idx < e)
                {
                    (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid].mixed_add(vec[mtx.col_data[idx].data_addr]);
                    idx += tnum;
                }
            }, mtx, vec
        );
        cudaStreamSynchronize(0);

        // reduction
        size_t t_count = t;
        size_t count = 1;
        while(t_count != 1)
        {
            libstl::launch<<<gridSize, blockSize>>>
            (
                [=]
                __device__ ()
                {
                    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid % (count * 2) == 0)
                    {
                        if(tid + count < gridDim.x * blockDim.x )
                        {
                            (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + (*p_max_mid_res)[tid + count];
                        }
                    }
                    if(tid == 0) (*res)[i] = (*p_max_mid_res)[0];
                }
            );
            cudaStreamSynchronize(0);
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
    }

    size_t n_other_total = 0;
    for(size_t i=0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if((e - s >= Gz) || (e - s < z)) continue;
        n_other_total += 1;
    }
    if(n_other_total == 0)
    {
        // for (int i = 0; i < gridSize + 1; i++) cudaStreamDestroy(streams[i]);
        delete[] hrow_ptr;
        cudaStreamDestroy(streamT);
        return ;
    }

    cudaStream_t streams[G];
    for (int i = 0; i < G; i++) {
        cudaStreamCreate(&streams[i]);
    }
    

    // libstl::vector<size_t> row_ptr;
    // row_ptr.resize_host(n_other_total + 1);
    size_t* row_ptr = new size_t[n_other_total + 1];
    row_ptr[0] = 0;

    size_t n_other_total_idx = 0;
    for(size_t i=0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if((e - s >= Gz) || (e - s < z)) continue;

        size_t n = (e - s) / z;
        row_ptr[n_other_total_idx + 1] = row_ptr[n_other_total_idx] + n;

        n_other_total_idx += 1;
    }

    libstl::vector<T>* c_mid_data = libstl::create_host<libstl::vector<T>>();
    libstl::vector<size_t>* c_mid_row = libstl::create_host<libstl::vector<size_t>>();
    libstl::vector<size_t>* c_mid_idx = libstl::create_host<libstl::vector<size_t>>();
    c_mid_data->presize_host(row_ptr[n_other_total], gridSize, blockSize);
    c_mid_row->presize_host(n_other_total + 1, gridSize, blockSize);
    c_mid_idx->presize_host(n_other_total, gridSize, blockSize);

    void* c_mid_row_addr;
    cudaMemcpy(&c_mid_row_addr, &c_mid_row->_data, sizeof(void *), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync((void *)c_mid_row_addr, row_ptr, (n_other_total + 1) * sizeof(size_t), cudaMemcpyHostToDevice, streamT);

    // vector_host2device(c_mid_row, &row_ptr, streams[gridSize]);

    size_t stream_id = 0;
    n_other_total_idx = 0;
    for(size_t i=0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if((e - s > G * z) || (e - s < z)) continue;
        
        size_t stream_G = (e - s) / z;
        size_t ptr = row_ptr[n_other_total_idx];
        libstl::launch_with_shared<<<stream_G, blockSize, blockSize * sizeof(T), streams[stream_id]>>>
        (
            [=]
            __device__ (const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, T& zero, unsigned char* smem)
            {
                // extern __shared__ T s_mid_res[];
                T* s_mid_res = (T*)smem;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t idx = s + tid;
                size_t bid = threadIdx.x;
                size_t gid = blockIdx.x;
                size_t tnum = blockDim.x * gridDim.x;
                s_mid_res[bid] = zero;

                while(idx < e)
                {    
                    s_mid_res[bid] = s_mid_res[bid].mixed_add(vec[mtx.col_data[idx].data_addr]);
                    idx += tnum;
                }
                __syncthreads();

                // reduce block
                size_t b_count = blockDim.x;
                size_t count = 1;
                while(b_count != 1)
                {
                    if(bid % (count * 2) == 0)
                    {
                        if(bid + count < blockDim.x )
                        {
                            s_mid_res[bid] = s_mid_res[bid] + s_mid_res[bid + count];
                        }
                    }
                    __syncthreads();
                    b_count = (b_count + 1) / 2;
                    count *= 2;
                }
                if(bid == 0)
                {
                    (*c_mid_data)[ptr + gid] = s_mid_res[0];
                }
                if(tid == 0)
                {
                    (*c_mid_idx)[n_other_total_idx] = i;
                }
            }, mtx, vec, zero
        );
        stream_id = (stream_id + 1) % G;
        n_other_total_idx += 1;
    }

    for(size_t i=0; i < G; i++) cudaStreamSynchronize(streams[i]);
    cudaStreamSynchronize(streamT);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            while(idx < n_other_total)
            {
                size_t s = (*c_mid_row)[idx];
                size_t e = (*c_mid_row)[idx + 1];
                size_t addr = (*c_mid_idx)[idx];
                for(size_t i = s; i < e; i++)
                {
                    (*res)[addr] = (*res)[addr] + (*c_mid_data)[i];
                }
                idx += tnum;
            }
        }
    );
    cudaStreamSynchronize(0);

    for (int i = 0; i < G; i++) cudaStreamDestroy(streams[i]);
    cudaStreamDestroy(streamT);

    delete[] hrow_ptr;
    delete[] row_ptr;
    return ;
}


template<typename T>
__host__ void p_spmv_csr_balanced_vector_one_host(libstl::vector<T>* res, const CSR_matrix_opt<T>& mtx, const libstl::vector<T>& vec, T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_balanced_vector_one_host<T, T>(res, mtx, vec, zero, gridSize, blockSize);
}


}

#endif