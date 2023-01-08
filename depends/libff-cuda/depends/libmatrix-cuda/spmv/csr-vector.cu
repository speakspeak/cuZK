#ifndef __CSR_VECTOR_CU__
#define __CSR_VECTOR_CU__

namespace libmatrix
{


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_vector(const CSR_matrix<T>& mtx, const libstl::vector<S>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    // size_t ideal_count = mtx.row_ptr[mtx.row_size] / (gridSize * blockSize);
    // printf("ideal count: %d\n", ideal_count);
    // libstl::vector<size_t>* vcount = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>(gridSize * blockSize, 0));

    libstl::launch<<<gridSize, blockSize, blockSize * sizeof(T)>>>
    (
        [=, &mtx, &v, &zero]
        __device__ ()
        {
            extern __shared__ T sdata[];

            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

            size_t bid = blockIdx.x;
            size_t bnum = gridDim.x;
            for(size_t row = bid; row < mtx.row_size; row += bnum)
            {
                size_t row_s = mtx.row_ptr[row];
                size_t row_e = mtx.row_ptr[row+1];
                size_t t_lane = threadIdx.x;
                size_t t_num = blockDim.x;
                for(size_t i = row_s + t_lane; i < row_e; i += t_num)
                {
                    // (*vcount)[tid] += 1;
                    sdata[threadIdx.x] = sdata[threadIdx.x] + v[mtx.col_idx[i]] * mtx.data[i];
                }
                
                // reduce local sums to row sum
                size_t count = 1;
                while(t_num != 1)
                {
                    if(t_lane % (count * 2) == 0)
                    {
                        if(t_lane + count < blockDim.x) 
                        {
                            sdata[t_lane] = sdata[t_lane] + sdata[t_lane + count];
                        }
                    }
                    t_num = (t_num + 1) / 2;
                    count *= 2;
                }
                if(threadIdx.x == 0)
                {
                    (*res)[row] = sdata[0];
                }
            }
        }
    );
    cudaDeviceSynchronize();

    return res;
}



template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_vector(const CSR_matrix<T>& mtx, const libstl::vector<T>& v, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_vector<T, T>(mtx, v, zero, gridSize, blockSize);
}


template<typename T, typename S>
__device__ libstl::vector<T>* p_spmv_csr_vector_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    res->presize(mtx.row_size, zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize, blockSize * sizeof(T)>>>
    (
        [=, &mtx]
        __device__ ()
        {
            extern __shared__ T sdata[];

            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

            size_t bid = blockIdx.x;
            size_t bnum = gridDim.x;
            for(size_t row = bid; row < mtx.row_size; row += bnum)
            {
                size_t row_s = mtx.row_ptr[row];
                size_t row_e = mtx.row_ptr[row+1];
                size_t t_lane = threadIdx.x;
                size_t t_num = blockDim.x;
                for(size_t i = row_s + t_lane; i < row_e; i += t_num)
                {
                    sdata[threadIdx.x] = sdata[threadIdx.x] + mtx.data[i];
                }
                
                // reduce local sums to row sum
                size_t count = 1;
                while(t_num != 1)
                {
                    if(t_lane % (count * 2) == 0)
                    {
                        if(t_lane + count < blockDim.x) 
                        {
                            sdata[t_lane] = sdata[t_lane] + sdata[t_lane + count];
                        }
                    }
                    t_num = (t_num + 1) / 2;
                    count *= 2;
                }
                if(threadIdx.x == 0)
                {
                    (*res)[row] = sdata[0];
                }
            }
        }
    );
    cudaDeviceSynchronize();

    return res;
}

template<typename T>
__device__ libstl::vector<T>* p_spmv_csr_vector_vector_one(const CSR_matrix<T>& mtx, const T& zero, size_t gridSize, size_t blockSize)
{
    return p_spmv_csr_vector_vector_one<T, T>(mtx, zero, gridSize, blockSize);
}



}

#endif