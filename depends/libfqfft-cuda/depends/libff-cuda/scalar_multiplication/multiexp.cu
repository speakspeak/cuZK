#ifndef __MULTIEXP_CU__
#define __MULTIEXP_CU__

#include "../fields/bigint.cuh"
#include "../common/utils.cuh"
#include "wnaf.cu"

namespace libff{

template<typename T, typename FieldT>
__device__ T multi_exp_inner_naive(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance)
{
    T result(t_instance.zero());
    // assert(vec.size() == scalar.size());

    for (int i=0; i < vec.size(); i++)
    {
        result = result + opt_window_wnaf_exp(vec[i], scalar[i].as_bigint(), scalar[i].as_bigint().num_bits());
    }

    return result;
}

template<typename T, typename FieldT>
__device__ T multi_exp_inner_naive_plain(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance)
{
    T result(t_instance.zero());
    // assert(vec.size() == scalar.size());

    for (int i=0; i < vec.size(); i++)
    {
        result = result + scalar[i] * vec[i];
    }

    return result;
}


template<typename T, typename FieldT>
__device__ T multi_exp_inner_naive_plain(typename libstl::vector<T>::const_iterator vec_begin, 
                                         typename libstl::vector<T>::const_iterator vec_end, 
                                         typename libstl::vector<FieldT>::const_iterator scalar_begin, 
                                         typename libstl::vector<FieldT>::const_iterator scalar_end,
                                         const T& t_instance)
{
    T result(t_instance.zero());
    // assert(vec.size() == scalar.size());

    typename libstl::vector<T>::const_iterator vec_it;
    typename libstl::vector<FieldT>::const_iterator scalar_it;
    for(vec_it = vec_begin, scalar_it = scalar_begin; vec_it != vec_end; vec_it++, scalar_it++)
    {
        result = result + (*scalar_it) * (*vec_it);
    }

    return result;
}


static __device__ libstl::ParallalAllocator* _multi_exp_inner_BDLO12_allocator;


template<typename T, typename FieldT>
__device__ T multi_exp_inner_BDLO12(typename libstl::vector<T>::const_iterator vec_begin, 
                                         typename libstl::vector<T>::const_iterator vec_end, 
                                         typename libstl::vector<FieldT>::const_iterator scalar_begin, 
                                         typename libstl::vector<FieldT>::const_iterator scalar_end,
                                         const T& t_instance)
{
    size_t length = vec_end - vec_begin;
    size_t log2_length = log2(length);
    size_t c = log2_length - (log2_length / 3 - 2);

    libstl::vector<bigint<4>> bn_exponents;
    bn_exponents.set_parallel_allocator(_multi_exp_inner_BDLO12_allocator);
    bn_exponents.resize(length);
    size_t num_bits = 0;

    for (size_t i = 0; i < length; i++)
    {
        bn_exponents[i] = scalar_begin[i].as_bigint();
        num_bits = num_bits > bn_exponents[i].num_bits() ? num_bits : bn_exponents[i].num_bits();
    }
    size_t num_groups = (num_bits + c - 1) / c;

    T result = t_instance.zero();
    bool result_nonzero = false;

    libstl::vector<T> buckets;
    libstl::vector<bool> bucket_nonzero;
    buckets.set_parallel_allocator(_multi_exp_inner_BDLO12_allocator);
    bucket_nonzero.set_parallel_allocator(_multi_exp_inner_BDLO12_allocator);
    buckets.resize(1 << c);
    bucket_nonzero.resize(1 << c);

    for (size_t k = num_groups - 1; k <= num_groups; k--)
    {
        if (result_nonzero)
        {
            for (size_t i = 0; i < c; i++)
            {
                result = result.dbl();
            }
        }

        for(size_t i=0; i < (1 << c); i++)
        {
            bucket_nonzero[i] = false;
        }
        // libstl::vector<T> buckets(1 << c);
        // libstl::vector<bool> bucket_nonzero(1 << c, false);

        for (size_t i = 0; i < length; i++)
        {
            size_t id = 0;
            for (size_t j = 0; j < c; j++)
            {
                if (bn_exponents[i].test_bit(k*c + j))
                {
                    id |= 1 << j;
                }
            }
            if(id == 0) continue;

            if (bucket_nonzero[id])
            {
                buckets[id] = buckets[id] + vec_begin[i];
            }
            else
            {
                buckets[id] = vec_begin[i];
                bucket_nonzero[id] = true;
            }
        }

        T running_sum = t_instance.zero();
        bool running_sum_nonzero = false;

        for (size_t i = (1u << c) - 1; i > 0; i--)
        {
            if (bucket_nonzero[i])
            {
                if (running_sum_nonzero)
                {
                    running_sum = running_sum + buckets[i];
                }
                else
                {
                    running_sum = buckets[i];
                    running_sum_nonzero = true;
                }
            }

            if (running_sum_nonzero)
            {
                if (result_nonzero)
                {
                    result = result + running_sum;
                }
                else
                {
                    result = running_sum;
                    result_nonzero = true;
                }
            }
        }
    }

    return result;
}


template<typename T, typename FieldT>
__device__ T multi_exp_inner_bos_coster(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance)
{
    T result(t_instance.zero());
    return result;
}


template<typename T, typename FieldT>
__device__ T multi_exp_inner_BDLO12(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance)
{
    T result(t_instance.zero());
    return result;
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T multi_exp_inner(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance)
{
    if(Method == multi_exp_method_naive)
    {
        // return multi_exp_inner_naive<T, FieldT>(vec, scalar, t_instance);
    }
    if(Method == multi_exp_method_naive_plain)
    {
        return multi_exp_inner_naive_plain<T, FieldT>(vec, scalar, t_instance);
    }
    if(Method == multi_exp_method_bos_coster)
    {
        // return multi_exp_inner_bos_coster<T, FieldT>(vec, scalar, t_instance);
    }
    if(Method == multi_exp_method_BDLO12)
    {
        // return multi_exp_inner_BDLO12<T, FieldT>(vec, scalar, t_instance);
    }

    return multi_exp_inner_naive_plain<T, FieldT>(vec, scalar, t_instance);
    // return multi_exp_inner_naive<T, FieldT>(vec, scalar, t_instance);
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T multi_exp_inner(typename libstl::vector<T>::const_iterator vec_begin, 
                             typename libstl::vector<T>::const_iterator vec_end, 
                             typename libstl::vector<FieldT>::const_iterator scalar_begin, 
                             typename libstl::vector<FieldT>::const_iterator scalar_end,
                             const T& t_instance)
{
    if(Method == multi_exp_method_naive_plain)
    {
        return multi_exp_inner_naive_plain<T, FieldT>(vec_begin, vec_end, scalar_begin, scalar_end, t_instance);
    }
        
    if(Method == multi_exp_method_BDLO12)
    {
        return multi_exp_inner_BDLO12<T, FieldT>(vec_begin, vec_end, scalar_begin, scalar_end, t_instance);
    }


    return multi_exp_inner_naive_plain<T, FieldT>(vec_begin, vec_end, scalar_begin, scalar_end, t_instance);
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T multi_exp(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const size_t chunks, const T& t_instance)
{
    assert(chunks == 1);
    // const size_t total = vec_size;
    // if ((total < chunks) || (chunks == 1))
    // {
        // no need to split into "chunks", can call implementation directly
    return multi_exp_inner<T, FieldT, Method>(vec, scalar, t_instance);
    // }
}

template<typename T, typename FieldT, multi_exp_method Method>
static __global__ void p_multi_exp_mid_res(libstl::vector<T>& mid_res, const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance)
{
    size_t total = vec.size();

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t tnum = blockDim.x * gridDim.x;
    size_t range_s = (total + tnum - 1) / tnum * tid;
    size_t range_e = (total + tnum - 1) / tnum * (tid + 1);

    if(range_s < total)
    {
        if(range_e >= total) range_e = total;
        mid_res[tid] = multi_exp_inner<T, FieldT, Method>(vec.begin() + range_s, vec.begin() + range_e,
                                  scalar.begin() + range_s, scalar.begin() + range_e,
                                  t_instance);
    }

    __syncthreads();
    if(threadIdx.x == 0)
    {
        for(size_t i=1; i < blockDim.x; i++)
        {
            mid_res[tid] = mid_res[tid] + mid_res[tid + i];
        }
    }
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const T& t_instance, size_t gridSize, size_t blockSize)
{
    size_t tnum = gridSize * blockSize;
    T* res = libstl::create<T>(t_instance.zero());
    libstl::vector<T>* mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>(tnum, t_instance.zero()));

    if(Method == multi_exp_method_BDLO12)
    {
        size_t total = vec.size();
        size_t tnum = gridSize * blockSize;
        size_t length = (total + tnum - 1) / tnum;
        size_t log2_length = log2(length);
        size_t c = log2_length - (log2_length / 3 - 2);
        _multi_exp_inner_BDLO12_allocator = libstl::allocate(gridSize, blockSize, length * sizeof(bigint<4>) + (1 << c) * (sizeof(T) + sizeof(bool)));
    }
    p_multi_exp_mid_res<T, FieldT, Method><<<gridSize, blockSize>>>(*mid_res, vec, scalar, t_instance);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < gridSize; ++i)
    {
        (*res) = (*res) + (*mid_res)[i * blockSize];
    }
    return *res;
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp_faster_multi_GPU(libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const FieldT& instance, const T& t_instance, size_t gridSize, size_t blockSize)
{
    int device_count;
    int device_id;
    cudaGetDeviceCount(&device_count);
    cudaGetDevice(&device_id);


    size_t tnum = gridSize * blockSize;
    size_t total = vec.size();
    size_t max_length = (total + tnum - 1) / tnum;
    size_t log2_total = log2(total);
    size_t c = log2_total - (log2_total / 3 - 2);

    size_t num_bits = instance.as_bigint().max_bits();
    size_t num_groups = (num_bits + c - 1) / c;

    size_t sgroup = (num_groups + device_count - 1) / device_count * (device_id);
    size_t egroup = (num_groups + device_count - 1) / device_count * (device_id + 1);
    if(sgroup > num_groups) sgroup = num_groups;
    if(egroup > num_groups) egroup = num_groups;

    T result = t_instance.zero();
    if(sgroup == egroup) return result;

    size_t group_grid = 1;
    while(group_grid * 2 < gridSize / (egroup - sgroup)) group_grid *= 2;

    libmatrix::ELL_matrix_opt<T>* p_ell_mtx = libstl::create<libmatrix::ELL_matrix_opt<T>>(libmatrix::ELL_matrix_opt<T>());
    p_ell_mtx->p_init(max_length, tnum, (size_t)1 << c, gridSize, blockSize);

    T* p_t_zero = libstl::create<T>(t_instance.zero());
    libstl::vector<libstl::vector<T>>* mid_res = libstl::create<libstl::vector<libstl::vector<T>>>(libstl::vector<libstl::vector<T>>((egroup - sgroup)));
    for(size_t i = 0; i < (egroup - sgroup); i++)
    {
        (*mid_res)[i].presize(group_grid * blockSize, *p_t_zero, group_grid, blockSize);
    }

    libstl::vector<libstl::vector<T>*>* v_buckets = libstl::create<libstl::vector<libstl::vector<T>*>>(libstl::vector<libstl::vector<T>*>(egroup - sgroup));

    for(size_t k = sgroup; k < egroup; k++)
    {   
        // libstl::lock();

        p_ell_mtx->p_reset(gridSize, blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=, &vec, &scalar]
            __device__ ()
            {
                size_t total = vec.size();
                size_t log2_total = log2(total);
                size_t c = log2_total - (log2_total / 3 - 2);

                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = gridDim.x * blockDim.x;
                size_t range_s = (total + tnum - 1) / tnum * tid;
                size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
                for(size_t i = range_s; i < range_e && i < total; i++)
                {
                    size_t id = 0;
                    auto bn_scalar = scalar[i].as_bigint();
                    for(size_t j = 0; j < c; j++)
                    {
                        if (bn_scalar.test_bit(k * c + j))
                        {
                            id |= 1 << j;
                        }
                    }
                    p_ell_mtx->insert(tid, id);
                }
            }
        );
        cudaDeviceSynchronize();

        p_ell_mtx->total = total;
        libmatrix::CSR_matrix_opt<T>* p_csr_mtx = libmatrix::p_transpose_ell2csr(*p_ell_mtx, gridSize, blockSize);

        if(k == num_groups - 1)
            (*v_buckets)[k - sgroup] = libmatrix::p_spmv_csr_balanced_vector_one(*p_csr_mtx, vec, *p_t_zero, gridSize, blockSize);
        else
            (*v_buckets)[k - sgroup] = libmatrix::p_spmv_csr_scalar_vector_one(*p_csr_mtx, vec, *p_t_zero, gridSize, blockSize);
    }

    libstl::launch<<<group_grid * (egroup - sgroup), blockSize>>>
    (
        [=, &vec, &scalar, &t_instance]
        __device__ ()
        {
            size_t gid = blockIdx.x / group_grid;
            size_t gnum = group_grid * blockDim.x;
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t gtid = tid % gnum;

            size_t total = (*v_buckets)[gid]->size();
            size_t range_s = (total + gnum - 1) / gnum * gtid;
            size_t range_e = (total + gnum - 1) / gnum * (gtid + 1);

            T result = t_instance.zero();
            T running_sum = t_instance.zero();
            libstl::vector<T>* buckets = (*v_buckets)[gid];
            for(size_t i = range_e > total ? total - 1 : range_e - 1; i >= range_s && i > 0; i--)
            {
                running_sum = running_sum + (*buckets)[i];
                result = result + running_sum;
            }

            if(range_s != 0)
                result = result - running_sum;

            result = result + running_sum * range_s;

            (*mid_res)[gid][gtid] = result;
            // running_sum = running_sum * (range_e - range_s);
            // (*pre_sum)[gid][gnum - 1 - gtid] = running_sum;
        }
    );
    cudaDeviceSynchronize();

    size_t t_count = group_grid * blockSize;
    size_t count = 1;
    while(t_count != 1)
    {
        libstl::launch<<<group_grid * (egroup - sgroup), blockSize>>>
        (
            [=, &vec, &scalar]
            __device__ ()
            {
                size_t gid = blockIdx.x / group_grid;
                size_t gnum = group_grid * blockDim.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t gtid = tid % gnum;

                // reduce local sums to row sum

                if(gtid % (count * 2) == 0)
                {
                    if(gtid + count < group_grid * blockDim.x)
                    {
                        (*mid_res)[gid][gtid] = (*mid_res)[gid][gtid] + (*mid_res)[gid][gtid + count];
                    }
                }
            }
        );
        cudaDeviceSynchronize();

        t_count = (t_count + 1) / 2;
        count *= 2;
    }

    for(size_t k = num_groups - 1; k <= num_groups; k--)
    {
        for (size_t i = 0; i < c; i++)
        {
            result = result.dbl();
        }
        if(k >= sgroup && k < egroup)
        {
            result = result + (*mid_res)[k-sgroup][0];
        }
    }

    return result;
}



template<typename T, typename FieldT, multi_exp_method Method>
__host__ T* p_multi_exp_faster_multi_GPU_host(libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, FieldT& instance, T& t_instance, size_t gridSize, size_t blockSize)
{
    T* result = t_instance.zero_host();
    size_t msmlockMem;
    libstl::lock_host(msmlockMem);

    size_t tnum = gridSize * blockSize;
    size_t total = vec.size_host();
    size_t max_length = (total + tnum - 1) / tnum;

    size_t* dlog2_total = libstl::create_host<size_t>();
    size_t* dc = libstl::create_host<size_t>();
    size_t* dnum_bits = libstl::create_host<size_t>();
    size_t* dnum_groups = libstl::create_host<size_t>();
    size_t* dsgroup = libstl::create_host<size_t>();
    size_t* degroup = libstl::create_host<size_t>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (FieldT& instance)
        {
            int device_count;
            int device_id;
            cudaGetDeviceCount(&device_count);
            cudaGetDevice(&device_id);

            *dlog2_total = log2(total);
            *dc = *dlog2_total - (*dlog2_total / 3 - 2);
            *dnum_bits = instance.size_in_bits();
            *dnum_groups = (*dnum_bits + *dc - 1) / *dc;
            *dsgroup = (*dnum_groups + device_count - 1) / device_count * (device_id);
            *degroup = (*dnum_groups + device_count - 1) / device_count * (device_id + 1);
            if(*dsgroup > *dnum_groups) *dsgroup = *dnum_groups;
            if(*degroup > *dnum_groups) *degroup = *dnum_groups;
        }, instance
    );
    cudaStreamSynchronize(0);

    size_t log2_total;
    size_t c;
    size_t num_bits;
    size_t num_groups;
    size_t sgroup;
    size_t egroup;

    cudaMemcpy(&log2_total, dlog2_total, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, dc, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_bits, dnum_bits, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_groups, dnum_groups, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sgroup, dsgroup, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&egroup, degroup, sizeof(size_t), cudaMemcpyDeviceToHost);



    if(egroup == sgroup) return result;

    size_t group_grid = 1;
    while(group_grid * 2 < gridSize / (egroup - sgroup)) group_grid *= 2;

    libmatrix::ELL_matrix_opt<T>* p_ell_mtx = libstl::create_host<libmatrix::ELL_matrix_opt<T>>();
    p_ell_mtx->p_init_host(max_length, tnum, (size_t)1 << c, gridSize, blockSize);

    T* p_t_zero = t_instance.zero_host();

    libstl::vector<T>* mid_res = (libstl::vector<T>*)libstl::allocate_host((egroup - sgroup) * sizeof(libstl::vector<T>));
    for(size_t i = 0; i < (egroup - sgroup); i++)
    {
        libstl::vector<T>* mid_res_i_addr = mid_res + i;
        libstl::construct_host(mid_res_i_addr);
        mid_res_i_addr->presize_host(group_grid * blockSize, p_t_zero, group_grid, blockSize);
    }
    libstl::vector<T>* v_buckets =  (libstl::vector<T>*)libstl::allocate_host((egroup - sgroup) * sizeof(libstl::vector<T>));
    for(size_t i = 0; i < (egroup - sgroup); i++)
    {
        libstl::vector<T>* v_buckets_i_addr = v_buckets + i;
        libstl::construct_host(v_buckets_i_addr);
        v_buckets_i_addr->presize_host((size_t)1 << c, p_t_zero, group_grid, blockSize);
    }

    for(size_t k = sgroup; k < egroup; k++)
    {
        p_ell_mtx->p_reset_host(gridSize, blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ (libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar)
            {
                size_t total = vec.size();
                size_t log2_total = log2(total);
                size_t c = log2_total - (log2_total / 3 - 2);

                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = gridDim.x * blockDim.x;
                size_t range_s = (total + tnum - 1) / tnum * tid;
                size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
                for(size_t i = range_s; i < range_e && i < total; i++)
                {
                    size_t id = 0;
                    auto bn_scalar = scalar[i].as_bigint();
                    for(size_t j = 0; j < c; j++)
                    {
                        if (bn_scalar.test_bit(k * c + j))
                        {
                            id |= 1 << j;
                        }
                    }
                    p_ell_mtx->insert(tid, id);
                }

                if(tid == 0)
                    p_ell_mtx->total = total;

            }, vec, scalar
        );
        cudaStreamSynchronize(0);

        size_t lockMem;
        libstl::lock_host(lockMem);

        libmatrix::CSR_matrix_opt<T>* p_csr_mtx = libmatrix::p_transpose_ell2csr_host(*p_ell_mtx, gridSize, blockSize);
        libmatrix::p_spmv_csr_balanced_vector_one_host(v_buckets + (k - sgroup), *p_csr_mtx, vec, *p_t_zero, gridSize, blockSize);

        libstl::resetlock_host(lockMem);
        // cudaMemcpy(&v_buckets[k - sgroup], &buckets, sizeof(libstl::vector<T>*), cudaMemcpyHostToDevice);
    }

    libstl::launch<<<group_grid * (egroup - sgroup), blockSize>>>
    (
        [=]
        __device__ (libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, T& t_instance)
        {
            size_t gid = blockIdx.x / group_grid;
            size_t gnum = group_grid * blockDim.x;
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t gtid = tid % gnum;

            size_t total = v_buckets[gid].size();
            size_t range_s = (total + gnum - 1) / gnum * gtid;
            size_t range_e = (total + gnum - 1) / gnum * (gtid + 1);

            T result = t_instance.zero();
            T running_sum = t_instance.zero();
            libstl::vector<T>* buckets = v_buckets + gid;
            for(size_t i = range_e > total ? total - 1 : range_e - 1; i >= range_s && i > 0; i--)
            {
                running_sum = running_sum + (*buckets)[i];
                result = result + running_sum;
            }

            if(range_s != 0)
                result = result - running_sum;

            result = result + running_sum * range_s;

            mid_res[gid][gtid] = result;

        }, vec, scalar, t_instance
    );
    cudaStreamSynchronize(0);

    size_t t_count = group_grid * blockSize;
    size_t count = 1;
    while(t_count != 1)
    {
        libstl::launch<<<group_grid * (egroup - sgroup), blockSize>>>
        (
            [=]
            __device__ (libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar)
            {
                size_t gid = blockIdx.x / group_grid;
                size_t gnum = group_grid * blockDim.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t gtid = tid % gnum;

                // reduce local sums to row sum
                if(gtid % (count * 2) == 0)
                {
                    if(gtid + count < group_grid * blockDim.x)
                    {
                        mid_res[gid][gtid] = mid_res[gid][gtid] + mid_res[gid][gtid + count];
                    }
                }
            }, vec, scalar
        );
        cudaStreamSynchronize(0);

        t_count = (t_count + 1) / 2;
        count *= 2;
    }

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            for(size_t k = num_groups - 1; k <= num_groups; k--)
            {
                if(k >= sgroup && k < egroup)
                {
                    for (size_t i = 0; i < c; i++)
                    {
                        *result = result->dbl();
                    }
                    *result = *result + mid_res[k-sgroup][0];
                }
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::resetlock_host(msmlockMem);

    return result;
}



template<typename T, typename FieldT, multi_exp_method Method>
__host__ void p_multi_exp_faster_multi_GPU_host(T* result, libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, FieldT& instance, T& t_instance, size_t gridSize, size_t blockSize)
{
    // T* result = t_instance.zero_host();
    size_t msmlockMem;
    libstl::lock_host(msmlockMem);

    size_t tnum = gridSize * blockSize;
    size_t total = vec.size_host();
    size_t max_length = (total + tnum - 1) / tnum;

    size_t* dlog2_total = libstl::create_host<size_t>();
    size_t* dc = libstl::create_host<size_t>();
    size_t* dnum_bits = libstl::create_host<size_t>();
    size_t* dnum_groups = libstl::create_host<size_t>();
    size_t* dsgroup = libstl::create_host<size_t>();
    size_t* degroup = libstl::create_host<size_t>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (FieldT& instance)
        {
            int device_count;
            int device_id;
            cudaGetDeviceCount(&device_count);
            cudaGetDevice(&device_id);

            *dlog2_total = log2(total);
            *dc = *dlog2_total - (*dlog2_total / 3 - 2);
            *dnum_bits = instance.size_in_bits();
            *dnum_groups = (*dnum_bits + *dc - 1) / *dc;
            *dsgroup = (*dnum_groups + device_count - 1) / device_count * (device_id);
            *degroup = (*dnum_groups + device_count - 1) / device_count * (device_id + 1);
            if(*dsgroup > *dnum_groups) *dsgroup = *dnum_groups;
            if(*degroup > *dnum_groups) *degroup = *dnum_groups;
        }, instance
    );
    cudaStreamSynchronize(0);

    size_t log2_total;
    size_t c;
    size_t num_bits;
    size_t num_groups;
    size_t sgroup;
    size_t egroup;

    cudaMemcpy(&log2_total, dlog2_total, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, dc, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_bits, dnum_bits, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_groups, dnum_groups, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sgroup, dsgroup, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&egroup, degroup, sizeof(size_t), cudaMemcpyDeviceToHost);


    if(egroup == sgroup) return ;

    size_t group_grid = 1;
    while(group_grid * 2 < gridSize / (egroup - sgroup)) group_grid *= 2;

    libmatrix::ELL_matrix_opt<T>* p_ell_mtx = libstl::create_host<libmatrix::ELL_matrix_opt<T>>();
    p_ell_mtx->p_init_host(max_length, tnum, (size_t)1 << c, gridSize, blockSize);

    T* p_t_zero = t_instance.zero_host();

    libstl::vector<T>* mid_res = (libstl::vector<T>*)libstl::allocate_host((egroup - sgroup) * sizeof(libstl::vector<T>));
    for(size_t i = 0; i < (egroup - sgroup); i++)
    {
        libstl::vector<T>* mid_res_i_addr = mid_res + i;
        libstl::construct_host(mid_res_i_addr);
        mid_res_i_addr->presize_host(group_grid * blockSize, p_t_zero, group_grid, blockSize);
    }
    libstl::vector<T>* v_buckets =  (libstl::vector<T>*)libstl::allocate_host((egroup - sgroup) * sizeof(libstl::vector<T>));
    for(size_t i = 0; i < (egroup - sgroup); i++)
    {
        libstl::vector<T>* v_buckets_i_addr = v_buckets + i;
        libstl::construct_host(v_buckets_i_addr);
        v_buckets_i_addr->presize_host((size_t)1 << c, p_t_zero, group_grid, blockSize);
    }

    for(size_t k = sgroup; k < egroup; k++)
    {
        p_ell_mtx->p_reset_host(gridSize, blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ (libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar)
            {
                size_t total = vec.size();
                size_t log2_total = log2(total);
                size_t c = log2_total - (log2_total / 3 - 2);

                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = gridDim.x * blockDim.x;
                size_t range_s = (total + tnum - 1) / tnum * tid;
                size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
                for(size_t i = range_s; i < range_e && i < total; i++)
                {
                    size_t id = 0;
                    auto bn_scalar = scalar[i].as_bigint();
                    for(size_t j = 0; j < c; j++)
                    {
                        if (bn_scalar.test_bit(k * c + j))
                        {
                            id |= 1 << j;
                        }
                    }
                    p_ell_mtx->insert(tid, id);
                }

                if(tid == 0)
                    p_ell_mtx->total = total;

            }, vec, scalar
        );
        cudaStreamSynchronize(0);

        size_t lockMem;
        libstl::lock_host(lockMem);

        libmatrix::CSR_matrix_opt<T>* p_csr_mtx = libmatrix::p_transpose_ell2csr_host(*p_ell_mtx, gridSize, blockSize);
        libmatrix::p_spmv_csr_balanced_vector_one_host(v_buckets + (k - sgroup), *p_csr_mtx, vec, *p_t_zero, gridSize, blockSize);

        libstl::resetlock_host(lockMem);
        // cudaMemcpy(&v_buckets[k - sgroup], &buckets, sizeof(libstl::vector<T>*), cudaMemcpyHostToDevice);
    }

    libstl::launch<<<group_grid * (egroup - sgroup), blockSize>>>
    (
        [=]
        __device__ (libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, T& t_instance)
        {
            size_t gid = blockIdx.x / group_grid;
            size_t gnum = group_grid * blockDim.x;
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t gtid = tid % gnum;

            size_t total = v_buckets[gid].size();
            size_t range_s = (total + gnum - 1) / gnum * gtid;
            size_t range_e = (total + gnum - 1) / gnum * (gtid + 1);

            T result = t_instance.zero();
            T running_sum = t_instance.zero();
            libstl::vector<T>* buckets = v_buckets + gid;
            for(size_t i = range_e > total ? total - 1 : range_e - 1; i >= range_s && i > 0; i--)
            {
                running_sum = running_sum + (*buckets)[i];
                result = result + running_sum;
            }

            if(range_s != 0)
                result = result - running_sum;

            result = result + running_sum * range_s;

            mid_res[gid][gtid] = result;

        }, vec, scalar, t_instance
    );
    cudaStreamSynchronize(0);

    size_t t_count = group_grid * blockSize;
    size_t count = 1;
    while(t_count != 1)
    {
        libstl::launch<<<group_grid * (egroup - sgroup), blockSize>>>
        (
            [=]
            __device__ (libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar)
            {
                size_t gid = blockIdx.x / group_grid;
                size_t gnum = group_grid * blockDim.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t gtid = tid % gnum;

                // reduce local sums to row sum
                if(gtid % (count * 2) == 0)
                {
                    if(gtid + count < group_grid * blockDim.x)
                    {
                        mid_res[gid][gtid] = mid_res[gid][gtid] + mid_res[gid][gtid + count];
                    }
                }
            }, vec, scalar
        );
        cudaStreamSynchronize(0);

        t_count = (t_count + 1) / 2;
        count *= 2;
    }

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            for(size_t k = num_groups - 1; k <= num_groups; k--)
            {
                if(k >= sgroup && k < egroup)
                {
                    for (size_t i = 0; i < c; i++)
                    {
                        *result = result->dbl();
                    }
                    *result = *result + mid_res[k-sgroup][0];
                }
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::resetlock_host(msmlockMem);

    return ;
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp_faster(libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const FieldT& instance, const T& t_instance, size_t gridSize, size_t blockSize)
{
    size_t tnum = gridSize * blockSize;
    size_t total = vec.size();
    size_t max_length = (total + tnum - 1) / tnum;
    size_t log2_total = log2(total);
    size_t c = log2_total - (log2_total / 3 - 2);

    size_t num_bits = instance.as_bigint().max_bits();
    size_t num_groups = (num_bits + c - 1) / c;

    size_t group_grid = 1;
    while(group_grid * 2 < gridSize / num_groups) group_grid *= 2;

    T result = t_instance.zero();

    libmatrix::ELL_matrix_opt<T>* p_ell_mtx = libstl::create<libmatrix::ELL_matrix_opt<T>>(libmatrix::ELL_matrix_opt<T>());
    p_ell_mtx->p_init(max_length, tnum, (size_t)1 << c, gridSize, blockSize);

    T* p_t_zero = libstl::create<T>(t_instance.zero());
    libstl::vector<libstl::vector<T>>* mid_res = libstl::create<libstl::vector<libstl::vector<T>>>(libstl::vector<libstl::vector<T>>(num_groups));
    // libstl::vector<libstl::vector<T>>* pre_sum = libstl::create<libstl::vector<libstl::vector<T>>>(libstl::vector<libstl::vector<T>>(num_groups));
    for(size_t i = 0; i < num_groups; i++)
    {
        (*mid_res)[i].presize(group_grid * blockSize, *p_t_zero, group_grid, blockSize);
        // (*pre_sum)[i].presize(group_grid * blockSize, *p_t_zero, group_grid, blockSize);
    }
    libstl::vector<libstl::vector<T>*>* v_buckets = libstl::create<libstl::vector<libstl::vector<T>*>>(libstl::vector<libstl::vector<T>*>(num_groups));

    for(size_t k = num_groups - 1; k <= num_groups; k--)
    {   
        // libstl::lock();

        p_ell_mtx->p_reset(gridSize, blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=, &vec, &scalar]
            __device__ ()
            {
                size_t total = vec.size();
                size_t log2_total = log2(total);
                size_t c = log2_total - (log2_total / 3 - 2);

                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t tnum = gridDim.x * blockDim.x;
                size_t range_s = (total + tnum - 1) / tnum * tid;
                size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
                for(size_t i = range_s; i < range_e && i < total; i++)
                {
                    size_t id = 0;
                    auto bn_scalar = scalar[i].as_bigint();
                    for(size_t j = 0; j < c; j++)
                    {
                        if (bn_scalar.test_bit(k * c + j))
                        {
                            id |= 1 << j;
                        }
                    }
                    p_ell_mtx->insert(tid, id);
                }
            }
        );
        cudaDeviceSynchronize();

        p_ell_mtx->total = total;
        libmatrix::CSR_matrix_opt<T>* p_csr_mtx = libmatrix::p_transpose_ell2csr(*p_ell_mtx, gridSize, blockSize);

        if(k == num_groups - 1)
            (*v_buckets)[k] = libmatrix::p_spmv_csr_balanced_vector_one(*p_csr_mtx, vec, *p_t_zero, gridSize, blockSize);
        else
            (*v_buckets)[k] = libmatrix::p_spmv_csr_scalar_vector_one(*p_csr_mtx, vec, *p_t_zero, gridSize, blockSize);
    }

    libstl::launch<<<group_grid * num_groups, blockSize>>>
    (
        [=, &vec, &scalar, &t_instance]
        __device__ ()
        {
            size_t gid = blockIdx.x / group_grid;
            size_t gnum = group_grid * blockDim.x;
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t gtid = tid % gnum;

            size_t total = (*v_buckets)[gid]->size();
            size_t range_s = (total + gnum - 1) / gnum * gtid;
            size_t range_e = (total + gnum - 1) / gnum * (gtid + 1);

            T result = t_instance.zero();
            T running_sum = t_instance.zero();
            libstl::vector<T>* buckets = (*v_buckets)[gid];
            for(size_t i = range_e > total ? total - 1 : range_e - 1; i >= range_s && i > 0; i--)
            {
                running_sum = running_sum + (*buckets)[i];
                result = result + running_sum;
            }

            if(range_s != 0)
                result = result - running_sum;

            result = result + running_sum * range_s;

            (*mid_res)[gid][gtid] = result;
            // running_sum = running_sum * (range_e - range_s);
            // (*pre_sum)[gid][gnum - 1 - gtid] = running_sum;
        }
    );
    cudaDeviceSynchronize();

    size_t t_count = group_grid * blockSize;
    size_t count = 1;
    while(t_count != 1)
    {
        libstl::launch<<<group_grid * num_groups, blockSize>>>
        (
            [=, &vec, &scalar]
            __device__ ()
            {
                size_t gid = blockIdx.x / group_grid;
                size_t gnum = group_grid * blockDim.x;
                size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                size_t gtid = tid % gnum;

                // reduce local sums to row sum

                if(gtid % (count * 2) == 0)
                {
                    if(gtid + count < group_grid * blockDim.x)
                    {
                        (*mid_res)[gid][gtid] = (*mid_res)[gid][gtid] + (*mid_res)[gid][gtid + count];
                    }
                }
            }
        );
        cudaDeviceSynchronize();

        t_count = (t_count + 1) / 2;
        count *= 2;
    }

    for(size_t k = num_groups - 1; k <= num_groups; k--)
    {
        for (size_t i = 0; i < c; i++)
        {
            result = result.dbl();
        }
        result = result + (*mid_res)[k][0];
    }
    return result;
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T multi_exp_with_mixed_addition(libstl::vector<T> vec, libstl::vector<FieldT> scalar, const size_t chunks, const T& t_instance)
{
    assert(chunks == 1);

    const FieldT& instance = scalar[0];
    const FieldT zero = instance.zero();
    const FieldT one = instance.one();

    T acc = t_instance.zero();

    size_t num_skip = 0;
    size_t num_add = 0;
    size_t num_other = 0;

    for(size_t i=0; i<vec.size(); i++)
    {
        if (scalar[i] == zero)
        {
            ++num_skip;
        }
        else if (scalar[i] == one)
        {
            ++num_add;
        }
        else
        {
            ++num_other;
        }
    }
    libstl::vector<FieldT> p(num_other);
    libstl::vector<T> g(num_other);

    for(size_t i=0; i<vec.size(); i++)
    {
        if (scalar[i] == zero)
        {
        }
        else if (scalar[i] == one)
        {
            acc = acc + vec[i];
        }
        else
        {
            p[i] = scalar[i];
            g[i] = vec[i];
        }
    }

    return acc + multi_exp<T, FieldT, Method>(g, p, chunks, t_instance);
}


template<typename T, typename FieldT, multi_exp_method Method>
__device__ T p_multi_exp_with_mixed_addition(const libstl::vector<T>& vec, const libstl::vector<FieldT>& scalar, const FieldT& instance, const T& t_instance, size_t gridSize, size_t blockSize)
{
    size_t tnum = gridSize * blockSize;
    libstl::vector<size_t>* num_other_mid_res_thread = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>(tnum, 0));
    libstl::vector<size_t>* num_other_mid_res_block = libstl::create<libstl::vector<size_t>>(libstl::vector<size_t>(gridSize, 0));
    libstl::vector<T>* acc_mid_res = libstl::create<libstl::vector<T>>(libstl::vector<T>(tnum, t_instance.zero()));
    T* acc = libstl::create<T>(t_instance.zero());
    size_t num_other = 0;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &vec, &scalar, &instance, &t_instance]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t range_s = (vec.size() + tnum - 1) / tnum * (idx);
            size_t range_e = (vec.size() + tnum - 1) / tnum * (idx + 1);

            const FieldT zero = instance.zero();
            const FieldT one = instance.one();

            for(size_t i = range_s; i < range_e && i < vec.size(); i++)
            {
                if(scalar[i] == zero) continue;
                else if(scalar[i] == one)
                {
                    (*acc_mid_res)[idx] = (*acc_mid_res)[idx] + vec[i];
                }
                else
                {
                    (*num_other_mid_res_thread)[idx] += 1;
                }
            }
            __syncthreads();

            if(threadIdx.x == 0)
            {
                for(size_t i = 1; i < blockDim.x; i++)
                {
                    (*acc_mid_res)[idx] = (*acc_mid_res)[idx] + (*acc_mid_res)[idx + i];
                    (*num_other_mid_res_block)[blockIdx.x] += (*num_other_mid_res_thread)[idx + i];
                }
                (*num_other_mid_res_block)[blockIdx.x] += (*num_other_mid_res_thread)[idx];
            }
        }
    );
    cudaDeviceSynchronize();
    
    for(size_t i = 0; i < gridSize; i++)
    {
        *acc = *acc + (*acc_mid_res)[i * blockSize];
        num_other += (*num_other_mid_res_block)[i];
    }

    libstl::vector<FieldT>* p = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>(num_other, instance.zero()));
    libstl::vector<T>* g = libstl::create<libstl::vector<T>>(libstl::vector<T>(num_other, t_instance.zero()));

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &vec, &scalar, &instance, &t_instance]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t range_s = (vec.size() + tnum - 1) / tnum * (idx);
            size_t range_e = (vec.size() + tnum - 1) / tnum * (idx + 1);
            size_t store_idx = 0;
            size_t num_other_count = 0;

            const FieldT zero = instance.zero();
            const FieldT one = instance.one();

            for(size_t i = 0; i < idx; i++)
            {
                store_idx += (*num_other_mid_res_thread)[i];
            }

            for(size_t i = range_s; i < range_e && i < vec.size(); i++)
            {
                if(scalar[i] == zero) continue;
                else if(scalar[i] == one) continue;
                else
                {
                    (*p)[store_idx + num_other_count] = scalar[i];
                    (*g)[store_idx + num_other_count] = vec[i];
                    num_other_count++;
                }
            }
        }
    );
    cudaDeviceSynchronize();

    

    return *acc + p_multi_exp<T, FieldT, Method>(*g, *p, t_instance, gridSize, blockSize);;
}



// template <typename T>
// __device__ T inner_product(T* a, size_t a_size, T* b, size_t b_size)
// {
//     return multi_exp<T, T, multi_exp_method_naive_plain>(
//         a, a_size,
//         b, b_size, 1);
// }


template<typename T>
__device__ size_t get_exp_window_size(const size_t num_scalars, const T& instance)
{
    if (instance.params->fixed_base_exp_window_table_length_size == 0)
    {
        return 17;
    }
    size_t window = 1;
    for (long i = instance.params->fixed_base_exp_window_table_length_size-1; i >= 0; --i)
    {
        if (instance.params->fixed_base_exp_window_table_length[i] != 0 && num_scalars >= instance.params->fixed_base_exp_window_table_length[i])
        {
            window = i+1;
            break;
        }
    }
    return window;
}

template<typename T>
__device__ window_table<T>* p_get_window_table(const size_t scalar_size, const size_t window, const T& g, size_t gridSize, size_t blockSize)
{
    const size_t in_window = 1ul<<window;
    const size_t outerc = (scalar_size + window - 1)/window;
    const T& instance = g;
    T* zero = libstl::create<T>(instance.zero());
    window_table<T>* p_powers_of_g = libstl::create<window_table<T>>(window_table<T>());
    window_table<T>& powers_of_g = *p_powers_of_g;
    powers_of_g.resize(outerc);
    for(size_t i=0; i<outerc; i++) powers_of_g[i].presize(in_window, *zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &powers_of_g, &g]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            size_t blocks_per_row = gridDim.x / outerc;

            size_t tnum_per_row = blocks_per_row * blockDim.x;

            size_t rid = tid / tnum_per_row;
            size_t cid = tid % tnum_per_row;
    
            size_t range_s = (in_window + tnum_per_row - 1) / tnum_per_row * cid;
            size_t range_e = (in_window + tnum_per_row - 1) / tnum_per_row * (cid + 1);

            auto b = *g.params->fr_params->modulus;
            b.clear();
            b.set_bit(rid * window);
            T gouter = b * g;
            T ginner = gouter * range_s;

            for(size_t i=range_s; i<range_e && i<in_window; i++)
            {
                powers_of_g[rid][i] = ginner;
                ginner = ginner + gouter;
            }
        }
    );
    cudaDeviceSynchronize();

    return p_powers_of_g;
}


template<typename T>
__device__ window_table<T> get_window_table(const size_t scalar_size, const size_t window, const T &g)
{
    const size_t in_window = 1ul<<window;
    const size_t outerc = (scalar_size+window-1)/window;
    const size_t last_in_window = 1ul<<(scalar_size - (outerc-1)*window);

    const T& instance = g;

    window_table<T> powers_of_g(outerc, libstl::vector<T>(in_window, instance.zero()));

    T gouter = g;
    for (size_t outer = 0; outer < outerc; ++outer)
    {
        T ginner = instance.zero();
        size_t cur_in_window = outer == outerc-1 ? last_in_window : in_window;
        for (size_t inner = 0; inner < cur_in_window; ++inner)
        {
            powers_of_g[outer][inner] = ginner;
            ginner = ginner + gouter;
        }

        for (size_t i = 0; i < window; ++i)
        {
            gouter = gouter + gouter;
        }
    }
    return powers_of_g;
}


template<typename T, typename FieldT>
__device__ T windowed_exp(const size_t scalar_size, const size_t window, const window_table<T> &powers_of_g, const FieldT &pow)
{
    const size_t outerc = (scalar_size+window-1)/window;
    auto b = pow.as_bigint();

    /* exp */
    T res = powers_of_g[0][0];

    for (size_t outer = 0; outer < outerc; ++outer)
    {
        size_t inner = 0;
        for (size_t i = 0; i < window; ++i)
        {
            if (b.test_bit(outer*window + i))
            {
                inner |= 1u << i;
            }
        }
        res = res + powers_of_g[outer][inner];
    }

    return res;
}

template<typename T, typename FieldT>
__device__ libstl::vector<T>* p_batch_exp(const size_t scalar_size, const size_t window, const window_table<T>& table, const libstl::vector<FieldT>& vec, T& t_instance, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* p_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    libstl::vector<T>& res = *p_res;
    res.presize(vec.size(), gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &res, &table, &vec]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            while(tid < vec.size())
            {
                res[tid] = windowed_exp(scalar_size, window, table, vec[tid]);
                tid += blockDim.x * gridDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return p_res;
}


template<typename T, typename FieldT>
__device__ libstl::vector<T>* p_batch_exp_with_coeff(const size_t scalar_size, const size_t window, const window_table<T>& table, const FieldT &coeff, const libstl::vector<FieldT>& vec, T& t_instance, size_t gridSize, size_t blockSize)
{
    libstl::vector<T>* p_res = libstl::create<libstl::vector<T>>(libstl::vector<T>());
    libstl::vector<T>& res = *p_res;
    res.presize(vec.size(), gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &res, &table, &coeff, &vec]
        __device__ ()
        {
            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            while(tid < vec.size())
            {
                res[tid] = windowed_exp(scalar_size, window, table, coeff * vec[tid]);
                tid += blockDim.x * gridDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return p_res;
}


template<typename T, typename FieldT>
__device__ libstl::vector<T> batch_exp(const size_t scalar_size, const size_t window, const window_table<T>& table, libstl::vector<FieldT>& vec)
{
    if(table.size()==0)
    {
        return libstl::vector<T>();
    }
    else if(table[0].size() == 0)
    {
        return libstl::vector<T>();
    }

    const T& instance = table[0][0];
    libstl::vector<T> res(vec.size(), instance.zero());

    for (size_t i = 0; i < vec.size(); ++i)
    {
        res[i] = windowed_exp(scalar_size, window, table, vec[i]);
    }

    return res;
}

template<typename T, typename FieldT>
__device__ libstl::vector<T> batch_exp_with_coeff(const size_t scalar_size, const size_t window, const window_table<T>& table, const FieldT &coeff, const libstl::vector<FieldT>& vec)
{
    if(table.size()==0)
    {
        return libstl::vector<T>();
    }
    else if(table[0].size() == 0)
    {
        return libstl::vector<T>();
    }
    const T& instance = table[0][0];
    libstl::vector<T> res(vec.size(), instance.zero());

    for (size_t i = 0; i < vec.size(); ++i)
    {
        res[i] = windowed_exp(scalar_size, window, table, coeff * vec[i]);
    }

    return res;
}

}

#endif