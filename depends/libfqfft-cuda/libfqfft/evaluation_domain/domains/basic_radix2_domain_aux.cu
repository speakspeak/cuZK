#ifndef __BASIC_RADIX2_DOMAIN_AUX_CU__
#define __BASIC_RADIX2_DOMAIN_AUX_CU__

#include <assert.h>
#include "../../../depends/libff-cuda/fields/field_utils.cuh"
#include "../../../depends/libff-cuda/common/utils.cuh"
#include "../../../depends/libstl-cuda/algorithm.cuh"
#include <math.h>

namespace libfqfft {

template<typename FieldT>
__device__ void _basic_radix2_FFT(libstl::vector<FieldT>& a, const FieldT &omega)
{
    const size_t n = a.size();
    const size_t logn = libff::log2(n);     // unfinished

    assert(n == (1u << logn));

    FieldT instance = a[0];

    for (size_t k = 0; k < n; ++k)
    {
        const size_t rk = libff::bitreverse(k, logn);
        if (k < rk)
            libstl::swap(a[k], a[rk]);
    }

    size_t m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logn; ++s)
    {
        // w_m is 2^s-th root of unity now
        const FieldT w_m = omega ^ (n / (2 * m));

        for (size_t k = 0; k < n; k += 2 * m)
        {
            FieldT w = instance.one();
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w * a[k + j + m];
                a[k + j + m] = a[k + j] - t;
                a[k + j] += t;
                w *= w_m;
            }
        }
        m *= 2;
    }

    return ;
}



template<typename FieldT>
__device__ void _p_basic_radix2_FFT(libstl::vector<FieldT>& a, const FieldT& omega, size_t gridSize, size_t blockSize)
{
    size_t num = gridSize * blockSize;
    size_t log_num = ((num & (num - 1)) == 0 ? libff::log2(num) : libff::log2(num) - 1); 

    if (log_num == 0)
        return _basic_radix2_FFT(a, omega);

    const size_t m = a.size();
    const size_t log_m = libff::log2(m);

    assert(m == 1ul << log_m);

    if (log_m < log_num)
        return _basic_radix2_FFT(a, omega);

    const size_t log_diff = log_m - log_num;
    const size_t diff = 1ul << log_diff;

    libstl::ParallalAllocator* cAllocator = libstl::allocate(gridSize, blockSize, diff * sizeof(FieldT));
    libstl::vector<FieldT>* cMatrix = (libstl::vector<FieldT>*)libstl::allocate(num * sizeof(libstl::vector<FieldT>));

    FieldT* omega_exp_num = libstl::create<FieldT>(omega ^ num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a, &omega]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT instance = a[0];

            libstl::construct(&cMatrix[idx]);
            cMatrix[idx].set_parallel_allocator(cAllocator);
            cMatrix[idx].resize(diff, instance.zero());


            for (size_t i = 0; i < diff; ++i)
                cMatrix[idx][i] = a[(i << log_num) + idx];
            
            _basic_radix2_FFT(cMatrix[idx], *omega_exp_num);

            FieldT omega_exp_idx = omega ^ idx;
            FieldT omega_exp_idx_i = instance.one();

            for (size_t i = 0; i < diff; ++i)
            {
                cMatrix[idx][i] *= omega_exp_idx_i;
                omega_exp_idx_i *= omega_exp_idx;
            }
        }
    );

    libstl::ParallalAllocator* rAllocator = libstl::allocate(gridSize, blockSize, max(num, diff) * sizeof(FieldT));
    libstl::vector<FieldT>* rMatrix = (libstl::vector<FieldT>*)libstl::allocate(diff * sizeof(libstl::vector<FieldT>));

    FieldT* omega_exp_diff = libstl::create<FieldT>(omega ^ diff);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT instance = a[0];

            for (size_t i = idx; i < diff; i += num)
            {
                libstl::construct(&rMatrix[i]);
                rMatrix[i].set_parallel_allocator(rAllocator);
                rMatrix[i].resize(num, instance.zero());

                for (size_t j = 0; j < num; ++j)
                     rMatrix[i][j] = cMatrix[j][i];

                _basic_radix2_FFT(rMatrix[i], *omega_exp_diff);

                for (size_t j = 0; j < num; ++j)
                    a[(j << log_diff) + i] = rMatrix[i][j];
            }
        }
    );

    cudaDeviceSynchronize();
}

template<typename FieldT>
__device__ void _pp_basic_radix2_FFT(libstl::vector<FieldT>& a, const FieldT& omega, size_t num) 
{
    size_t m = a.size();
    size_t log_m = libff::log2(m);

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t k = idx; k < m; k += num)
            {
                size_t rk = libff::bitreverse(k, log_m);
                if (k < rk)
                     libstl::swap(a[k], a[rk]);
            }
        }
    );

    FieldT* table = (FieldT*)libstl::allocate(m / 2 * sizeof(FieldT));

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a, &omega]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t tnum = min(num, m / 2);

            if (idx < tnum)
            {
                size_t start = idx * m / (2 * tnum);
                size_t end = start + m / (2 * tnum);

                FieldT w = omega ^ start;
                for (size_t i = start; i < end; ++i, w *= omega)
                    libstl::construct(&table[i], w);
            }
        }
    );

    for (size_t s = 0; s < log_m; ++s)
    {
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=, &a, &omega]
            __device__ ()
            {            
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                size_t n = 1ul << s;

                for (size_t i = idx; i < m / 2; i += num)
                {
                    size_t k = i / n * 2 * n;
                    size_t j = i % n;

                    FieldT t = table[(m * j) / (2 * n)] * a[k + j + n];

                    a[k + j + n] = a[k + j] - t;
                    a[k + j] += t;
                }
            }
        );
    }

    cudaDeviceSynchronize();
    
    libstl::reset(m / 2 * sizeof(FieldT));
}



template<typename FieldT>
__host__ void _pp_basic_radix2_FFT_host(libstl::vector<FieldT>& a, const FieldT& omega, size_t num)
{
    size_t m = a.size_host();
    size_t log_m = libff::log2(m);

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__(libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t k = idx; k < m; k += num)
            {
                size_t rk = libff::bitreverse(k, log_m);
                if (k < rk)
                     libstl::swap(a[k], a[rk]);
            }
        }, a
    );
    cudaStreamSynchronize(0);

    FieldT* table = (FieldT*)libstl::allocate_host(m / 2 * sizeof(FieldT));
    // FieldT* table;
    // cudaMalloc((void**)&table, m / 2 * sizeof(FieldT));

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a, const FieldT& omega)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t tnum = min(num, m / 2);

            if (idx < tnum)
            {
                size_t start = idx * m / (2 * tnum);
                size_t end = start + m / (2 * tnum);

                FieldT w = omega ^ start;
                for (size_t i = start; i < end; ++i, w *= omega)
                    libstl::construct(&table[i], w);
            }
        }, a, omega
    );
    cudaStreamSynchronize(0);

    for (size_t s = 0; s < log_m; ++s)
    {
        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ (libstl::vector<FieldT>& a)
            {            
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                size_t n = 1ul << s;

                for (size_t i = idx; i < m / 2; i += num)
                {
                    size_t k = i / n * 2 * n;
                    size_t j = i % n;

                    FieldT t = table[(m * j) / (2 * n)] * a[k + j + n];

                    a[k + j + n] = a[k + j] - t;
                    a[k + j] += t;
                }
            }, a
        );
        cudaStreamSynchronize(0);
    }

    libstl::reset_host(m / 2 * sizeof(FieldT));
    // cudaFree(table);
}

template<typename FieldT>
__device__ void _multiply_by_coset(libstl::vector<FieldT>& a, const FieldT &g) 
{
    FieldT u = g;
    for (size_t i = 1; i < a.size(); ++i)
    {
        a[i] *= u;
        u *= g;
    }
}

template<typename FieldT>
__device__ void _p_multiply_by_coset(libstl::vector<FieldT>& a, const FieldT &g, size_t gridSize, size_t blockSize) 
{
    size_t num = gridSize * blockSize;
    size_t size = a.size();

    FieldT* g_exp_num = libstl::create<FieldT>(g ^ num);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a, &g]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            FieldT u = g ^ idx; 

            for (size_t i = idx; i < size; i += num)
            {
                a[i] *= u;
                u *= *g_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();
}

template<typename FieldT>
__device__ void _pp_multiply_by_coset(libstl::vector<FieldT>& a, const FieldT &g, size_t num) 
{
    size_t size = a.size();

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    FieldT* g_exp_num = libstl::create<FieldT>(g ^ num);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a, &g]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            FieldT u = g ^ idx; 

            for (size_t i = idx; i < size; i += num)
            {
                a[i] *= u;
                u *= *g_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();
}


template<typename FieldT>
__host__ void _pp_multiply_by_coset_host(libstl::vector<FieldT>& a, const FieldT &g, size_t num) 
{
    size_t size = a.size_host();

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;


    FieldT* g_exp_num = libstl::create_host<FieldT>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (const FieldT &g)
        {
            *g_exp_num = g ^ num;
        }, g
    );
    cudaStreamSynchronize(0);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a, const FieldT &g)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            FieldT u = g ^ idx; 

            for (size_t i = idx; i < size; i += num)
            {
                a[i] *= u;
                u *= *g_exp_num;
            }
        }, a, g
    );
    cudaStreamSynchronize(0);
}

static __device__ libstl::ParallalAllocator* _basic_domian_pAllocator;

template<typename FieldT>
__global__ void inverse_multi(libstl::vector<FieldT>& u, const size_t m, const FieldT& t)
{
    const FieldT& instance = t;
    const FieldT omega = libff::get_root_of_unity<FieldT>(m, instance);
    const FieldT Z = (t ^ m) - instance.one();
    FieldT l = Z * FieldT(instance.params, m).inverse();
    _basic_domian_pAllocator->reset();

    FieldT s = omega ^ (blockDim.x * gridDim.x);

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    FieldT r = omega ^ tid;
    l *= r;

    while(tid < m)
    {
        u[tid] = l * (t - r).inverse();
        _basic_domian_pAllocator->reset();
        r *= s;
        l *= s;
        tid += blockDim.x * gridDim.x;
    }

    // while(tid < m)
    // {
    //     FieldT r = omega ^ tid;
    //     FieldT s = l * r;
    //     u[tid] = s * (t - r).inverse();
    //     tid += blockDim.x * gridDim.x;
    // }
}



template<typename FieldT>
__device__ libstl::vector<FieldT>& _p_basic_radix2_evaluate_all_lagrange_polynomials(const size_t m, const FieldT &t, size_t gridSize, size_t blockSize)
{
    libstl::vector<FieldT>* pu = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());
    libstl::vector<FieldT>& u = *pu;

    const FieldT& instance = t;
    if (m == 1)
    {
        u.resize(1, instance.one());
        return u;
    }

    const FieldT omega = libff::get_root_of_unity<FieldT>(m, instance);
    FieldT* zero = libstl::create<FieldT>(instance.zero());
    u.presize(m, *zero, gridSize, blockSize);

    if ((t ^ m) == (instance.one()))
    {
        FieldT omega_i = instance.one();
        for (size_t i = 0; i < m; ++i)
        {
            if (omega_i == t) // i.e., t equals omega^i
            {
                u[i] = instance.one();
                return u;
            }

            omega_i *= omega;
        }
    }

    _basic_domian_pAllocator = libstl::allocate(gridSize, blockSize, 5000);
    gmp_set_parallel_allocator_(_basic_domian_pAllocator);
    
    inverse_multi<<<gridSize, blockSize>>>(u, m, t);
    cudaDeviceSynchronize();

    gmp_set_serial_allocator_();

    return u;

}

template<typename FieldT>
__device__ libstl::vector<FieldT> _basic_radix2_evaluate_all_lagrange_polynomials(const size_t m, const FieldT &t) 
{
    if (m == 1)
        return libstl::vector<FieldT>(1, t.one());

    // assert(m == (1u << libff::log2(m)));

    const FieldT& instance = t;

    const FieldT omega = libff::get_root_of_unity<FieldT>(m, instance);

    libstl::vector<FieldT> u(m, instance.zero());

    /*
     If t equals one of the roots of unity in S={omega^{0},...,omega^{m-1}}
     then output 1 at the right place, and 0 elsewhere
     */

    if ((t ^ m) == (instance.one()))
    {
        FieldT omega_i = instance.one();
        for (size_t i = 0; i < m; ++i)
        {
            if (omega_i == t) // i.e., t equals omega^i
            {
                u[i] = instance.one();
                return u;
            }

            omega_i *= omega;
        }
    }

    /*
     Otherwise, if t does not equal any of the roots of unity in S,
     then compute each L_{i,S}(t) as Z_{S}(t) * v_i / (t-\omega^i)
     where:
     - Z_{S}(t) = \prod_{j} (t-\omega^j) = (t^m-1), and
     - v_{i} = 1 / \prod_{j \neq i} (\omega^i-\omega^j).
     Below we use the fact that v_{0} = 1/m and v_{i+1} = \omega * v_{i}.
     */

    // libstl::vector<FieldT>* uu = new libstl::vector<FieldT>;
    // uu->resize(m, instance.zero());
    // inverse_multi<<<1, 32>>>(uu, m, t);

    const FieldT Z = (t ^ m) - instance.one();
    FieldT l = Z * FieldT(instance.params, m).inverse();
    FieldT r = instance.one();

    for (size_t i = 0; i < m; ++i)
    {
        u[i] = l * (t - r).inverse();
        l *= omega;
        r *= omega;
    }

    return u;
}


}

#endif