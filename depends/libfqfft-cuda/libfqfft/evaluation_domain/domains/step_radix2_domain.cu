#ifndef __STEP_RADIX2_DOMAIN_CU__
#define __STEP_RADIX2_DOMAIN_CU__

#include <assert.h>
#include "../../../depends/libff-cuda/fields/field_utils.cuh"
#include "../../../depends/libff-cuda/common/utils.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ step_radix2_domain<FieldT>::step_radix2_domain(const size_t m, FieldT* instance): m(m), instance(instance)
{
    assert(m > 1);

    big_m = 1ul << (libff::log2(m) - 1);
    small_m = m - big_m;

    assert(small_m == 1ul << libff::log2(small_m));

    omega = libff::get_root_of_unity<FieldT>(1ul << libff::log2(m), *this->instance);
    
    big_omega = omega.squared();
    small_omega = libff::get_root_of_unity<FieldT>(small_m, *this->instance);
}

template<typename FieldT>
__host__ __device__ bool step_radix2_domain<FieldT>::valid(size_t m)
{
    if(m <= 1) return false;
    size_t big_m = 1ul << (libff::log2(m) - 1);
    size_t small_m = m - big_m;
    if(small_m != 1ul << libff::log2(small_m)) return false;

    return true;
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::FFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT> c(big_m, this->instance->zero());
    libstl::vector<FieldT> d(big_m, this->instance->zero());

    FieldT omega_i = this->instance->one();
    for (size_t i = 0; i < big_m; ++i)
    {
        c[i] = (i < small_m ? a[i] + a[i + big_m] : a[i]);
        d[i] = omega_i * (i < small_m ? a[i] - a[i + big_m] : a[i]);
        omega_i *= omega;
    }

    libstl::vector<FieldT> e(small_m, this->instance->zero());
    
    const size_t compr = 1ul << (libff::log2(big_m) - libff::log2(small_m));
    for (size_t i = 0; i < small_m; ++i)
        for (size_t j = 0; j < compr; ++j)
            e[i] += d[i + j * small_m];
        

    _basic_radix2_FFT(c, big_omega);
    _basic_radix2_FFT(e, small_omega);

    for (size_t i = 0; i < big_m; ++i)
        a[i] = c[i];

    for (size_t i = 0; i < small_m; ++i)
        a[i + big_m] = e[i];
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::pFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT>* c = libstl::create<libstl::vector<FieldT>>(big_m, this->instance->zero());
    libstl::vector<FieldT>* e = libstl::create<libstl::vector<FieldT>>(small_m, this->instance->zero());

    size_t num = gridSize * blockSize;
    FieldT* omega_exp_num = libstl::create<FieldT>((*omega) ^ num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_idx = (*omega) ^ idx;

            for (size_t i = idx; i < big_m; i += num)
            {
                (*c)[i] = (i < small_m ? a[i] + a[i + big_m] : a[i]);
                (*e)[i % small_m] += omega_idx * (i < small_m ? a[i] - a[i + big_m] : a[i]);
                omega_idx *= *omega_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();
        
    _p_basic_radix2_FFT(c, big_omega, gridSize, blockSize);
    _p_basic_radix2_FFT(e, small_omega, gridSize, blockSize);

    for (size_t i = 0; i < big_m; ++i)
        a[i] = (*c)[i];

    for (size_t i = 0; i < small_m; ++i)
        a[i + big_m] = (*e)[i];
}


template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::ppFFT(libstl::vector<FieldT>& a, size_t num)
{
    // assert(a.size() == this->m);

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    // libstl::vector<FieldT>* c = libstl::create<libstl::vector<FieldT>>(big_m, this->instance->zero());
    // libstl::vector<FieldT>* e = libstl::create<libstl::vector<FieldT>>(small_m, this->instance->zero());
    libstl::vector<FieldT>* c = libstl::create<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* e = libstl::create<libstl::vector<FieldT>>();
    c->presize(big_m, this->instance->zero(), gridSize, blockSize);
    e->presize(small_m, this->instance->zero(), gridSize, blockSize);

    FieldT* omega_exp_num = libstl::create<FieldT>((*omega) ^ num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_idx = (*omega) ^ idx;

            for (size_t i = idx; i < big_m; i += num)
            {
                (*c)[i] = (i < small_m ? a[i] + a[i + big_m] : a[i]);
                (*e)[i % small_m] += omega_idx * (i < small_m ? a[i] - a[i + big_m] : a[i]);
                omega_idx *= *omega_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();
        
    _pp_basic_radix2_FFT(c, big_omega, num);
    _pp_basic_radix2_FFT(e, small_omega, num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                a[i] = (*c)[i];
            }
        }
    );
    cudaDeviceSynchronize();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                a[i + big_m] = (*e)[i];
            }
        }
    );
    cudaDeviceSynchronize();
}


template<typename FieldT>
__host__ void step_radix2_domain<FieldT>::ppFFT_host(libstl::vector<FieldT>& a, size_t num)
{
    // assert(a.size() == this->m);

    size_t blockSize = min(num, (size_t)32);
    size_t gridSize = num / blockSize;

    // libstl::vector<FieldT>* c = libstl::create<libstl::vector<FieldT>>(big_m, this->instance->zero());
    // libstl::vector<FieldT>* e = libstl::create<libstl::vector<FieldT>>(small_m, this->instance->zero());
    libstl::vector<FieldT>* c = libstl::create_host<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* e = libstl::create_host<libstl::vector<FieldT>>();

    size_t hm = a.size_host();
    size_t hbig_m = 1ul << (libff::log2(hm) - 1);
    size_t hsmall_m = hm - hbig_m;



    FieldT* instance_addr;
    cudaMemcpy(&instance_addr, &instance, sizeof(FieldT *), cudaMemcpyDeviceToHost);
    FieldT* zero = instance_addr->zero_host();

    c->presize_host(hbig_m, zero, gridSize, blockSize);
    e->presize_host(hsmall_m, zero, gridSize, blockSize);

    // // FieldT* omega_exp_num = libstl::create_host<FieldT>(omega ^ num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_idx = omega ^ idx;
            FieldT omega_exp_num = omega ^ num;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*c)[i] = (i < small_m ? a[i] + a[i + big_m] : a[i]);
                (*e)[i % small_m] += omega_idx * (i < small_m ? a[i] - a[i + big_m] : a[i]);
                omega_idx *= omega_exp_num;
            }
        }, a
    );
    cudaStreamSynchronize(0);
        
    _pp_basic_radix2_FFT_host(*c, big_omega, num);
    _pp_basic_radix2_FFT_host(*e, small_omega, num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                a[i] = (*c)[i];
            }
        }, a
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                a[i + big_m] = (*e)[i];
            }
        }, a
    );
    cudaStreamSynchronize(0);
}


template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::iFFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT> U0(a.begin(), a.begin() + big_m);
    libstl::vector<FieldT> U1(a.begin() + big_m, a.end());

    _basic_radix2_FFT(U0, big_omega->inverse());
    _basic_radix2_FFT(U1, small_omega->inverse());

    const FieldT U0_size_inv = FieldT(this->instance->params, big_m).inverse();
    for (size_t i = 0; i < big_m; ++i)
        U0[i] *= U0_size_inv;

    const FieldT U1_size_inv = FieldT(this->instance->params, small_m).inverse();
    for (size_t i = 0; i < small_m; ++i)
        U1[i] *= U1_size_inv;

    libstl::vector<FieldT> tmp = U0;
    FieldT omega_i = this->instance->one();
    for (size_t i = 0; i < big_m; ++i)
    {
        tmp[i] *= omega_i;
        omega_i *= omega;
    }

    // save A_suffix
    for (size_t i = small_m; i < big_m; ++i)
        a[i] = U0[i];

    const size_t compr = 1ul << (libff::log2(big_m) - libff::log2(small_m));
    for (size_t i = 0; i < small_m; ++i)
        for (size_t j = 1; j < compr; ++j)
            U1[i] -= tmp[i + j * small_m];

    const FieldT omega_inv = omega->inverse();
    FieldT omega_inv_i = this->instance->one();
    for (size_t i = 0; i < small_m; ++i)
    {
        U1[i] *= omega_inv_i;
        omega_inv_i *= omega_inv;
    }

    // compute A_prefix
    const FieldT over_two = FieldT(this->instance->params, 2).inverse();
    for (size_t i = 0; i < small_m; ++i)
        a[i] = (U0[i] + U1[i]) * over_two;

    // compute B2
    for (size_t i = 0; i < small_m; ++i)
        a[big_m + i] = (U0[i] - U1[i]) * over_two;
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::piFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT>* U0 = libstl::create<libstl::vector<FieldT>>(a.begin(), a.begin() + big_m);
    libstl::vector<FieldT>* U1 = libstl::create<libstl::vector<FieldT>>(a.begin() + big_m, a.end());

    FieldT* big_omega_inv = libstl::create<FieldT>(big_omega->inverse());
    FieldT* small_omega_inv = libstl::create<FieldT>(small_omega->inverse());

    _p_basic_radix2_FFT(U0, *big_omega_inv, gridSize, blockSize);
    _p_basic_radix2_FFT(U1, *small_omega_inv, gridSize, blockSize);

    const FieldT U0_size_inv = FieldT(this->instance->params, big_m).inverse();
    for (size_t i = 0; i < big_m; ++i)
        (*U0)[i] *= U0_size_inv;

    const FieldT U1_size_inv = FieldT(this->instance->params, small_m).inverse();
    for (size_t i = 0; i < small_m; ++i)
        (*U1)[i] *= U1_size_inv;

    libstl::vector<FieldT> tmp = U0;
    FieldT omega_i = this->instance->one();
    for (size_t i = 0; i < big_m; ++i)
    {
        tmp[i] *= omega_i;
        omega_i *= *omega;
    }

    // save A_suffix
    for (size_t i = small_m; i < big_m; ++i)
        a[i] = (*U0)[i];

    const size_t compr = 1ul << (libff::log2(big_m) - libff::log2(small_m));
    for (size_t i = 0; i < small_m; ++i)
        for (size_t j = 1; j < compr; ++j)
            (*U1)[i] -= tmp[i + j * small_m];

    const FieldT omega_inv = omega->inverse();
    FieldT omega_inv_i = this->instance->one();
    for (size_t i = 0; i < small_m; ++i)
    {
        (*U1)[i] *= omega_inv_i;
        omega_inv_i *= omega_inv;
    }

    // compute A_prefix
    const FieldT over_two = FieldT(this->instance->params, 2).inverse();
    for (size_t i = 0; i < small_m; ++i)
        a[i] = ((*U0)[i] + (*U1)[i]) * over_two;

    // compute B2
    for (size_t i = 0; i < small_m; ++i)
        a[big_m + i] = ((*U0)[i] - (*U1)[i]) * over_two;    
}


template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::ppiFFT(libstl::vector<FieldT>& a, size_t num)
{
    // assert(a.size() == this->m);

    // libstl::vector<FieldT>* U0 = libstl::create<libstl::vector<FieldT>>(a.begin(), a.begin() + big_m);
    // libstl::vector<FieldT>* U1 = libstl::create<libstl::vector<FieldT>>(a.begin() + big_m, a.end());

    size_t blockSize = min(num, (size_t)32);
    size_t gridSize = num / blockSize;

    libstl::vector<FieldT>* U0 = libstl::create<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* U1 = libstl::create<libstl::vector<FieldT>>();
    U0->presize(big_m, gridSize, blockSize);
    U1->presize(small_m, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*U0)[i] = a[i];
            }
        }
    );
    cudaDeviceSynchronize();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                (*U1)[i] = a[i+big_m];
            }
        }
    );
    cudaDeviceSynchronize();

    FieldT* big_omega_inv = libstl::create<FieldT>(big_omega->inverse());
    FieldT* small_omega_inv = libstl::create<FieldT>(small_omega->inverse());

    _pp_basic_radix2_FFT(U0, *big_omega_inv, num);
    _pp_basic_radix2_FFT(U1, *small_omega_inv, num);

    FieldT* U0_size_inv = libstl::create<FieldT>(FieldT(this->instance->params, big_m).inverse());

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*U0)[i] *= *U0_size_inv;
            }
        }
    );
    cudaDeviceSynchronize();

    // for (size_t i = 0; i < big_m; ++i)
    //     (*U0)[i] *= U0_size_inv;

    FieldT* U1_size_inv = libstl::create<FieldT>(FieldT(this->instance->params, small_m).inverse());

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                (*U1)[i] *= *U1_size_inv;
            }
        }
    );
    cudaDeviceSynchronize();

    // for (size_t i = 0; i < small_m; ++i)
    //     (*U1)[i] *= U1_size_inv;

    libstl::vector<FieldT>* tmp = libstl::create<libstl::vector<FieldT>>();
    tmp->presize(big_m, gridSize, blockSize);
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*tmp)[i] = (*U0)[i];
            }
        }
    );
    cudaDeviceSynchronize();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_i = *omega ^ idx;
            FieldT omega_exp_num = *omega ^ num;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*tmp)[i] *= omega_i;
                omega_i *= omega_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();

    // libstl::vector<FieldT> tmp = U0;
    // FieldT omega_i = this->instance->one();
    // for (size_t i = 0; i < big_m; ++i)
    // {
    //     tmp[i] *= omega_i;
    //     omega_i *= *omega;
    // }

    // save A_suffix
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx + small_m; i < big_m; i += num)
            {
                a[i] = (*U0)[i];
            }
        }
    );
    cudaDeviceSynchronize();
    // for (size_t i = small_m; i < big_m; ++i)
    //     a[i] = (*U0)[i];

    const size_t compr = 1ul << (libff::log2(big_m) - libff::log2(small_m));
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                for (size_t j = 1; j < compr; ++j)
                {
                    (*U1)[i] -= (*tmp)[i + j * small_m];
                }
            }
        }
    );
    cudaDeviceSynchronize();
    // for (size_t i = 0; i < small_m; ++i)
    //     for (size_t j = 1; j < compr; ++j)
    //         (*U1)[i] -= tmp[i + j * small_m];


    FieldT* omega_inv = libstl::create<FieldT>(omega->inverse());
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_inv_i = *omega_inv ^ idx;
            FieldT omega_inv_exp_num = *omega_inv ^ num;
            for (size_t i = idx; i < small_m; i += num)
            {
                (*U1)[i] *= omega_inv_i;
                omega_inv_i *= omega_inv_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();

    // FieldT omega_inv_i = this->instance->one();
    // for (size_t i = 0; i < small_m; ++i)
    // {
    //     (*U1)[i] *= omega_inv_i;
    //     omega_inv_i *= omega_inv;
    // }

    // compute A_prefix
    FieldT* over_two = libstl::create<FieldT>(FieldT(this->instance->params, 2).inverse());
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                a[i] = ((*U0)[i] + (*U1)[i]) * (*over_two);
            }
        }
    );
    cudaDeviceSynchronize();
    // for (size_t i = 0; i < small_m; ++i)
    //     a[i] = ((*U0)[i] + (*U1)[i]) * over_two;

    // compute B2
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                a[big_m + i] = ((*U0)[i] - (*U1)[i]) * (*over_two);    
            }
        }
    );
    cudaDeviceSynchronize();
    // for (size_t i = 0; i < small_m; ++i)
    //     a[big_m + i] = ((*U0)[i] - (*U1)[i]) * over_two;    
}


template<typename FieldT>
__host__ void step_radix2_domain<FieldT>::ppiFFT_host(libstl::vector<FieldT>& a, size_t num)
{
    // assert(a.size() == this->m);

    // libstl::vector<FieldT>* U0 = libstl::create<libstl::vector<FieldT>>(a.begin(), a.begin() + big_m);
    // libstl::vector<FieldT>* U1 = libstl::create<libstl::vector<FieldT>>(a.begin() + big_m, a.end());

    size_t blockSize = min(num, (size_t)32);
    size_t gridSize = num / blockSize;

    libstl::vector<FieldT>* U0 = libstl::create_host<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* U1 = libstl::create_host<libstl::vector<FieldT>>();

    size_t hm = a.size_host();
    size_t hbig_m = 1ul << (libff::log2(hm) - 1);
    size_t hsmall_m = hm - hbig_m;

    U0->presize_host(hbig_m, gridSize, blockSize);
    U1->presize_host(hsmall_m, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*U0)[i] = a[i];
            }
        }, a
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                (*U1)[i] = a[i + big_m];
            }
        }, a
    );
    cudaStreamSynchronize(0);

    FieldT* big_omega_inv = big_omega.inverse_host();
    FieldT* small_omega_inv = small_omega.inverse_host();

    _pp_basic_radix2_FFT_host(*U0, *big_omega_inv, num);
    _pp_basic_radix2_FFT_host(*U1, *small_omega_inv, num);

    size_t* dbig_m = libstl::create_host<size_t>();
    cudaMemcpy(dbig_m, &hbig_m, sizeof(size_t), cudaMemcpyHostToDevice);

    FieldT* instance_addr;
    cudaMemcpy(&instance_addr, &instance, sizeof(FieldT *), cudaMemcpyDeviceToHost);
    FieldT* U0_size = libstl::create_host<FieldT>(instance_addr->params, *dbig_m);
    FieldT* U0_size_inv = U0_size->inverse_host();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*U0)[i] *= *U0_size_inv;
            }
        }
    );
    cudaStreamSynchronize(0);

    size_t* dsmall_m = libstl::create_host<size_t>();
    cudaMemcpy(dsmall_m, &hsmall_m, sizeof(size_t), cudaMemcpyHostToDevice);
    FieldT* U1_size = libstl::create_host<FieldT>(instance_addr->params, *dsmall_m);
    FieldT* U1_size_inv = U1_size->inverse_host();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                (*U1)[i] *= *U1_size_inv;
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::vector<FieldT>* tmp = libstl::create_host<libstl::vector<FieldT>>();
    tmp->presize_host(hbig_m, gridSize, blockSize);
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*tmp)[i] = (*U0)[i];
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_i = omega ^ idx;
            FieldT omega_exp_num = omega ^ num;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*tmp)[i] *= omega_i;
                omega_i *= omega_exp_num;
            }
        }
    );
    cudaStreamSynchronize(0);

    // save A_suffix
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx + small_m; i < big_m; i += num)
            {
                a[i] = (*U0)[i];
            }
        }, a
    );
    cudaStreamSynchronize(0);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t compr = 1ul << (libff::log2(big_m) - libff::log2(small_m));
            for (size_t i = idx; i < small_m; i += num)
            {
                for (size_t j = 1; j < compr; ++j)
                {
                    (*U1)[i] -= (*tmp)[i + j * small_m];
                }
            }
        }, a
    );
    cudaStreamSynchronize(0);

    FieldT* omega_inv = omega.inverse_host();
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT omega_inv_i = *omega_inv ^ idx;
            FieldT omega_inv_exp_num = *omega_inv ^ num;
            for (size_t i = idx; i < small_m; i += num)
            {
                (*U1)[i] *= omega_inv_i;
                omega_inv_i *= omega_inv_exp_num;
            }
        }
    );
    cudaStreamSynchronize(0);

    // compute A_prefix
    // FieldT* over_two = libstl::create<FieldT>(FieldT(this->instance->params, 2).inverse());

    size_t htwo = 2;    
    size_t* dtwo = libstl::create_host<size_t>();
    cudaMemcpy(dtwo, &htwo, sizeof(size_t), cudaMemcpyHostToDevice);

    FieldT* over_two = libstl::create_host<FieldT>(instance_addr->params, *dtwo);
    FieldT* over_two_inv = over_two->inverse_host();

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                a[i] = ((*U0)[i] + (*U1)[i]) * (*over_two_inv);
            }
        }, a
    );
    cudaStreamSynchronize(0);

    // compute B2
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                a[big_m + i] = ((*U0)[i] - (*U1)[i]) * (*over_two_inv);    
            }
        }, a
    );
    cudaStreamSynchronize(0);
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::cosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    _multiply_by_coset(a, g);
    FFT(a);
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::pcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    _p_multiply_by_coset(a, g, gridSize, blockSize);
    pFFT(a, gridSize, blockSize);
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::ppcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    _pp_multiply_by_coset(a, g, num);
    ppFFT(a, num);
}

template<typename FieldT>
__host__ void step_radix2_domain<FieldT>::ppcosetFFT_host(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    _pp_multiply_by_coset_host(a, g, num);
    ppFFT_host(a, num);
}


template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::icosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    iFFT(a);
    _multiply_by_coset(a, g.inverse());
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::picosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    piFFT(a, gridSize, blockSize);

    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _p_multiply_by_coset(a, *g_inv, gridSize, blockSize);
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::ppicosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    ppiFFT(a, num);

    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _pp_multiply_by_coset(a, *g_inv, num);
}

template<typename FieldT>
__host__ void step_radix2_domain<FieldT>::ppicosetFFT_host(libstl::vector<FieldT>& a, FieldT& g, size_t num)
{
    ppiFFT_host(a, num);

    FieldT* g_inv = g.inverse_host();
    _pp_multiply_by_coset_host(a, *g_inv, num);
}

static __device__ libstl::ParallalAllocator* _step_domian_pAllocator;

template<typename FieldT>
__device__ libstl::vector<FieldT> step_radix2_domain<FieldT>::p_evaluate_all_lagrange_polynomials(const FieldT &t, size_t gridSize, size_t blockSize)
{
    libstl::vector<FieldT>& inner_big = _p_basic_radix2_evaluate_all_lagrange_polynomials(big_m, t, gridSize, blockSize);
    FieldT* t_omega_inverse = libstl::create<FieldT>();
    *t_omega_inverse = t * omega.inverse();
    libstl::vector<FieldT>& inner_small = _p_basic_radix2_evaluate_all_lagrange_polynomials(small_m, *t_omega_inverse, gridSize, blockSize);

    FieldT* zero = libstl::create<FieldT>(this->instance->zero());
    libstl::vector<FieldT>* result = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());
    result->presize(this->m, *zero, gridSize, blockSize);

    FieldT* L0 = libstl::create<FieldT>();
    FieldT* omega_to_small_m = libstl::create<FieldT>();
    FieldT* big_omega_to_small_m_threads = libstl::create<FieldT>();
    *L0 = (t ^ small_m) - (omega ^ small_m);
    *omega_to_small_m = omega ^ small_m;
    *big_omega_to_small_m_threads = big_omega ^ (small_m * gridSize * blockSize);

    // const FieldT L0 = (t ^ small_m) - (*omega ^ small_m);
    // const FieldT omega_to_small_m = *omega ^ small_m;
    // const FieldT big_omega_to_small_m = *big_omega ^ small_m;
    // FieldT elt = this->instance->one();

    _step_domian_pAllocator = libstl::allocate(gridSize, blockSize, 5000);
    gmp_set_parallel_allocator_(_step_domian_pAllocator);

    // for (size_t i = 0; i < big_m; ++i)
    // {
    //     result[i] = inner_big[i] * L0 * (elt - omega_to_small_m).inverse();
    //     elt *= big_omega_to_small_m;
    // }

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &inner_big]
        __device__ () 
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT elt = big_omega ^ (small_m * idx);
            while(idx < big_m)
            {
                (*result)[idx] = inner_big[idx] * (*L0) * (elt - (*omega_to_small_m)).inverse();
                _step_domian_pAllocator->reset();
                elt *= (*big_omega_to_small_m_threads);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    gmp_set_serial_allocator_();

    FieldT* L1 = libstl::create<FieldT>();
    *L1 = ((t ^ big_m) - this->instance->one()) * ((omega ^ big_m) - this->instance->one()).inverse();

    // for (size_t i = 0; i < small_m; ++i)
    //     result[big_m + i] = L1 * inner_small[i];

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &inner_small]
        __device__ () 
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < small_m)
            {
                (*result)[big_m + idx] = (*L1) * inner_small[idx];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return *result;
}

template<typename FieldT>
__device__ libstl::vector<FieldT> step_radix2_domain<FieldT>::evaluate_all_lagrange_polynomials(const FieldT &t)
{
    libstl::vector<FieldT> inner_big = _basic_radix2_evaluate_all_lagrange_polynomials(big_m, t);
    libstl::vector<FieldT> inner_small = _basic_radix2_evaluate_all_lagrange_polynomials(small_m, t * omega->inverse());

    libstl::vector<FieldT> result(this->m, this->instance->zero());

    const FieldT L0 = (t ^ small_m) - (omega ^ small_m);
    const FieldT omega_to_small_m = omega ^ small_m;
    const FieldT big_omega_to_small_m = big_omega ^ small_m;
    FieldT elt = this->instance->one();

    for (size_t i = 0; i < big_m; ++i)
    {
        result[i] = inner_big[i] * L0 * (elt - omega_to_small_m).inverse();
        elt *= big_omega_to_small_m;
    }

    const FieldT L1 = ((t ^ big_m) - this->instance->one()) * ((omega ^ big_m) - this->instance->one()).inverse();

    for (size_t i = 0; i < small_m; ++i)
        result[big_m + i] = L1 * inner_small[i];

    return result;
}

template<typename FieldT>
__device__ FieldT step_radix2_domain<FieldT>::get_domain_element(const size_t idx)
{
    if (idx < big_m)
        return big_omega ^ idx;
    else
        return omega * (small_omega ^ (idx - big_m));
}

template<typename FieldT>
__device__ FieldT step_radix2_domain<FieldT>::compute_vanishing_polynomial(const FieldT &t)
{
    return ((t ^ big_m) - this->instance->one()) * ((t ^ small_m) - (omega ^ small_m));
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H)
{
    assert(H.size() == this->m + 1);

    const FieldT omega_to_small_m = omega ^ small_m;

    H[this->m] += coeff;
    H[big_m] -= coeff * omega_to_small_m;
    H[small_m] -= coeff;
    H[0] += coeff * omega_to_small_m;
}

template<typename FieldT>
__device__ void step_radix2_domain<FieldT>::divide_by_Z_on_coset(libstl::vector<FieldT>& P)
{
    // (c^{2^k}-1) * (c^{2^r} * w^{2^{r+1}*i) - w^{2^r})
    const FieldT coset = FieldT(this->instance->params, *this->instance->params->multiplicative_generator);

    const FieldT Z0 = (coset ^ big_m) - this->instance->one();
    const FieldT coset_to_small_m_times_Z0 = (coset ^ small_m) * Z0;
    const FieldT omega_to_small_m_times_Z0 = (omega ^ small_m) * Z0;
    const FieldT omega_to_2small_m = omega ^ (2 * small_m);
    FieldT elt = this->instance->one();

    for (size_t i = 0; i < big_m; ++i)
    {
        P[i] *= (coset_to_small_m_times_Z0 * elt - omega_to_small_m_times_Z0).inverse();
        elt *= omega_to_2small_m;
    }

    // (c^{2^k}*w^{2^k}-1) * (c^{2^k} * w^{2^r} - w^{2^r})

    const FieldT Z1 = ((((coset * omega) ^ big_m) - this->instance->one()) * (((coset * omega) ^ small_m) - (omega ^ small_m)));
    const FieldT Z1_inverse = Z1.inverse();

    for (size_t i = 0; i < small_m; ++i)
        P[big_m + i] *= Z1_inverse;
}


template<typename FieldT>
__device__ __inline__ static void p_batch_to_special_invert(libstl::vector<FieldT> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const FieldT& instance)
{
    libstl::vector<FieldT> prod;
    prod.set_parallel_allocator(_step_domian_pAllocator);
    prod.resize(zero_idx.size());

    FieldT acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx];
    }

    FieldT acc_inverse = acc.inverse();

    for (size_t i = non_zero_length - 1; i <= non_zero_length; --i)
    { 
        size_t idx = zero_idx[i];
        const FieldT old_el = vec[idx];
        vec[idx] = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}


template<typename FieldT>
__host__ void step_radix2_domain<FieldT>::p_divide_by_Z_on_coset_host(libstl::vector<FieldT>& P, size_t gridSize, size_t blockSize)
{
    FieldT* coset = libstl::create_host<FieldT>();

    FieldT* Z0 = libstl::create_host<FieldT>();
    FieldT* coset_to_small_m_times_Z0 = libstl::create_host<FieldT>();
    FieldT* omega_to_small_m_times_Z0 = libstl::create_host<FieldT>();
    FieldT* omega_to_2small_m = libstl::create_host<FieldT>();

    FieldT* Z1 = libstl::create_host<FieldT>(); 
    FieldT* Z1_inverse = libstl::create_host<FieldT>();

    libstl::launch<<<1, 1>>>
    (
        [=] 
        __device__ ()
        {
            *coset = FieldT(this->instance->params, *this->instance->params->multiplicative_generator);
            *Z0 = (*coset ^ big_m) - instance->one();
            *coset_to_small_m_times_Z0 = (*coset ^ small_m) * (*Z0);
            *omega_to_small_m_times_Z0 = (omega ^ small_m) * (*Z0);
            *omega_to_2small_m = omega ^ (2 * small_m);
            *Z1 = ((((*coset * omega) ^ big_m) - instance->one()) * (((*coset * omega) ^ small_m) - (omega ^ small_m)));
            *Z1_inverse = Z1->inverse();
        }
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<1, 1>>>
    (
        [=] 
        __device__ ()
        {
            size_t tnum = gridSize * blockSize;
            size_t alloc_size = (big_m + tnum - 1) / tnum * (sizeof(size_t) + sizeof(FieldT));
            _step_domian_pAllocator = libstl::allocate(gridSize, blockSize, 2124 + alloc_size);
            gmp_set_parallel_allocator_(_step_domian_pAllocator);
        }
    );
    cudaStreamSynchronize(0);

    size_t hm = P.size_host();
    size_t hbig_m = 1ul << (libff::log2(hm) - 1);
    // size_t hsmall_m = hm - hbig_m;

    libstl::vector<FieldT>* tmp = libstl::create_host<libstl::vector<FieldT>>();
    tmp->presize_host(hbig_m, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t num = gridDim.x * blockDim.x;
            FieldT elt_i = *omega_to_2small_m ^ idx;
            FieldT omega_to_2small_m_exp_num = *omega_to_2small_m ^ num;
            for (size_t i = idx; i < big_m; i += num)
            {
                (*tmp)[i] = *coset_to_small_m_times_Z0 * elt_i - *omega_to_small_m_times_Z0;
                elt_i *= omega_to_2small_m_exp_num;
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = gridDim.x * blockDim.x;
            size_t total = tmp->size();
            size_t range_s = (total + tnum - 1) / tnum * tid;
            size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
            libstl::vector<size_t> zero_idx;
            zero_idx.set_parallel_allocator(_step_domian_pAllocator);
            zero_idx.resize(range_e - range_s);
            size_t zero_length = range_e - range_s - 1;
            size_t non_zero_length = 0;

            FieldT zero = instance->zero();
            for(size_t i=range_s; i < range_e && i < total; i++)
            {
                if((*tmp)[i] == zero)
                {
                    zero_idx[zero_length--] = i;
                }
                else
                {
                    zero_idx[non_zero_length++] = i;
                }
            }

            p_batch_to_special_invert(*tmp, zero_idx, non_zero_length, *instance);
        }
    );

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& P)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t num = gridDim.x * blockDim.x;
            FieldT elt_i = *omega_to_2small_m ^ idx;
            FieldT omega_to_2small_m_exp_num = *omega_to_2small_m ^ num;
            for (size_t i = idx; i < big_m; i += num)
            {
                P[i] *= (*tmp)[i];
                // _step_domian_pAllocator->reset();
                // elt_i *= omega_to_2small_m_exp_num;
            }
        }, P
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<1, 1>>>
    (
        [=] 
        __device__ ()
        {
            gmp_set_serial_allocator_();
        }
    );
    cudaStreamSynchronize(0);
    // // (c^{2^k}*w^{2^k}-1) * (c^{2^k} * w^{2^r} - w^{2^r})

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& P)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t num = gridDim.x * blockDim.x;
            for (size_t i = idx; i < small_m; i += num)
            {
                P[big_m + i] *= *Z1_inverse;
            }
        }, P
    );
    cudaStreamSynchronize(0);
}

}

#endif
