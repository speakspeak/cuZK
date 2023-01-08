#ifndef __EXTENDED_RADIX2_DOMAIN_CU__
#define __EXTENDED_RADIX2_DOMAIN_CU__

#include <assert.h>
#include "../../../depends/libff-cuda/fields/field_utils.cuh"
#include "../../../depends/libff-cuda/common/utils.cuh"
#include "../../../depends/libff-cuda/fields/bigint.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ extended_radix2_domain<FieldT>::extended_radix2_domain(const size_t m, FieldT* instance) : m(m), instance(instance)
{
    assert(m > 1);

    const size_t logm = libff::log2(m);
    assert(logm <= this->instance->params->s);

    small_m = m / 2;

    omega = (FieldT*)libstl::create<FieldT>(libff::get_root_of_unity<FieldT>(small_m, *this->instance));
    shift = (FieldT*)libstl::create<FieldT>(libff::coset_shift<FieldT>(*this->instance));
}


template<typename FieldT>
__host__ __device__ bool extended_radix2_domain<FieldT>::valid(const size_t m)
{
    if(m <= 1) return false;
    const size_t logm = libff::log2(m);
    // if(logm > instance->params->s) return false;
    size_t n = m / 2;
    size_t logn = libff::log2(n);
    if(n != (1u << logn)) return false;

    return true;
}


template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::FFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT> a0(small_m, this->instance->zero());
    libstl::vector<FieldT> a1(small_m, this->instance->zero());

    const FieldT shift_to_small_m = (*shift) ^ libff::bigint<1>(small_m);

    FieldT shift_i = this->instance->one();
    for (size_t i = 0; i < small_m; ++i)
    {
        a0[i] = a[i] + a[small_m + i];
        a1[i] = shift_i * (a[i] + shift_to_small_m * a[small_m + i]);

        shift_i *= *shift;
    }

    _basic_radix2_FFT(a0, *omega);
    _basic_radix2_FFT(a1, *omega);

    for (size_t i = 0; i < small_m; ++i)
    {
        a[i] = a0[i];
        a[i + small_m] = a1[i];
    }
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::pFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT>* a0 = libstl::create<libstl::vector<FieldT>>(small_m, this->instance->zero());
    libstl::vector<FieldT>* a1 = libstl::create<libstl::vector<FieldT>>(small_m, this->instance->zero());

    size_t num = gridSize * blockSize;

    FieldT* shift_to_small_m = libstl::create<FieldT>((*shift) ^ libff::bigint<1>(small_m));
    FieldT* shift_exp_num = libstl::create<FieldT>((*shift) ^ num);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT shift_i = (*shift) ^ idx;

            for (size_t i = idx; i < small_m; i += num)
            {
                (*a0)[i] = a[i] + a[small_m + i];
                (*a1)[i] = shift_i * (a[i] + *shift_to_small_m * a[small_m + i]);

                shift_i *= *shift_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();

    _p_basic_radix2_FFT(*a0, *omega, gridSize, blockSize);
    _p_basic_radix2_FFT(*a1, *omega, gridSize, blockSize);

    for (size_t i = 0; i < small_m; ++i)
    {
        a[i] = (*a0)[i];
        a[i + small_m] = (*a1)[i];
    }
}


template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::ppFFT(libstl::vector<FieldT>& a, size_t num)
{
    libstl::vector<FieldT>* a0 = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());
    libstl::vector<FieldT>* a1 = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    FieldT* zero = libstl::create<FieldT>(this->instance->zero());
    a0->presize(small_m, *zero, gridSize, blockSize);
    a1->presize(small_m, *zero, gridSize, blockSize);

    FieldT* shift_to_small_m = libstl::create<FieldT>((*shift) ^ libff::bigint<1>(small_m));
    FieldT* shift_exp_num = libstl::create<FieldT>((*shift) ^ num);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT shift_i = (*shift) ^ idx;

            for (size_t i = idx; i < small_m; i += num)
            {
                (*a0)[i] = a[i] + a[small_m + i];
                (*a1)[i] = shift_i * (a[i] + *shift_to_small_m * a[small_m + i]);

                shift_i *= *shift_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();

    _pp_basic_radix2_FFT(*a0, *omega, num);
    _pp_basic_radix2_FFT(*a1, *omega, num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < small_m)
            {
                a[idx] = (*a0)[idx];
                a[idx + small_m] = (*a1)[idx];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    // for (size_t i = 0; i < small_m; ++i)
    // {
    //     a[i] = (*a0)[i];
    //     a[i + small_m] = (*a1)[i];
    // }
}



template<typename FieldT>
__host__ void extended_radix2_domain<FieldT>::ppFFT_host(libstl::vector<FieldT>& a, size_t num)
{
    libstl::vector<FieldT>* a0 = libstl::create_host<libstl::vector<FieldT>>(libstl::vector<FieldT>());
    libstl::vector<FieldT>* a1 = libstl::create_host<libstl::vector<FieldT>>(libstl::vector<FieldT>());

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    FieldT* zero = libstl::create_host<FieldT>(this->instance->zero());
    size_t dsmall_m;
    cudaMemcpy(&dsmall_m, &small_m, sizeof(size_t), cudaMemcpyDeviceToHost);
    a0->presize_host(dsmall_m, zero, gridSize, blockSize);
    a1->presize_host(dsmall_m, zero, gridSize, blockSize);

    // FieldT* shift_to_small_m = libstl::create_host<FieldT>((*shift) ^ libff::bigint<1>(small_m));
    // FieldT* shift_exp_num = libstl::create_host<FieldT>((*shift) ^ num);
    FieldT* shift_to_small_m = libstl::create_host<FieldT>();
    FieldT* shift_exp_num = libstl::create_host<FieldT>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *shift_to_small_m = (*shift) ^ libff::bigint<1>(small_m);
            *shift_exp_num = (*shift) ^ num;
        }
    );
    cudaStreamSynchronize(0);
    
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT shift_i = (*shift) ^ idx;

            for (size_t i = idx; i < small_m; i += num)
            {
                (*a0)[i] = a[i] + a[small_m + i];
                (*a1)[i] = shift_i * (a[i] + *shift_to_small_m * a[small_m + i]);

                shift_i *= *shift_exp_num;
            }
        }, a
    );
    cudaStreamSynchronize(0);

    _pp_basic_radix2_FFT_host(*a0, *omega, num);
    _pp_basic_radix2_FFT_host(*a1, *omega, num);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < small_m)
            {
                a[idx] = (*a0)[idx];
                a[idx + small_m] = (*a1)[idx];
                idx += gridDim.x * blockDim.x;
            }
        }, a
    );
    cudaStreamSynchronize(0);

    // for (size_t i = 0; i < small_m; ++i)
    // {
    //     a[i] = (*a0)[i];
    //     a[i + small_m] = (*a1)[i];
    // }
}



template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::iFFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    libstl::vector<FieldT> a0(a.begin(), a.begin() + small_m);
    libstl::vector<FieldT> a1(a.begin() + small_m, a.end());

    const FieldT omega_inv = omega->inverse();
    _basic_radix2_FFT(a0, omega_inv);
    _basic_radix2_FFT(a1, omega_inv);

    const FieldT shift_to_small_m = shift ^ libff::bigint<1>(small_m);
    const FieldT sconst = (FieldT(this->instance->params, small_m) * (this->instance->one() - shift_to_small_m)).inverse();

    const FieldT shift_inv = shift->inverse();
    FieldT shift_inv_i = this->instance->one();

    for (size_t i = 0; i < small_m; ++i)
    {
        a[i] = sconst * (-shift_to_small_m * a0[i] + shift_inv_i * a1[i]);
        a[i + small_m] = sconst * (a0[i] - shift_inv_i * a1[i]);

        shift_inv_i *= shift_inv;
    }
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::piFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    assert(a.size() == this->m);
 
    libstl::vector<FieldT>* a0 = libstl::create<libstl::vector<FieldT>>(a.begin(), a.begin() + small_m);
    libstl::vector<FieldT>* a1 = libstl::create<libstl::vector<FieldT>>(a.begin() + small_m, a.end());

    FieldT* omega_inv = libstl::create<FieldT>(omega->inverse());

    _p_basic_radix2_FFT(a0, *omega_inv, gridSize, blockSize);
    _p_basic_radix2_FFT(a1, *omega_inv, gridSize, blockSize);

    size_t num = gridSize * blockSize;

    FieldT* shift_to_small_m = libstl::create<FieldT>((*shift) ^ libff::bigint<1>(small_m));
    FieldT* sconst = libstl::create<FieldT>((FieldT(this->instance->params, small_m) * (this->instance->one() - *shift_to_small_m)).inverse());
    FieldT* shift_inv_exp_num = libstl::create<FieldT>(shift->inverse() ^ num);
    FieldT* shift_inv = libstl::create<FieldT>(shift->inverse() );


    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT shift_inv_i = (*shift_inv) ^ idx ;
            for (size_t i = 0; i < small_m; i += num)
            {
                a[i] = *sconst * (-*shift_to_small_m * a0[i] + *shift_inv_i * a1[i]);
                a[i + small_m] = *sconst * (a0[i] - *shift_inv_i * a1[i]);

                shift_inv_i *= shift_inv_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();

}


template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::ppiFFT(libstl::vector<FieldT>& a, size_t num)
{
    // libstl::vector<FieldT>* a0 = libstl::create<libstl::vector<FieldT>>(a.begin(), a.begin() + small_m);
    // libstl::vector<FieldT>* a1 = libstl::create<libstl::vector<FieldT>>(a.begin() + small_m, a.end());
    libstl::vector<FieldT>* a0 = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());
    libstl::vector<FieldT>* a1 = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    a0->presize(small_m, gridSize, blockSize);
    a1->presize(a.size() - small_m, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < small_m)
            {
                (*a0)[idx] = a[idx];
                idx += gridDim.x * blockDim.x;
            }
            while(idx < a.size() && idx >= small_m)
            {
                (*a1)[idx - small_m] = a[idx];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();


    FieldT* omega_inv = libstl::create<FieldT>(omega->inverse());

    _pp_basic_radix2_FFT(a0, *omega_inv, num);
    _pp_basic_radix2_FFT(a1, *omega_inv, num);

    FieldT* shift_to_small_m = libstl::create<FieldT>((*shift) ^ libff::bigint<1>(small_m));
    FieldT* sconst = libstl::create<FieldT>((FieldT(this->instance->params, small_m) * (this->instance->one() - *shift_to_small_m)).inverse());
    FieldT* shift_inv_exp_num = libstl::create<FieldT>(shift->inverse() ^ num);
    FieldT* shift_inv = libstl::create<FieldT>(shift->inverse() );

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT shift_inv_i = *shift_inv ^ idx;
            for (size_t i = 0; i < small_m; i += num)
            {
                a[i] = *sconst * (-*shift_to_small_m * a0[i] + *shift_inv_i * a1[i]);
                a[i + small_m] = *sconst * (a0[i] - *shift_inv_i * a1[i]);

                shift_inv_i *= shift_inv_exp_num;
            }
        }
    );
    cudaDeviceSynchronize();

}



template<typename FieldT>
__host__ void extended_radix2_domain<FieldT>::ppiFFT_host(libstl::vector<FieldT>& a, size_t num)
{
    printf("ifft begin\n");
    // libstl::vector<FieldT>* a0 = libstl::create_host<libstl::vector<FieldT>>(a.begin(), a.begin() + small_m);
    // libstl::vector<FieldT>* a1 = libstl::create_host<libstl::vector<FieldT>>(a.begin() + small_m, a.end());
    libstl::vector<FieldT>* a0 = libstl::create_host<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* a1 = libstl::create_host<libstl::vector<FieldT>>();

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    size_t dm = a.size_host();
    size_t dsmall_m = dm / 2;

    // printf("mem cpy begin\n");
    // size_t dsmall_m;
    // size_t dm;
    // cudaMemcpy(&dsmall_m, &this->small_m, sizeof(size_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&dm, &this->m, sizeof(size_t), cudaMemcpyDeviceToHost);

    printf("mem cpy ok\n");

    printf("resize begin\n");
    printf("dm : %lu\n", dm);
    printf("dsmall_m : %lu\n", dsmall_m);

    FieldT* zero = instance->zero_host();

    a0->presize_host(dsmall_m, zero, gridSize, blockSize);
    a1->presize_host(dm - dsmall_m, zero, gridSize, blockSize);
    printf("resize ok\n");

    // libstl::launch<<<gridSize, blockSize>>>
    // (
    //     [=]
    //     __device__ (libstl::vector<FieldT>& a)
    //     {
    //         size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //         while(idx < small_m)
    //         {
    //             (*a0)[idx] = a[idx];
    //             idx += gridDim.x * blockDim.x;
    //         }
    //     }, a
    // );
    // cudaStreamSynchronize(0);

    // printf("set ok\n");

    // libstl::launch<<<gridSize, blockSize>>>
    // (
    //     [=]
    //     __device__ (libstl::vector<FieldT>& a)
    //     {
    //         size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //         while(idx < m - small_m)
    //         {
    //             (*a1)[idx] = a[0];
    //             idx += gridDim.x * blockDim.x;
    //         }
    //     }, a
    // );
    // cudaStreamSynchronize(0);

    // printf("set ok 2\n");

    // FieldT* omega_inv = libstl::create_host<FieldT>(omega->inverse());
    FieldT* omega_inv = this->omega->inverse_host();

    _pp_basic_radix2_FFT_host(*a0, *omega_inv, num);
    _pp_basic_radix2_FFT_host(*a1, *omega_inv, num);

    printf("basic fft ok\n");

    // FieldT* shift_to_small_m = libstl::create_host<FieldT>((*shift) ^ libff::bigint<1>(small_m));
    // FieldT* sconst = libstl::create_host<FieldT>((FieldT(this->instance->params, small_m) * (this->instance->one() - *shift_to_small_m)).inverse());
    // FieldT* shift_inv_exp_num = libstl::create_host<FieldT>(shift->inverse() ^ num);
    // FieldT* shift_inv = libstl::create_host<FieldT>(shift->inverse() );
    FieldT* shift_to_small_m = libstl::create_host<FieldT>();
    FieldT* sconst = libstl::create_host<FieldT>();
    FieldT* shift_inv_exp_num = libstl::create_host<FieldT>();
    FieldT* shift_inv = libstl::create_host<FieldT>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *shift_to_small_m = (*shift) ^ libff::bigint<1>(small_m);
            *sconst = (FieldT(this->instance->params, small_m) * (this->instance->one() - *shift_to_small_m)).inverse();
            *shift_inv_exp_num = shift->inverse() ^ num;
            *shift_inv = shift->inverse();
        }
    );
    cudaStreamSynchronize(0);

    printf("Field ok\n");

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            FieldT shift_inv_i = (*shift_inv) ^ idx;
            for (size_t i = 0; i < small_m; i += num)
            {
                a[i] = *sconst * (-(*shift_to_small_m) * (*a0)[i] + shift_inv_i * (*a1)[i]);
                a[i + small_m] = (*sconst) * ((*a0)[i] - shift_inv_i * (*a1)[i]);

                shift_inv_i *= (*shift_inv_exp_num);
            }
        }, a
    );
    cudaStreamSynchronize(0);

}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::cosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    _multiply_by_coset(a, g);
    FFT(a);
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::pcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    _p_multiply_by_coset(a, g, gridSize, blockSize);
    pFFT(a, gridSize, blockSize);
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::ppcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    _pp_multiply_by_coset(a, g, num);
    ppFFT(a, num);
}

template<typename FieldT>
__host__ void extended_radix2_domain<FieldT>::ppcosetFFT_host(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    _pp_multiply_by_coset_host(a, g, num);
    ppFFT_host(a, num);
}


template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::icosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    iFFT(a);
    _multiply_by_coset(a, g.inverse());
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::picosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    piFFT(a, gridSize, blockSize);

    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _p_multiply_by_coset(a, *g_inv, gridSize, blockSize);
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::ppicosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    ppiFFT(a, num);
    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _pp_multiply_by_coset(a, *g_inv, num);
}


template<typename FieldT>
__host__ void extended_radix2_domain<FieldT>::ppicosetFFT_host(libstl::vector<FieldT>& a, FieldT& g, size_t num)
{
    ppiFFT_host(a, num);
    FieldT* g_inv = g.inverse_host();
    _pp_multiply_by_coset_host(a, *g_inv, num);
}


template<typename FieldT>
__device__ libstl::vector<FieldT> extended_radix2_domain<FieldT>::evaluate_all_lagrange_polynomials(const FieldT &t)
{
    const libstl::vector<FieldT> T0 = _basic_radix2_evaluate_all_lagrange_polynomials(small_m, t);
    const libstl::vector<FieldT> T1 = _basic_radix2_evaluate_all_lagrange_polynomials(small_m, t * shift->inverse());

    libstl::vector<FieldT> result(this->m, this->instance->zero());

    const FieldT t_to_small_m = t ^ libff::bigint<1>(small_m);
    const FieldT shift_to_small_m = (*shift) ^ libff::bigint<1>(small_m);
    const FieldT one_over_denom = (shift_to_small_m - this->instance->one()).inverse();
    const FieldT T0_coeff = (t_to_small_m - shift_to_small_m) * (-one_over_denom);
    const FieldT T1_coeff = (t_to_small_m - this->instance->one()) * one_over_denom;

    for (size_t i = 0; i < small_m; ++i)
    {
        result[i] = T0[i] * T0_coeff;
        result[i + small_m] = T1[i] * T1_coeff;
    }

    return result;
}


template<typename FieldT>
__device__ libstl::vector<FieldT> extended_radix2_domain<FieldT>::p_evaluate_all_lagrange_polynomials(const FieldT &t, size_t gridSize, size_t blockSize)
{
    libstl::vector<FieldT>& T0 = _p_basic_radix2_evaluate_all_lagrange_polynomials(small_m, t, gridSize, blockSize);
    FieldT* t_shift_inverse = libstl::create<FieldT>();
    *t_shift_inverse = t * shift->inverse();
    libstl::vector<FieldT>& T1 = _p_basic_radix2_evaluate_all_lagrange_polynomials(small_m, *t_shift_inverse, gridSize, blockSize);

    FieldT* zero = libstl::create<FieldT>(this->instance->zero());
    libstl::vector<FieldT>* result = libstl::create<libstl::vector<FieldT>>(libstl::vector<FieldT>());
    result->presize(this->m, *zero, gridSize, blockSize);

    const FieldT t_to_small_m = t ^ libff::bigint<1>(small_m);
    const FieldT shift_to_small_m = (*shift) ^ libff::bigint<1>(small_m);
    const FieldT one_over_denom = (shift_to_small_m - this->instance->one()).inverse();


    FieldT* T0_coeff = libstl::create<FieldT>();
    FieldT* T1_coeff = libstl::create<FieldT>();
    *T0_coeff = (t_to_small_m - shift_to_small_m) * (-one_over_denom);
    *T1_coeff = (t_to_small_m - this->instance->one()) * one_over_denom;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &T0, &T1]
        __device__ () 
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < small_m)
            {
                (*result)[idx] = T0[idx] * (*T0_coeff);
                (*result)[idx + small_m] = T1[idx] * (*T1_coeff);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return *result;
}

template<typename FieldT>
__device__ FieldT extended_radix2_domain<FieldT>::get_domain_element(const size_t idx)
{
    if (idx < small_m)
        return (*omega) ^ idx;
    else
        return (*shift) * ((*omega) ^ (idx - small_m));
}

template<typename FieldT>
__device__ FieldT extended_radix2_domain<FieldT>::compute_vanishing_polynomial(const FieldT &t)
{
    return ((t ^ small_m) - this->instance->one()) * ((t ^ small_m) - ((*shift) ^ small_m));
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H)
{
    assert(H.size() == this->m + 1);

    const FieldT shift_to_small_m = (*shift) ^ small_m;

    H[this->m] += coeff;
    H[small_m] -= coeff * (shift_to_small_m + this->instance->one());
    H[0] += coeff * shift_to_small_m;
}

template<typename FieldT>
__device__ void extended_radix2_domain<FieldT>::divide_by_Z_on_coset(libstl::vector<FieldT>& P)
{
    const FieldT coset = FieldT(this->instance->params, *this->instance->params->multiplicative_generator);

    const FieldT coset_to_small_m = coset ^ small_m;
    const FieldT shift_to_small_m = (*shift) ^ small_m;

    const FieldT Z0 = (coset_to_small_m - this->instance->one()) * (coset_to_small_m - shift_to_small_m);
    const FieldT Z1 = (coset_to_small_m * shift_to_small_m - this->instance->one()) * (coset_to_small_m * shift_to_small_m - shift_to_small_m);

    const FieldT Z0_inv = Z0.inverse();
    const FieldT Z1_inv = Z1.inverse();

    for (size_t i = 0; i < small_m; ++i)
    {
        P[i] *= Z0_inv;
        P[i + small_m] *= Z1_inv;
    }
}

}

#endif