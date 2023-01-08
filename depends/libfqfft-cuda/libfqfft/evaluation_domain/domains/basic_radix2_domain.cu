#ifndef __BASIC_RADIX2_DOMAIN_CU__
#define __BASIC_RADIX2_DOMAIN_CU__

#include <assert.h>
#include "../../../depends/libff-cuda/fields/field_utils.cuh"
#include "../../../depends/libff-cuda/common/utils.cuh"
#include "basic_radix2_domain_aux.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ basic_radix2_domain<FieldT>::basic_radix2_domain(const size_t m, FieldT* instance): m(m), instance(instance)
{
    assert(m > 1);
    const size_t logm = libff::log2(m);
    assert(logm <= this->instance->params->s);
    omega = libff::get_root_of_unity(m, *this->instance);
}

template<typename FieldT>
__host__ __device__ bool basic_radix2_domain<FieldT>::valid(const size_t m)
{
    if(m <= 1) return false;
    const size_t logm = libff::log2(m);
    // if(logm > instance->params->s) return false;
    if(m != (1u << logm)) return false;

    return true;
}


template<typename FieldT>
__device__ basic_radix2_domain<FieldT>& basic_radix2_domain<FieldT>::operator=(const basic_radix2_domain<FieldT>& other)
{
    this->m = other.m;
    this->instance = other.instance;
    this->omega = other.omega;
    return *this;
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::FFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);
    _basic_radix2_FFT(a, omega);
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::pFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    assert(a.size() == this->m);
    _p_basic_radix2_FFT(a, omega, gridSize, blockSize);
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::ppFFT(libstl::vector<FieldT>& a, size_t num)
{
    assert(a.size() == this->m);
    _pp_basic_radix2_FFT(a, omega, num);
}

template<typename FieldT>
__host__ void basic_radix2_domain<FieldT>::ppFFT_host(libstl::vector<FieldT>& a, size_t num)
{
    _pp_basic_radix2_FFT_host(a, omega, num);
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::iFFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);
    _basic_radix2_FFT(a, omega.inverse());

    const FieldT sconst = FieldT(this->instance->params, a.size()).inverse();
    
    for (size_t i = 0; i < a.size(); ++i)
        a[i] *= sconst;
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::piFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    assert(a.size() == this->m);

    FieldT* omega_inv = libstl::create<FieldT>(omega.inverse());
    _p_basic_radix2_FFT(a, *omega_inv, gridSize, blockSize);

    size_t num = gridSize * blockSize;
    size_t size = a.size();

    FieldT* sconst = libstl::create<FieldT>(FieldT(this->instance->params, a.size()).inverse());

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a] 
        __device__ () 
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for(size_t i = idx; i < size; i += num)
                a[i] *= *sconst;
        
        }     
    );
    cudaDeviceSynchronize();
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::ppiFFT(libstl::vector<FieldT>& a, size_t num)
{
    assert(a.size() == this->m);

    FieldT* omega_inv = libstl::create<FieldT>(omega.inverse());
    _pp_basic_radix2_FFT(a, *omega_inv, num);

    size_t size = a.size();

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    FieldT* sconst = libstl::create<FieldT>(FieldT(this->instance->params, a.size()).inverse());

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &a] 
        __device__ () 
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for(size_t i = idx; i < size; i += num)
                a[i] *= *sconst;
        
        }     
    );
    cudaDeviceSynchronize();
}


template<typename FieldT>
__host__ void basic_radix2_domain<FieldT>::ppiFFT_host(libstl::vector<FieldT>& a, size_t num)
{
    FieldT* omega_inv = this->omega.inverse_host();

    _pp_basic_radix2_FFT_host(a, *omega_inv, num);

    size_t a_size = a.size_host();
    size_t* da_size = libstl::create_host<size_t>();
    cudaMemcpy(da_size, &a_size, sizeof(size_t), cudaMemcpyHostToDevice);

    FieldT* instance_addr;
    cudaMemcpy(&instance_addr, &instance, sizeof(FieldT *), cudaMemcpyDeviceToHost);

    FieldT* f = libstl::create_host<FieldT>(instance_addr->params, *da_size);

    FieldT* sconst = f->inverse_host();

    size_t blockSize = min(num, (size_t)64);
    size_t gridSize = num / blockSize;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=] 
        __device__ (libstl::vector<FieldT>& a)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            for(size_t i = idx; i < a_size; i += num)
                a[i] *= *sconst;
        
        }, a
    );
    cudaStreamSynchronize(0);

}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::cosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    _multiply_by_coset(a, g);
    FFT(a);
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::pcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    _p_multiply_by_coset(a, g, gridSize, blockSize);
    pFFT(a, gridSize, blockSize);
}



template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::ppcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    _pp_multiply_by_coset(a, g, num);
    ppFFT(a, num);
}


template<typename FieldT>
__host__ void basic_radix2_domain<FieldT>::ppcosetFFT_host(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    _pp_multiply_by_coset_host(a, g, num);
    ppFFT_host(a, num);
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::icosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    iFFT(a);
    _multiply_by_coset(a, g.inverse());
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::picosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    piFFT(a, gridSize, blockSize);

    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _p_multiply_by_coset(a, *g_inv, gridSize, blockSize);
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::ppicosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t num)
{
    ppiFFT(a, num);
    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _pp_multiply_by_coset(a, *g_inv, num);
}

template<typename FieldT>
__host__ void basic_radix2_domain<FieldT>::ppicosetFFT_host(libstl::vector<FieldT>& a, FieldT& g, size_t num)
{
    ppiFFT_host(a, num);
    FieldT* g_inv = g.inverse_host();
    _pp_multiply_by_coset_host(a, *g_inv, num);
}

template<typename FieldT>
__device__ libstl::vector<FieldT> basic_radix2_domain<FieldT>::evaluate_all_lagrange_polynomials(const FieldT &t)
{
    return _basic_radix2_evaluate_all_lagrange_polynomials(this->m, t);
}

template<typename FieldT>
__device__ libstl::vector<FieldT> basic_radix2_domain<FieldT>::p_evaluate_all_lagrange_polynomials(const FieldT &t, size_t gridSize, size_t blockSize)
{
    return _p_basic_radix2_evaluate_all_lagrange_polynomials(this->m, t, gridSize, blockSize);
}


template<typename FieldT>
__device__ FieldT basic_radix2_domain<FieldT>::get_domain_element(const size_t idx)
{
    return omega ^ idx;
}

template<typename FieldT>
__device__ FieldT basic_radix2_domain<FieldT>::compute_vanishing_polynomial(const FieldT &t)
{
    return (t ^ this->m) - t.one();
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H)
{
    assert(H.size() == this->m + 1);

    H[this->m] += coeff;
    H[0] -= coeff;
}



template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::divide_by_Z_on_coset(libstl::vector<FieldT>& P)
{
    const FieldT coset = FieldT(this->instance->params, *this->instance->params->multiplicative_generator);
    const FieldT Z_inverse_at_coset = this->compute_vanishing_polynomial(coset).inverse();

    for(size_t i = 0; i < this->m; i++)
    {
        P[i] *= Z_inverse_at_coset;
    }
}

template<typename FieldT>
static __global__ void _divide_by_Z_on_coset_product_Z_inverse(libstl::vector<FieldT>& P, const FieldT& Z_inverse_at_coset, const size_t m)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0)
    {
        printf("kernel begin\n");
        printf("m: %lu\n", m);
        printf("P size: %lu", P.size());
    } 
    while(idx < m)
    {
        P[idx] *= Z_inverse_at_coset;
        idx += gridDim.x * blockDim.x;
    }
}

template<typename FieldT>
__device__ void basic_radix2_domain<FieldT>::p_divide_by_Z_on_coset(libstl::vector<FieldT>& P, size_t gridSize, size_t blockSize)
{
    const FieldT coset = FieldT(this->instance->params, *this->instance->params->multiplicative_generator);
    FieldT* Z_inverse_at_coset = libstl::create<FieldT>(this->compute_vanishing_polynomial(coset).inverse());

    _divide_by_Z_on_coset_product_Z_inverse<<<gridSize, blockSize>>>(P, *Z_inverse_at_coset, this->m);
    cudaDeviceSynchronize();
}

template<typename FieldT>
__host__ void basic_radix2_domain<FieldT>::p_divide_by_Z_on_coset_host(libstl::vector<FieldT>& P, size_t gridSize, size_t blockSize)
{
    FieldT* Z_inverse_at_coset = libstl::create_host<FieldT>();

    libstl::launch<<<1, 1>>>
    (
        [=] 
        __device__ ()
        {
            const FieldT coset = FieldT(this->instance->params, *this->instance->params->multiplicative_generator);
            *Z_inverse_at_coset = this->compute_vanishing_polynomial(coset).inverse();
        }
    );
    cudaStreamSynchronize(0);


    libstl::launch<<<gridSize, blockSize>>>
    (
        [=] 
        __device__ (libstl::vector<FieldT>& P)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < m)
            {
                P[idx] *= *Z_inverse_at_coset;
                idx += gridDim.x * blockDim.x;
            }
        }, P
    );
    cudaStreamSynchronize(0);
}


}

#endif