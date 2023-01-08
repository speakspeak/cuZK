#ifndef __GEOMETRIC_SEQUENCE_DOMAIN_CU__
#define __GEOMETRIC_SEQUENCE_DOMAIN_CU__

#include <assert.h>
#include "../../../depends/libff-cuda/fields/field_utils.cuh"
#include "../../../depends/libff-cuda/common/utils.cuh"
#include "../../polynomial_arithmetic/basis_change.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ geometric_sequence_domain<FieldT>::geometric_sequence_domain(const size_t m, const FieldT& instance): m(m), instance(instance)
{
    assert(m > 1);
    assert(this->instance.geometric_generator() != this->instance.zero());
    precomputation_sentinel = 0;
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::FFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    if (!this->precomputation_sentinel) 
        do_precomputation();

    monomial_to_newton_basis_geometric(a, this->geometric_sequence, this->geometric_triangular_sequence, this->m);

    /* Newton to Evaluation */
    libstl::vector<FieldT> T(this->m, this->instance);
    T[0] = this->instance.one();

    libstl::vector<FieldT> g(this->m, this->instance);
    g[0] = a[0];

    for (size_t i = 1; i < this->m; i++)
    {
        T[i] = T[i-1] * (this->geometric_sequence[i] - this->instance.one()).inverse();
        g[i] = this->geometric_triangular_sequence[i] * a[i];
    }

    _polynomial_multiplication(a, g, T);
    a.resize(this->m, this->instance);

    for (size_t i = 0; i < this->m; i++)
        a[i] *= T[i].inverse();
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::pFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    FFT(a);
}


template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::iFFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    if (!this->precomputation_sentinel) 
        do_precomputation();

    /* Interpolation to Newton */
    libstl::vector<FieldT> T(this->m, this->instance);
    T[0] = this->instance.one();

    libstl::vector<FieldT> W(this->m, this->instance);
    W[0] = a[0] * T[0];

    FieldT prev_T = T[0];
    for (size_t i = 1; i < this->m; i++)
    {
        prev_T *= (this->geometric_sequence[i] - this->instance.one()).inverse();

        W[i] = a[i] * prev_T;
        T[i] = this->geometric_triangular_sequence[i] * prev_T;
        if (i % 2 == 1) 
            T[i] = -T[i];
    }

    _polynomial_multiplication(a, W, T);
    a.resize(this->m, this->instance);

    for (size_t i = 0; i < this->m; i++)
        a[i] *= this->geometric_triangular_sequence[i].inverse();

    newton_to_monomial_basis_geometric(a, this->geometric_sequence, this->geometric_triangular_sequence, this->m);
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::piFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    iFFT(a);
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::cosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    _multiply_by_coset(a, g);
    FFT(a);
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::pcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    _p_multiply_by_coset(a, g, gridSize, blockSize);
    pFFT(a, gridSize, blockSize);    
}


template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::icosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    iFFT(a);
    _multiply_by_coset(a, g.inverse());
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::picosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    piFFT(a, gridSize, blockSize);

    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _p_multiply_by_coset(a, *g_inv, gridSize, blockSize);
}

template<typename FieldT>
__device__ libstl::vector<FieldT> geometric_sequence_domain<FieldT>::evaluate_all_lagrange_polynomials(const FieldT &t)
{
    /* Compute Lagrange polynomial of size m, with m+1 points (x_0, y_0), ... ,(x_m, y_m) */
    /* Evaluate for x = t */
    /* Return coeffs for each l_j(x) = (l / l_i[j]) * w[j] */

    /* for all i: w[i] = (1 / r) * w[i-1] * (1 - a[i]^m-i+1) / (1 - a[i]^-i) */

    if (!this->precomputation_sentinel) 
        do_precomputation();

    /**
    * If t equals one of the geometric progression values,
    * then output 1 at the right place, and 0 elsewhere.
    */
    for (size_t i = 0; i < this->m; ++i)
    {
        if (this->geometric_sequence[i] == t) // i.e., t equals a[i]
        {
            libstl::vector<FieldT> res(this->m, this->instance.zero());
            res[i] = this->instance.one();
            return res;
        }
    }

    /**
    * Otherwise, if t does not equal any of the geometric progression values,
    * then compute each Lagrange coefficient.
    */
    libstl::vector<FieldT> l(this->m, this->instance);
    l[0] = t - this->geometric_sequence[0];

    libstl::vector<FieldT> g(this->m, this->instance);
    g[0] = this->instance.zero();

    FieldT l_vanish = l[0];
    FieldT g_vanish = this->instance.one();
    for (size_t i = 1; i < this->m; i++)
    {
        l[i] = t - this->geometric_sequence[i];
        g[i] = this->instance.one() - this->geometric_sequence[i];

        l_vanish *= l[i];
        g_vanish *= g[i];
    }

    FieldT r = this->geometric_sequence[this->m - 1].inverse();
    FieldT r_i = r;

    libstl::vector<FieldT> g_i(this->m, this->instance);
    g_i[0] = g_vanish.inverse();

    l[0] = l_vanish * l[0].inverse() * g_i[0];
    for (size_t i = 1; i < this->m; i++)
    {
        g_i[i] = g_i[i-1] * g[this->m-i] * -g[i].inverse() * this->geometric_sequence[i];
        l[i] = l_vanish * r_i * l[i].inverse() * g_i[i];
        r_i *= r;
    }

    return l;
}

template<typename FieldT>
__device__ FieldT geometric_sequence_domain<FieldT>::get_domain_element(const size_t idx)
{
    if (!this->precomputation_sentinel) 
        do_precomputation();

    return this->geometric_sequence[idx];
}

template<typename FieldT>
__device__ FieldT geometric_sequence_domain<FieldT>::compute_vanishing_polynomial(const FieldT &t)
{
    if (!this->precomputation_sentinel) 
        do_precomputation();

    /* Notes: Z = prod_{i = 0 to m} (t - a[i]) */
    /* Better approach: Montgomery Trick + Divide&Conquer/FFT */
    FieldT Z = this->instance.one();
    for (size_t i = 0; i < this->m; i++)
        Z *= (t - this->geometric_sequence[i]);
    
    return Z;
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H)
{
    assert(H.size() == this->m + 1);

    if (!this->precomputation_sentinel) 
        do_precomputation();

    libstl::vector<FieldT> x(2, this->instance.zero());
    x[0] = -this->geometric_sequence[0];
    x[1] = this->instance.one();

    libstl::vector<FieldT> t(2, this->instance.zero());

    for (size_t i = 1; i < this->m+1; i++)
    {
        t[0] = -this->geometric_sequence[i];
        t[1] = this->instance.one();

        _polynomial_multiplication(x, x, t);
    }

    for (size_t i = 0; i < this->m+1; i++)
        H[i] += (x[i] * coeff);
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::divide_by_Z_on_coset(libstl::vector<FieldT>& P)
{
    const FieldT coset = FieldT(this->instance.params, *this->instance.params->multiplicative_generator); /* coset in geometric sequence? */
    const FieldT Z_inverse_at_coset = this->compute_vanishing_polynomial(coset).inverse();
    for (size_t i = 0; i < this->m; ++i)
        P[i] *= Z_inverse_at_coset;
}

template<typename FieldT>
__device__ void geometric_sequence_domain<FieldT>::do_precomputation()
{
    this->geometric_sequence = libstl::vector<FieldT>(this->m, this->instance.zero());
    this->geometric_sequence[0] = this->instance.one();

    this->geometric_triangular_sequence = libstl::vector<FieldT>(this->m, this->instance.zero());
    this->geometric_triangular_sequence[0] = this->instance.one();

    for (size_t i = 1; i < this->m; i++)
    {
        this->geometric_sequence[i] = this->geometric_sequence[i - 1] * this->instance.geometric_generator();
        this->geometric_triangular_sequence[i] = this->geometric_triangular_sequence[i - 1] * this->geometric_sequence[i - 1];
    }

    this->precomputation_sentinel = 1;
}

}

#endif