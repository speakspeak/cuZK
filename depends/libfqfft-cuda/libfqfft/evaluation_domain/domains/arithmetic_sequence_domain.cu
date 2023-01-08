#ifndef __ARITHMETIC_SEQUENCE_DOMAIN_CU__
#define __ARITHMETIC_SEQUENCE_DOMAIN_CU__

#include <assert.h>
#include "../../../depends/libff-cuda/fields/field_utils.cuh"
#include "../../../depends/libff-cuda/common/utils.cuh"
#include "../../polynomial_arithmetic/basis_change.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ arithmetic_sequence_domain<FieldT>::arithmetic_sequence_domain(const size_t m, const FieldT& instance): m(m), instance(instance), arithmetic_generator(instance)
{
    assert(m > 1);
    assert(this->instance.arithmetic_generator() != this->instance.zero());
    precomputation_sentinel = 0;
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::FFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    if (!this->precomputation_sentinel) 
        do_precomputation();

    /* Monomial to Newton */
    monomial_to_newton_basis(a, this->subproduct_tree, this->m);
    
    /* Newton to Evaluation */
    libstl::vector<FieldT> S(this->m, this->instance); /* i! * arithmetic_generator */
    S[0] = this->instance.one();

    FieldT factorial = this->instance.one();
    for (size_t i = 1; i < this->m; i++)
    {
        factorial *= FieldT(this->instance.params, i);
        S[i] = (factorial * this->arithmetic_generator).inverse();
    }

    _polynomial_multiplication(a, a, S);
    a.resize(this->m, this->instance);

    for (size_t i = 0; i < this->m; i++)
        a[i] *= S[i].inverse();
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::pFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    FFT(a);
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::iFFT(libstl::vector<FieldT>& a)
{
    assert(a.size() == this->m);

    if (!this->precomputation_sentinel) 
        do_precomputation();

    /* Interpolation to Newton */
    libstl::vector<FieldT> S(this->m, this->instance); /* i! * arithmetic_generator */
    S[0] = this->instance.one();

    libstl::vector<FieldT> W(this->m, this->instance);
    W[0] = a[0] * S[0];

    FieldT factorial = this->instance.one();
    for (size_t i = 1; i < this->m; i++)
    {
        factorial *= FieldT(this->instance.params, i);
        S[i] = (factorial * this->arithmetic_generator).inverse();
        W[i] = a[i] * S[i];
        if (i % 2 == 1) 
            S[i] = -S[i];
    }

    _polynomial_multiplication(a, W, S);
    a.resize(this->m, this->instance);

    /* Newton to Monomial */
    newton_to_monomial_basis(a, this->subproduct_tree, this->m);
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::piFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize)
{
    iFFT(a);
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::cosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    _multiply_by_coset(a, g);
    FFT(a);
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::pcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    _p_multiply_by_coset(a, g, gridSize, blockSize);
    pFFT(a, gridSize, blockSize);
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::icosetFFT(libstl::vector<FieldT>& a, const FieldT& g)
{
    iFFT(a);
    _multiply_by_coset(a, g.inverse());
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::picosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize)
{
    piFFT(a, gridSize, blockSize);

    FieldT* g_inv = libstl::create<FieldT>(g.inverse());
    _p_multiply_by_coset(a, *g_inv, gridSize, blockSize);
}

template<typename FieldT>
__device__ libstl::vector<FieldT> arithmetic_sequence_domain<FieldT>::evaluate_all_lagrange_polynomials(const FieldT &t)
{
    /* Compute Lagrange polynomial of size m, with m+1 points (x_0, y_0), ... ,(x_m, y_m) */
    /* Evaluate for x = t */
    /* Return coeffs for each l_j(x) = (l / l_i[j]) * w[j] */

    if (!this->precomputation_sentinel) 
        do_precomputation();

    /**
    * If t equals one of the arithmetic progression values,
    * then output 1 at the right place, and 0 elsewhere.
    */
    for (size_t i = 0; i < this->m; ++i)
    {
        if (this->arithmetic_sequence[i] == t) // i.e., t equals this->arithmetic_sequence[i]
        {
            libstl::vector<FieldT> res(this->m, this->instance.zero());
            res[i] = this->instance.one();
            return res;
        }
    }

    /**
    * Otherwise, if t does not equal any of the arithmetic progression values,
    * then compute each Lagrange coefficient.
    */
    libstl::vector<FieldT> l(this->m, this->instance);
    l[0] = t - this->arithmetic_sequence[0];

    FieldT l_vanish = l[0];
    FieldT g_vanish = this->instance.one();

    for (size_t i = 1; i < this->m; i++)
    {
        l[i] = t - this->arithmetic_sequence[i];
        l_vanish *= l[i];
        g_vanish *= -this->arithmetic_sequence[i];
    }

    libstl::vector<FieldT> w(this->m, this->instance);
    w[0] = g_vanish.inverse() * (this->arithmetic_generator ^ (this->m - 1));
    
    l[0] = l_vanish * l[0].inverse() * w[0];
    for (size_t i = 1; i < this->m; i++)
    {
        FieldT num = this->arithmetic_sequence[i-1] - this->arithmetic_sequence[this->m-1];
        w[i] = w[i-1] * num * this->arithmetic_sequence[i].inverse();
        l[i] = l_vanish * l[i].inverse() * w[i];
    }

    return l;
}

template<typename FieldT>
__device__ FieldT arithmetic_sequence_domain<FieldT>::get_domain_element(const size_t idx)
{
    if (!this->precomputation_sentinel) 
        do_precomputation();

    return this->arithmetic_sequence[idx];
}

template<typename FieldT>
__device__ FieldT arithmetic_sequence_domain<FieldT>::compute_vanishing_polynomial(const FieldT &t)
{
    if (!this->precomputation_sentinel) 
        do_precomputation();

    /* Notes: Z = prod_{i = 0 to m} (t - a[i]) */
    FieldT Z = this->instance.one();
    for (size_t i = 0; i < this->m; i++)
        Z *= (t - this->arithmetic_sequence[i]);
    
    return Z;
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H)
{
    assert(H.size() == this->m + 1);

    if (!this->precomputation_sentinel) 
        do_precomputation();

    libstl::vector<FieldT> x(2, this->instance.zero());
    x[0] = -this->arithmetic_sequence[0];
    x[1] = this->instance.one();

    libstl::vector<FieldT> t(2, this->instance.zero());

    for (size_t i = 1; i < this->m + 1; i++)
    {
        t[0] = -this->arithmetic_sequence[i];
        t[1] = this->instance.one();

        _polynomial_multiplication(x, x, t);
    }

    for (size_t i = 0; i < this->m+1; i++)
        H[i] += (x[i] * coeff);
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::divide_by_Z_on_coset(libstl::vector<FieldT>& P)
{
    const FieldT coset = this->arithmetic_generator; /* coset in arithmetic sequence? */
    const FieldT Z_inverse_at_coset = this->compute_vanishing_polynomial(coset).inverse();
    for (size_t i = 0; i < this->m; ++i)
        P[i] *= Z_inverse_at_coset;
}

template<typename FieldT>
__device__ void arithmetic_sequence_domain<FieldT>::do_precomputation()
{
    compute_subproduct_tree(libff::log2(this->m), this->instance, this->subproduct_tree);

    this->arithmetic_generator = this->instance.arithmetic_generator();

    this->arithmetic_sequence = libstl::vector<FieldT>(this->m, this->instance);
    
    for (size_t i = 0; i < this->m; i++)
        this->arithmetic_sequence[i] = this->arithmetic_generator * FieldT(this->instance.params, i);

    this->precomputation_sentinel = 1;
}

}

#endif