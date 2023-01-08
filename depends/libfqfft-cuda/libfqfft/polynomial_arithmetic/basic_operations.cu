#ifndef __BASIC_OPERATIONS_CU__
#define __BASIC_OPERATIONS_CU__

#include <assert.h>
#include "../../depends/libstl-cuda/algorithm.cuh"
#include "../../depends/libstl-cuda/functional.cuh"
#include "../evaluation_domain/domains/basic_radix2_domain_aux.cuh"
//#include "../kronecker_substitution/kronecker_substitution.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ bool _is_zero(const libstl::vector<FieldT>& a)
{
    FieldT instance = a[0];
    return libstl::all_of(a.begin(), a.end(), [&](FieldT i) { return i == instance.zero(); });
}

template<typename FieldT>
__device__ void _condense(libstl::vector<FieldT>& a)
{
    FieldT instance = a[0];

    size_t new_size = a.size();

    while (new_size > 0 && a[new_size - 1].is_zero())
        new_size--;

    a.resize(new_size, instance);
}

template<typename FieldT>
__device__ void _reverse(libstl::vector<FieldT>& a, const size_t n)
{
    FieldT instance = a[0];

    libstl::reverse(a.begin(), a.end());

    a.resize(n, instance);
}

template<typename FieldT>
__device__ void _polynomial_addition(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b)
{
    FieldT instance = a[0];

    if (_is_zero(a))
        c = b;
    else if (_is_zero(b))
        c = a;
    else
    {
        size_t a_size = a.size();
        size_t b_size = b.size();

        if (a_size > b_size)
        {
            c.resize(a_size, instance);
            libstl::transform(b.begin(), b.end(), a.begin(), c.begin(), libstl::plus<FieldT>());
            libstl::copy(a.begin() + b_size, a.end(), c.begin() + b_size);
        }
        else
        {
            c.resize(b_size, instance);
            libstl::transform(a.begin(), a.end(), b.begin(), c.begin(), libstl::plus<FieldT>());
            libstl::copy(b.begin() + a_size, b.end(), c.begin() + a_size);
        }
    }
        
    _condense(c);
}

template<typename FieldT>
__device__ void _polynomial_subtraction(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b)
{
    FieldT instance = a[0];

    if (_is_zero(b))
        c = a;
    else if (_is_zero(a))
    {
        c.resize(b.size(), instance);
        libstl::transform(b.begin(), b.end(), c.begin(), libstl::negate<FieldT>());
    }
    else
    {
        size_t a_size = a.size();
        size_t b_size = b.size();
        
        if (a_size > b_size)
        {
            c.resize(a_size, instance);
            libstl::transform(a.begin(), a.begin() + b_size, b.begin(), c.begin(), libstl::minus<FieldT>());
            libstl::copy(a.begin() + b_size, a.end(), c.begin() + b_size);
        }
        else
        {
            c.resize(b_size, instance);
            libstl::transform(a.begin(), a.end(), b.begin(), c.begin(), libstl::minus<FieldT>());
            libstl::transform(b.begin() + a_size, b.end(), c.begin() + a_size, libstl::negate<FieldT>());
        }
    }

    _condense(c);
}

template<typename FieldT>
__device__ void _polynomial_multiplication(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b)
{
    _polynomial_multiplication_on_fft(c, a, b);
}

template<typename FieldT>
__device__ void _polynomial_multiplication_on_fft(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b)
{
    FieldT instance = a[0];

    const size_t n = 1;//libff::get_power_of_two(a.size() + b.size() - 1);
    FieldT omega = libff::get_root_of_unity<FieldT>(n, instance);

    libstl::vector<FieldT> u(a);
    libstl::vector<FieldT> v(b);

    u.resize(n, a[0].zero());
    v.resize(n, a[0].zero());
    c.resize(n, a[0].zero());

    _basic_radix2_FFT(u, omega);
    _basic_radix2_FFT(v, omega);

    libstl::transform(u.begin(), u.end(), v.begin(), c.begin(), libstl::multiplies<FieldT>());

    _basic_radix2_FFT(c, omega.inverse());

    const FieldT sconst = FieldT(instance.params, n).inverse();
    libstl::transform(c.begin(), c.end(), c.begin(), libstl::bind1st(libstl::multiplies<FieldT>(), sconst));
    
    _condense(c);
}

template<typename FieldT>
__device__ void _polynomial_multiplication_on_kronecker(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b)
{
    //kronecker_substitution(c, a, b);
}

template<typename FieldT>
__device__ libstl::vector<FieldT> _polynomial_multiplication_transpose(const size_t& n, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& c)
{
    const size_t m = a.size();

    assert(c.size() - 1 <= m + n);

    libstl::vector<FieldT> r(a);
    _reverse(r, m);
    _polynomial_multiplication(r, r, c);

    /* Determine Middle Product */
    return libstl::vector<FieldT>(r.begin() + m - 1, r.begin() + n + m);
}

template<typename FieldT>
__device__ void _polynomial_division(libstl::vector<FieldT>& q, libstl::vector<FieldT>& r, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b)
{
    FieldT instance = a[0];

    size_t d = b.size() - 1; /* Degree of B */
    FieldT c = b.back().inverse(); /* Inverse of Leading Coefficient of B */

    r = libstl::vector<FieldT>(a);
    q = libstl::vector<FieldT>(r.size(), instance.zero());

    size_t r_deg = r.size() - 1;
    size_t shift;

    while (r_deg >= d && !_is_zero(r))
    {
        shift = r_deg - d;

        FieldT lead_coeff = r.back() * c;

        q[shift] += lead_coeff;

        auto glambda = [=](FieldT x, FieldT y) { return y - (x * lead_coeff); };

        libstl::transform(b.begin(), b.end(), r.begin() + shift, r.begin() + shift, glambda);

        _condense(r);

        r_deg = r.size() - 1;
    }

    _condense(q);
}

}


#endif