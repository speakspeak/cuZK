#ifndef __BASIS_CHANGE_CU__
#define __BASIS_CHANGE_CU__

#include <assert.h>
#include "../../depends/libstl-cuda/algorithm.cuh"
#include "../evaluation_domain/domains/basic_radix2_domain_aux.cuh"
#include "basic_operations.cuh"
#include "xgcd.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ void compute_subproduct_tree(const size_t& m, const FieldT& instance, libstl::vector<libstl::vector<libstl::vector<FieldT>>>& T)
{
    if (T.size() != m + 1) 
        T.resize(m + 1);

    /*
     * Subproduct tree T is represented as a 2-dimensional array T_{i, j}.
     * T_{i, j} = product_{l = [2^i * j] to [2^i * (j+1) - 1]} (x - x_l)
     * Note: n = 2^m.
     */

    /* Precompute the first row. */
    T[0] = libstl::vector<libstl::vector<FieldT>>(1u << m);

    for (size_t j = 0; j < (1u << m); j++)
    {
        T[0][j] = libstl::vector<FieldT>(2, instance.one());
        T[0][j][0] = FieldT(instance.params, -j);
    }

    libstl::vector<FieldT> a;
    libstl::vector<FieldT> b;

    size_t index = 0;
    for (size_t i = 1; i <= m; i++)
    {
        T[i] = libstl::vector<libstl::vector<FieldT>>(1u << (m - i));
        for (size_t j = 0; j < (1u << (m-i)); j++)
        {
            a = T[i-1][index];
            index++;

            b = T[i-1][index];
            index++;

            _polynomial_multiplication(T[i][j], a, b);
        }
        index = 0;
    }
}

template<typename FieldT>
__device__ void monomial_to_newton_basis(libstl::vector<FieldT>& a, const libstl::vector<libstl::vector<libstl::vector<FieldT>>>& T, const size_t& n)
{
    size_t m = libff::log2(n);

    assert(T.size() == m + 1u);

    FieldT instance = T[0][0][0];

    /* MonomialToNewton */
    libstl::vector<FieldT> I(T[m][0]);
    _reverse(I, n);

    libstl::vector<FieldT> mod(n + 1, instance.zero());
    mod[n] = instance.one();

    _polynomial_xgcd(mod, I, mod, mod, I);

    I.resize(n, instance);

    libstl::vector<FieldT> Q(_polynomial_multiplication_transpose(n - 1, I, a));
    _reverse(Q, n);

    /* TNewtonToMonomial */
    libstl::vector<libstl::vector<FieldT>> c(n);
    c[0] = Q;

    size_t row_length;
    size_t c_vec;
    /* NB: unsigned reverse iteration: cannot do i >= 0, but can do i < m
       because unsigned integers are guaranteed to wrap around */
    for (size_t i = m - 1; i < m; i--)
    {
        row_length = T[i].size() - 1;
        c_vec = 1u << i;

        /* NB: unsigned reverse iteration */
        for (size_t j = (1u << (m - i - 1)) - 1;
             j < (1u << (m - i - 1));
             j--)
        {
            c[2 * j + 1] = _polynomial_multiplication_transpose(
                (1u << i) - 1, T[i][row_length - 2 * j], c[j]);
            c[2 * j] = c[j];
            c[2 * j].resize(c_vec, instance);
        }
    }

    /* Store Computed Newton Basis Coefficients */
    size_t j = 0;
    /* NB: unsigned reverse iteration */
    for (size_t i = c.size() - 1; i < c.size(); i--)
        a[j++] = c[i][0];
}

template<typename FieldT>
__device__ void newton_to_monomial_basis(libstl::vector<FieldT>& a, const libstl::vector<libstl::vector<libstl::vector<FieldT>>>& T, const size_t& n)
{
    size_t m = libff::log2(n);

    assert(T.size() == m + 1u);

    FieldT instance = T[0][0][0];

    libstl::vector <libstl::vector<FieldT>> f(n);
    for (size_t i = 0; i < n; i++)
        f[i] = libstl::vector<FieldT>(1, a[i]);

    /* NewtonToMonomial */
    libstl::vector<FieldT> temp(1, instance.zero());
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < (1u << (m - i - 1)); j++)
        {
            _polynomial_multiplication(temp, T[i][2 * j], f[2 * j + 1]);
            _polynomial_addition(f[j], f[2 * j], temp);
        }
    }

    a = f[0];
}

template<typename FieldT>
__device__ void monomial_to_newton_basis_geometric(libstl::vector<FieldT>& a,
                                             const libstl::vector<FieldT>& geometric_sequence,
                                             const libstl::vector<FieldT>& geometric_triangular_sequence,
                                             const size_t& n)
{

    FieldT instance = a[0];

    libstl::vector<FieldT> u(n, instance.zero());
    libstl::vector<FieldT> w(n, instance.zero());
    libstl::vector<FieldT> z(n, instance.zero());
    libstl::vector<FieldT> f(n, instance.zero());

    u[0] = instance.one();
    w[0] = a[0];
    z[0] = instance.one();
    f[0] = a[0];

    for (size_t i = 1; i < n; i++)
    {
        u[i] = u[i - 1] * geometric_sequence[i] * (instance.one() - geometric_sequence[i]).inverse();
        w[i] = a[i] * (u[i].inverse());
        z[i] = u[i] * geometric_triangular_sequence[i].inverse();
        f[i] = w[i] * geometric_triangular_sequence[i];

        if (i % 2 == 1)
        {
            z[i] = -z[i];
            f[i] = -f[i];
        }
    }

    w = _polynomial_multiplication_transpose(n - 1, z, f);

    for (size_t i = 0; i < n; i++)
        a[i] = w[i] * z[i];
}

template<typename FieldT>
__device__ void newton_to_monomial_basis_geometric(libstl::vector<FieldT>& a,
                                             const libstl::vector<FieldT>& geometric_sequence,
                                             const libstl::vector<FieldT>& geometric_triangular_sequence,
                                             const size_t& n)
{
    FieldT instance = a[0];

    libstl::vector<FieldT> v(n, instance.zero());
    libstl::vector<FieldT> u(n, instance.zero());
    libstl::vector<FieldT> w(n, instance.zero());
    libstl::vector<FieldT> z(n, instance.zero());

    v[0] = a[0];
    u[0] = instance.one();
    w[0] = a[0];
    z[0] = instance.one();

    for (size_t i = 1; i < n; i++)
    {
        v[i] = a[i] * geometric_triangular_sequence[i];
        if (i % 2 == 1) 
            v[i] = -v[i];

        u[i] = u[i-1] * geometric_sequence[i] * (instance.one() - geometric_sequence[i]).inverse();
        w[i] = v[i] * u[i].inverse();

        z[i] = u[i] * geometric_triangular_sequence[i].inverse();
        if (i % 2 == 1) 
            z[i] = -z[i];
    }

    w = _polynomial_multiplication_transpose(n - 1, u, w);

    for (size_t i = 0; i < n; i++)
        a[i] = w[i] * z[i];
}

} // libfqfft

#endif