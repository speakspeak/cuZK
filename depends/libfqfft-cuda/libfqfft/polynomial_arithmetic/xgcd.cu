#ifndef __XGCD_CU__
#define __XGCD_CU__

#include "../../depends/libstl-cuda/algorithm.cuh"
#include "../evaluation_domain/domains/basic_radix2_domain_aux.cuh"
#include "basic_operations.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ void _polynomial_xgcd(const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b, libstl::vector<FieldT>& g, libstl::vector<FieldT>& u, libstl::vector<FieldT>& v)
{
    FieldT instance = a[0];

    if (_is_zero(b))
    {
        g = a;
        u = libstl::vector<FieldT>(1, instance.one());
        v = libstl::vector<FieldT>(1, instance.zero());
        return;
    }

    libstl::vector<FieldT> U(1, instance.one());
    libstl::vector<FieldT> V1(1, instance.zero());
    libstl::vector<FieldT> G(a);
    libstl::vector<FieldT> V3(b);

    libstl::vector<FieldT> Q(1, instance.zero());
    libstl::vector<FieldT> R(1, instance.zero());
    libstl::vector<FieldT> T(1, instance.zero());

    while (!_is_zero(V3))
    {
        _polynomial_division(Q, R, G, V3);
        _polynomial_multiplication(G, V1, Q);
        _polynomial_subtraction(T, U, G);

        U = V1;
        G = V3;
        V1 = T;
        V3 = R;
    }

    _polynomial_multiplication(V3, a, U);
    _polynomial_subtraction(V3, G, V3);
    _polynomial_division(V1, R, V3, b);

    FieldT lead_coeff = G.back().inverse();
    libstl::transform(G.begin(), G.end(), G.begin(), libstl::bind1st(libstl::multiplies<FieldT>(), lead_coeff));
    libstl::transform(U.begin(), U.end(), U.begin(), libstl::bind1st(libstl::multiplies<FieldT>(), lead_coeff));
    libstl::transform(V1.begin(), V1.end(), V1.begin(), libstl::bind1st(libstl::multiplies<FieldT>(), lead_coeff));

    g = G;
    u = U;
    v = V1;
}

} // libfqfft

#endif