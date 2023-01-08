#ifndef __NAIVE_EVALUATE_CU__
#define __NAIVE_EVALUATE_CU__

#include <assert.h>
#include "../../depends/libstl-cuda/algorithm.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ FieldT evaluate_polynomial(const size_t& m, const libstl::vector<FieldT>& coeff, const FieldT& t)
{
    assert(m == coeff.size());

    FieldT instance = t;

    FieldT result = instance.zero();

    /* NB: unsigned reverse iteration: cannot do i >= 0, but can do i < m
       because unsigned integers are guaranteed to wrap around */
    for (size_t i = m - 1; i < m; i--)
        result = (result * t) + coeff[i];

    return result;
}

template<typename FieldT>
__device__ FieldT evaluate_lagrange_polynomial(const size_t& m, const libstl::vector<FieldT>& domain, const FieldT& t, const size_t& idx)
{
    assert(m == domain.size());
    assert(idx < m);

    FieldT instance = t;

    FieldT num = instance.one();
    FieldT denom = instance.one();

    for (size_t k = 0; k < m; ++k)
    {
        if (k == idx)
            continue;

        num *= t - domain[k];
        denom *= domain[idx] - domain[k];
    }

    return num * denom.inverse();
}

} // libfqfft

#endif
