#ifndef __FIELD_UTILS_CU__
#define __FIELD_UTILS_CU__

#include <assert.h>
#include "../common/utils.cuh"

namespace libff {

template<typename FieldT>
__device__ FieldT coset_shift(const FieldT& instance)
{
    FieldT f(instance.params, *instance.params->multiplicative_generator);
    return f.squared();
}


// assume Field == Fp_model
template<typename FieldT>
__device__ FieldT get_root_of_unity(const size_t n, const FieldT& instance)
{
    const size_t logn = log2(n);
    assert(n == (1u << logn));
    assert(logn <= instance.params->s);


    FieldT omega(instance.params, *instance.params->root_of_unity);
    for (size_t i = instance.params->s; i > logn; --i)
    {
        omega *= omega;
    }

    return omega;
}

template<typename FieldT>
__device__ void batch_invert(libstl::vector<FieldT> &vec, const FieldT& instance)
{
    libstl::vector<FieldT> prod;
    prod.resize(vec.size());

    FieldT acc = instance.one();

    for (size_t i=0; i<vec.size(); i++)
    {
        prod[i] = acc;
        acc = acc * vec[i];
    }

    FieldT acc_inverse = acc.inverse();

    for (size_t i = vec.size()-1; i <= vec.size(); --i)
    {
        const FieldT old_el = vec[i];
        vec[i] = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}

}


#endif