#ifndef __EXPONENTIATION_CU__
#define __EXPONENTIATION_CU__

namespace libff {

template<typename FieldT, mp_size_t_ m>
__device__ FieldT power(const FieldT &base, const bigint<m> &exponent)
{
    FieldT t(base.params);
    FieldT result = t.one();

    bool found_one = false;

    for (long i = exponent.max_bits() - 1; i >= 0; --i)
    {
        if (found_one)
        {
            result = result * result;
        }

        if (exponent.test_bit(i))
        {
            found_one = true;
            result = result * base;
        }
    }

    return result;
}

template<typename FieldT>
__device__ FieldT power(const FieldT &base, const unsigned long exponent)
{
    return power<FieldT>(base, bigint<1>(exponent));
}


template<typename FieldT, mp_size_t_ m>
__host__ FieldT power_host(const FieldT &base, const bigint_host<m> &exponent)
{
    FieldT t(base.params);
    FieldT result = t.one();

    bool found_one = false;

    for (long i = exponent.max_bits() - 1; i >= 0; --i)
    {
        if (found_one)
        {
            result = result * result;
        }

        if (exponent.test_bit(i))
        {
            found_one = true;
            result = result * base;
        }
    }

    return result;
}

template<typename FieldT>
__host__ FieldT power_host(const FieldT &base, const unsigned long exponent)
{
    return power_host<FieldT>(base, bigint_host<1>(exponent));
}

}

#endif