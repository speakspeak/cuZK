
#ifndef __CURVE_UTILS_CU__
#define __CURVE_UTILS_CU__

namespace libff{


template<typename GroupT, mp_size_t_ m>
__device__ GroupT scalar_mul(const GroupT &base, const bigint<m> &scalar)
{
    
    GroupT result = base.zero();

    bool found_one = false;
    for (long i = scalar.max_bits() - 1; i >= 0; --i)
    {
        if (found_one)
        {
            result = result.dbl();
        }

        if (scalar.test_bit(i))
        {
            found_one = true;
            result = result + base;
        }
    }

    return result;
}


template<typename GroupT>
__device__ GroupT scalar_mul(const GroupT &base, const unsigned long scalar)
{
    return scalar_mul<GroupT>(base, bigint<1>(scalar));
}

}


#endif