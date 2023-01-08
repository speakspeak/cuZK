
#ifndef __CURVE_UTILS_HOST_CU__
#define __CURVE_UTILS_HOST_CU__

namespace libff{


template<typename GroupT, mp_size_t m>
GroupT scalar_mul_host(const GroupT &base, const bigint_host<m> &scalar)
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
GroupT scalar_mul_host(const GroupT &base, const unsigned long scalar)
{
    return scalar_mul_host<GroupT>(base, bigint_host<1>(scalar));
}

}


#endif