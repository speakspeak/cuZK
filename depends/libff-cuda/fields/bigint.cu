#ifndef __BIGINT_CU__
#define __BIGINT_CU__

#include <assert.h>

namespace libff {

__device__ static size_t dstrlen(const char *str)
{
    const char *cp = str;
    while (*cp++);
    return (cp - str - 1);
}

template<mp_size_t_ n>
__device__ bigint<n>::bigint(const unsigned long x)
{
    clear();
    data[0] = x;
}

template<mp_size_t_ n>
__device__ bigint<n>::bigint(const char* s)
{
    clear();
    size_t l = dstrlen(s);

    unsigned char* s_copy = (unsigned char*)malloc(l * sizeof(unsigned char));

    for (size_t i = 0; i < l; ++i)
    {
        assert(s[i] >= '0' && s[i] <= '9');
        s_copy[i] = s[i] - '0';
    }

    mp_size_t_ limbs_written = mpn_set_str_(this->data, s_copy, l, 10);
    assert(limbs_written <= n);

    free(s_copy);
}

template<mp_size_t_ n>
__device__ bigint<n>::bigint(const mpz_t_ r)
{
    clear();
    mpz_t_ k;
    mpz_init__set_(k, r);

    for (size_t i = 0; i < n; ++i)
    {
        data[i] = mpz_get_ui_(k);
        mpz_fdiv_q__2exp_(k, k, 8 * sizeof(mp_limb_t_));
    }

    assert(mpz_sgn_(k) == 0);
    mpz_clear_(k);
}

template<mp_size_t_ n>
__device__ bool bigint<n>::operator==(const bigint<n>& other) const
{
    for (size_t i = 0; i < n; i++)
        if (data[i] != other.data[i])
            return false;
    return true;
}

template<mp_size_t_ n>
__device__ bool bigint<n>::operator!=(const bigint<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t_ n>
__device__ bool bigint<n>::operator>=(const bigint<n>& other) const
{
    for (int i = n - 1; i >= 0; i--) 
    {
        if (data[i] > other.data[i])
            return true;
        if (data[i] < other.data[i])
            return false;
    }

    return true;
}

template<mp_size_t_ n>
__device__ bigint<n>& bigint<n>::operator+=(const bigint<n>& other)
{
    bool carry = 0;
    for (size_t i = 0; i < n; i++) 
    {
        mp_limb_t_ old = data[i];
        data[i] += other.data[i] + carry;
        carry = carry ? old >= data[i] : old > data[i];
    }
    return *this;
}

template<mp_size_t_ n>
__device__ bigint<n>& bigint<n>::operator-=(const bigint<n>& other)
{
    bool borrow = 0;
    for (size_t i = 0; i < n; i++) 
    {
        mp_limb_t_ old = data[i];
        data[i] -= other.data[i] + borrow;
        borrow = borrow ? old <= data[i] : old < data[i];
    }
    return *this;
}

template<mp_size_t_ n>
__device__ void bigint<n>::clear()
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
}

template<mp_size_t_ n>
__device__ bool bigint<n>::is_zero() const
{
    for (size_t i = 0; i < n; ++i)
        if (this->data[i])
            return false;

    return true;
}

template<mp_size_t_ n>
__device__ size_t bigint<n>::num_bits() const
{
    for (long i = max_bits(); i >= 0; --i)
        if (this->test_bit(i))
            return i + 1;

    return 0;
}

template<mp_size_t_ n>
__device__ unsigned long bigint<n>::as_ulong() const
{
    return this->data[0];
}

template<mp_size_t_ n>
__device__ void bigint<n>::to_mpz(mpz_t_ r) const
{
    mpz_set__ui_(r, 0);

    for (int i = n - 1; i >= 0; --i)
    {
        mpz_mul__2exp_(r, r, 8 * sizeof(mp_limb_t_));
        mpz_add__ui_(r, r, this->data[i]);
    }
}

template<mp_size_t_ n>
__device__ bool bigint<n>::test_bit(const std::size_t bitno) const
{
    if (bitno >= n * 8 * sizeof(mp_limb_t_))
        return false;

    const std::size_t part = bitno / (8 * sizeof(mp_limb_t_));
    const std::size_t bit = bitno - (8 * sizeof(mp_limb_t_) * part);
    const mp_limb_t_ one = 1;
    return (this->data[part] & (one << bit)) != 0;
}

template<mp_size_t_ n>
__device__ void bigint<n>::set_bit(const std::size_t bitno)
{
    if (bitno >= n * 8 * sizeof(mp_limb_t_))
        return;

    const std::size_t part = bitno / (8 * sizeof(mp_limb_t_));
    const std::size_t bit = bitno - (8 * sizeof(mp_limb_t_) * part);
    const mp_limb_t_ one = 1;
    this->data[part] = one << bit;
    
}

// unfinished
template<mp_size_t_ n>
__device__ bigint<n>& bigint<n>::randomize()
{
    for (size_t i = 0; i < n; i++)
    {
        if (sizeof(mp_limb_t_) == 8)
            this->data[i] = 0x1746567387465673;
        else if (sizeof(mp_limb_t_) == 4)
            this->data[i] = 0x17465673;
    }
    return *this;
}

template<mp_size_t_ n>
__device__ void bigint<n>::print() const
{
    for (size_t i = 0; i < n; i++)
        printf("%016lx ", data[n - 1 - i]);
    printf("\n");
}

}

#endif