#ifndef __BIGINT_HOST_CU__
#define __BIGINT_HOST_CU__

#include <cstring>

namespace libff {

template<mp_size_t n>
bigint_host<n>::bigint_host(const unsigned long x)
{
    data[0] = x;
}

template<mp_size_t n>
bigint_host<n>::bigint_host(const char* s)
{
    size_t l = strlen(s);

    unsigned char* s_copy = new unsigned char[l];

    for (size_t i = 0; i < l; ++i)
    {
        s_copy[i] = s[i] - '0';
    }

    mp_size_t limbs_written = mpn_set_str(this->data, s_copy, l, 10);
    delete[] s_copy;
}

template<mp_size_t n>
bigint_host<n>::bigint_host(const mpz_t r)
{
    mpz_t k;
    mpz_init_set(k, r);

    for (size_t i = 0; i < n; ++i)
    {
        data[i] = mpz_get_ui(k);
        mpz_fdiv_q_2exp(k, k, GMP_NUMB_BITS);
    }
    mpz_clear(k);
}

template<mp_size_t n>
bool bigint_host<n>::operator==(const bigint_host<n>& other) const
{
    for (size_t i = 0; i < n; i++)
        if (data[i] != other.data[i])
            return false;
    return true;
}

template<mp_size_t n>
bool bigint_host<n>::operator!=(const bigint_host<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t n>
bool bigint_host<n>::operator>=(const bigint_host<n>& other) const
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

template<mp_size_t n>
bigint_host<n>& bigint_host<n>::operator+=(const bigint_host<n>& other)
{
    bool carry = 0;
    for (size_t i = 0; i < n; i++) 
    {
        mp_limb_t old = data[i];
        data[i] += other.data[i] + carry;
        carry = carry ? old >= data[i] : old > data[i];
    }
    return *this;
}

template<mp_size_t n>
bigint_host<n>& bigint_host<n>::operator-=(const bigint_host<n>& other)
{
    bool borrow = 0;
    for (size_t i = 0; i < n; i++) 
    {
        mp_limb_t old = data[i];
        data[i] -= other.data[i] + borrow;
        borrow = borrow ? old <= data[i] : old < data[i];
    }
    return *this;
}

template<mp_size_t n>
void bigint_host<n>::clear()
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
}


template<mp_size_t n>
bool bigint_host<n>::is_zero() const
{
    for (size_t i = 0; i < n; ++i)
        if (this->data[i])
            return false;

    return true;
}

template<mp_size_t n>
size_t bigint_host<n>::num_bits() const
{
    for (long i = max_bits(); i >= 0; --i)
        if (this->test_bit(i))
            return i + 1;

    return 0;
}

template<mp_size_t n>
unsigned long bigint_host<n>::as_ulong() const
{
    return this->data[0];
}

template<mp_size_t n>
void bigint_host<n>::to_mpz(mpz_t r) const
{
    mpz_set_ui(r, 0);

    for (int i = n - 1; i >= 0; --i)
    {
        mpz_mul_2exp(r, r, 8 * sizeof(mp_limb_t));
        mpz_add_ui(r, r, this->data[i]);
    }
}

template<mp_size_t n>
bool bigint_host<n>::test_bit(const std::size_t bitno) const
{
    if (bitno >= n * 8 * sizeof(mp_limb_t))
        return false;

    const std::size_t part = bitno / (8 * sizeof(mp_limb_t));
    const std::size_t bit = bitno - (8 * sizeof(mp_limb_t) * part);
    const mp_limb_t one = 1;
    return (this->data[part] & (one << bit)) != 0;
}

template<mp_size_t n>
void bigint_host<n>::set_bit(const std::size_t bitno)
{
    if (bitno >= n * 8 * sizeof(mp_limb_t))
        return;

    const std::size_t part = bitno / (8 * sizeof(mp_limb_t));
    const std::size_t bit = bitno - (8 * sizeof(mp_limb_t) * part);
    const mp_limb_t one = 1;
    this->data[part] = one << bit;
}

// unfinished
template<mp_size_t n>
bigint_host<n>& bigint_host<n>::randomize()
{
    for (size_t i = 0; i < n; i++)
    {
        if (sizeof(mp_limb_t) == 8)
            this->data[i] = 0x1746567387465673;
        else if (sizeof(mp_limb_t) == 4)
            this->data[i] = 0x17465673;
    }
    return *this;
}

template<mp_size_t n>
void bigint_host<n>::print() const
{
    for (size_t i = 0; i < n; i++)
        printf("%016lx ", data[n - 1 - i]);
    printf("\n");
}


}


#endif