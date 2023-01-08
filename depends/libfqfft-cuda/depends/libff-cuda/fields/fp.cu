#ifndef __FP_CU__
#define __FP_CU__

#include <assert.h>
#include <stdio.h>

namespace libff {

__device__ inline ulong mac_with_carry(ulong a, ulong b, ulong c, ulong *d) 
{
    ulong lo, hi;
    asm
    (
        "mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64   %1, %2, %3,  0;\r\n"
        "add.cc.u64    %0, %0, %5;    \r\n"
        "addc.u64      %1, %1,  0;    \r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d)
    );

    *d = hi;
    return lo;
}

// Returns a + b, puts the carry in d
__device__ inline ulong add_with_carry(ulong a, ulong *b) 
{
    ulong lo, hi;
    asm
    ( 
        "add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64   %1,  0,  0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b)
    );

    *b = hi;
    return lo;
}

template<mp_size_t_ n>
__device__ Fp_model<n>::Fp_model(Fp_params<n>* params, const bigint<n>& b) : params(params), mont_repr(*params->Rsquared)
{
    *this *= b;
}

template<mp_size_t_ n>
__device__ Fp_model<n>::Fp_model(Fp_params<n>* params, const long x, const bool is_unsigned) : params(params)
{
    mont_repr.clear();
    if (is_unsigned || x >= 0)
        mont_repr.data[0] = (mp_limb_t_)x;
    else
        mpn_sub__1_(this->mont_repr.data, params->modulus->data, n, (mp_limb_t_)-x);

    *this *= *params->Rsquared;
}

template<mp_size_t_ n>
__device__ bool Fp_model<n>::operator==(const Fp_model<n>& other) const
{
    return (this->mont_repr == other.mont_repr);
}

template<mp_size_t_ n>
__device__ bool Fp_model<n>::operator!=(const Fp_model<n>& other) const
{
    return (this->mont_repr != other.mont_repr);
}

template<mp_size_t_ n>
__device__ Fp_model<n>& Fp_model<n>::operator+=(const Fp_model<n>& other)
{
    mont_repr += other.mont_repr;
    if (mont_repr >= *params->modulus)
        mont_repr -= *params->modulus;
    return *this;
}

template<mp_size_t_ n>
__device__ Fp_model<n>& Fp_model<n>::operator-=(const Fp_model<n>& other)
{
    if (mont_repr >= other.mont_repr)
        mont_repr -= other.mont_repr;
    else
        mont_repr -= other.mont_repr, mont_repr += *params->modulus;
    return *this;
}
   
template<mp_size_t_ n>
__device__ Fp_model<n>& Fp_model<n>::operator*=(const bigint<n>& other)
{
    mp_limb_t_ t[n + 2] = { 0 };

    for (size_t i = 0; i < n; i++) 
    {
        mp_limb_t_ carry = 0;
        for (size_t j = 0; j < n; j++)
            t[j] = mac_with_carry(mont_repr.data[j], other.data[i], t[j], &carry);

        t[n] = add_with_carry(t[n], &carry);
        t[n + 1] = carry;

        carry = 0;
        mp_limb_t_ m = params->inv * t[0];
        mac_with_carry(m, params->modulus->data[0], t[0], &carry);

        for (size_t j = 1; j < n; j++)
            t[j - 1] = mac_with_carry(m, params->modulus->data[j], t[j], &carry);

        t[n - 1] = add_with_carry(t[n], &carry);
        t[n] = t[n + 1] + carry;
    }

    for (size_t i = 0; i < n; i++) 
        mont_repr.data[i] = t[i];

    if (mont_repr >= *params->modulus) 
        mont_repr -= *params->modulus;

    return *this;
}

template<mp_size_t_ n>
__device__ Fp_model<n>& Fp_model<n>::operator*=(const Fp_model<n>& other)
{
    return *this *= other.mont_repr;
}

template<mp_size_t_ n>
__noinline__ __device__ Fp_model<n>& Fp_model<n>::operator^=(const unsigned long pow)
{
    (*this) = power<Fp_model<n>>(*this, pow);
    return (*this);
}

template<mp_size_t_ n>
template<mp_size_t_ m>
__noinline__ __device__ Fp_model<n>& Fp_model<n>::operator^=(const bigint<m> pow)
{
    (*this) = power<Fp_model<n>, m>(*this, pow);
    return (*this);
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::operator+(Fp_model<n> other) const
{
    Fp_model<n> r(*this);
    return (r += other);
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::operator-(Fp_model<n> other) const
{
    Fp_model<n> r(*this);
    return (r -= other);
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::operator*(const Fp_model<n>& other) const
{
    Fp_model<n> r(*this);
    return (r *= other);
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::operator-() const
{
    if (this->is_zero())
        return (*this);
    
    Fp_model<n> r(params);
    r.mont_repr = *params->modulus;
    r.mont_repr -= this->mont_repr;
    return r;
}

template<mp_size_t_ n>
__noinline__ __device__ Fp_model<n> Fp_model<n>::operator^(const unsigned long pow) const
{
    Fp_model<n> r(*this);
    return (r ^= pow);
}

template<mp_size_t_ n>
template<mp_size_t_ m>
__noinline__ __device__ Fp_model<n> Fp_model<n>::operator^(const bigint<m>& pow) const
{
    Fp_model<n> r(*this);
    return (r ^= pow);
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::dbl() const
{
    Fp_model<n> res(params);

    for (size_t i = n - 1; i >= 1; i--)
        res.mont_repr.data[i] = (mont_repr.data[i] << 1) | (mont_repr.data[i - 1] >> (64 - 1));
    
    res.mont_repr.data[0] = mont_repr.data[0] << 1;

    if (res.mont_repr >= *params->modulus)
        res.mont_repr -= *params->modulus;
    return res;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::squared() const
{
    return (*this) * (*this);
}

template<mp_size_t_ n>
__device__ Fp_model<n>& Fp_model<n>::invert()
{
    // assert(!this->is_zero());
    mpz_t_ zg;
    // mpz_init_(zg);
    mpz_init_2_(zg, (n + 1) * 8 * sizeof(mp_limb_t_));

    mpz_t_ zs;
    // mpz_init_(zs);
    mpz_init_2_(zs, (n + 1) * 8 * sizeof(mp_limb_t_));

    mpz_t_ zv;
    // mpz_init_(zv);
    mpz_init_2_(zv, (n + 1) * 8 * sizeof(mp_limb_t_));
    params->modulus->to_mpz(zv);

    mpz_t_ zu;
    // mpz_init_(zu);
    mpz_init_2_(zu, (n + 1) * 8 * sizeof(mp_limb_t_));
    mont_repr.to_mpz(zu);

    mpz_gcd_ext(zg, zs, NULL, zu, zv);
    // assert(zg->_mp_size == 1 && zg->_mp_d[0] == 1); /* inverse exists */

    // mp_limb_t_ q; /* division result fits into q, as sn <= n+1 */
    /* sn < 0 indicates negative sn; will fix up later */

    mpz_t_ zr;
    // mpz_init_(zr);
    mpz_init_2_(zr, ( n+ 1) * 8 * sizeof(mp_limb_t_));

    mpz_t_div_q_r_(NULL, zr, zs, zv);
    mont_repr.clear();
    mpn_copyi_(mont_repr.data, zr->_mp_d, abs(zr->_mp_size));

    /* fix up the negative sn */
    if (zr->_mp_size < 0)
        mpn_sub__n_(this->mont_repr.data, params->modulus->data, this->mont_repr.data, n);
        // assert(borrow == 0);

    mpz_clear_(zg);
    mpz_clear_(zs);
    mpz_clear_(zv);
    mpz_clear_(zu);
    mpz_clear_(zr);

    *this *= *params->Rcubed;
    return *this;
}


template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::inverse() const
{
    Fp_model<n> r(*this);
    return (r.invert());
}

template<mp_size_t_ n>
__host__ Fp_model<n>* Fp_model<n>::inverse_host()
{
    Fp_model<n>* r = libstl::create_host<Fp_model<n>>(*this);
    libstl::launch<<<1, 1>>>
    (
        [=] 
        __device__ () 
        {
            r->invert();
        } 
    );
    cudaStreamSynchronize(0);

    return r;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::sqrt() const
{
    Fp_model<n> one = this->one();

    size_t v = params->s;
    Fp_model<n> z(params, *params->nqr_to_t);
    Fp_model<n> w = (*this) ^ (*params->t_minus_1_over_2);
    Fp_model<n> x = (*this) * w;
    Fp_model<n> b = x * w;

    Fp_model<n> check = b;
    for (size_t i = 0; i < v - 1; ++i)
        check = check.squared();
    
    if (check != one)
        assert(0);

    // compute square root with Tonelli--Shanks
    // (does not terminate if not a square!)

    while (b != one)
    {
        size_t m = 0;
        Fp_model<n> b2m = b;
        while (b2m != one)
        {
            /* invariant: b2m = b^(2^m) after entering this loop */
            b2m = b2m.squared();
            m += 1;
        }

        int j = v - m - 1;
        w = z;
        while (j > 0)
        {
            w = w.squared();
            --j;
        } // w = z^2^(v-m-1)

        z = w.squared();
        b = b * z;
        x = x * w;
        v = m;
    }

    return x;
}

template<mp_size_t_ n>
__device__ void Fp_model<n>::init_zero()
{
    mont_repr.clear();
}

template<mp_size_t_ n>
__device__ void Fp_model<n>::init_one()
{
    mont_repr.clear();
    mont_repr.data[0] = 1;
    *this *= *params->Rsquared;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::zero() const 
{
    return *params->zero;
}

template<mp_size_t_ n>
__host__ Fp_model<n>* Fp_model<n>::zero_host()
{
    Fp_params<n>* params_addr;
    cudaMemcpy(&params_addr, &this->params, sizeof(Fp_params<n>*), cudaMemcpyDeviceToHost);
    Fp_model<n>* zero;
    cudaMemcpy(&zero, &params_addr->zero, sizeof(Fp_model<n>*), cudaMemcpyDeviceToHost);

    return zero;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::one() const 
{
    return *params->one;
}

template<mp_size_t_ n>
__host__ Fp_model<n>* Fp_model<n>::one_host()
{
    Fp_params<n>* params_addr;
    cudaMemcpy(&params_addr, &this->params, sizeof(Fp_params<n>*), cudaMemcpyDeviceToHost);
    Fp_model<n>* one;
    cudaMemcpy(&one, &params_addr->one, sizeof(Fp_model<n>*), cudaMemcpyDeviceToHost);

    return one;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::random_element() const
{
    /* note that as Montgomery representation is a bijection then
       selecting a random element of {xR} is the same as selecting a
       random element of {x} */

//     Fp_model<n> r(params);
//     static const mp_size_t_ GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;

//     do
//     {
//         r.mont_repr.randomize();

//         /* clear all bits higher than MSB of modulus */
//         size_t bitno = GMP_NUMB_BITS_ * n - 1;
//         while (params->modulus->test_bit(bitno) == false)
//         {
//             const size_t part = bitno/GMP_NUMB_BITS_;
//             const size_t bit = bitno - (GMP_NUMB_BITS_*part);

//             r.mont_repr.data[part] &= ~(1ul<<bit);

//             bitno--;
//         }
//     }
//    /* if r.data is still >= modulus -- repeat (rejection sampling) */
//     while (mpn_cmp_(r.mont_repr.data, params->modulus->data, n) >= 0);

//     return r;

    Fp_model<n> r(params);
    r.mont_repr.randomize();
    r *= *params->Rsquared;
    return r;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::geometric_generator()
{
    Fp_model<n> res(params);
    res.mont_repr.data[0] = 2;
    res *= *params->Rsquared;
    return res;
}

template<mp_size_t_ n>
__device__ Fp_model<n> Fp_model<n>::arithmetic_generator()
{
    Fp_model<n> res(params);
    res.mont_repr.data[0] = 1;
    res *= *params->Rsquared;
    return res;
}

template<mp_size_t_ n>
__device__ void Fp_model<n>::set_ulong(const unsigned long x)
{
    this->mont_repr.clear();
    this->mont_repr.data[0] = x;
    *this *= *params->Rsquared;
}

template<mp_size_t_ n>
__device__ bigint<n> Fp_model<n>::as_bigint() const
{
    bigint<n> one;
    one.clear();
    one.data[0] = 1;
    
    Fp_model<n> res(*this);
    res *= one;
    return res.mont_repr;
}

template<mp_size_t_ n>
__device__ unsigned long Fp_model<n>::as_ulong() const
{
    return as_bigint().as_ulong();
}

template<mp_size_t_ n>
__device__ bool Fp_model<n>::is_zero() const
{
    return mont_repr.is_zero(); 
}

template<mp_size_t_ n>
__device__ void Fp_model<n>::clear()
{
    mont_repr.clear();
}

template<mp_size_t_ n>
__device__ void Fp_model<n>::set_params(Fp_params<n>* params)
{
    this->params = params;
}

}

#endif