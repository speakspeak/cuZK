#ifndef __FP_HOST_CU__
#define __FP_HOST_CU__

namespace libff {

template<mp_size_t n>
void Fp_model_host<n>::mul_reduce(const bigint_host<n> &other)
{
    mp_limb_t res[2*n];
    mpn_mul_n(res, this->mont_repr.data, other.data, n);

    for (size_t i = 0; i < n; ++i)
    {
        mp_limb_t k = params->inv * res[i];

        mp_limb_t carryout = mpn_addmul_1(res+i, params->modulus->data, n, k);
        carryout = mpn_add_1(res+n+i, res+n+i, n-i, carryout);

    }

    if (mpn_cmp(res+n, params->modulus->data, n) >= 0)
    {
        const mp_limb_t borrow = mpn_sub(res+n, res+n, n, params->modulus->data, n);
    }
    mpn_copyi(this->mont_repr.data, res+n, n);
}



template<mp_size_t n>
Fp_model_host<n>::Fp_model_host(Fp_params_host<n>* params, const bigint_host<n>& b) : params(params)
{
    mpn_copyi(this->mont_repr.data, params->Rsquared->data, n);
    mul_reduce(b);
}

template<mp_size_t n>
Fp_model_host<n>::Fp_model_host(Fp_params_host<n>* params, const long x, const bool is_unsigned) : params(params)
{
    if (is_unsigned || x >= 0)
        mont_repr.data[0] = (mp_limb_t)x;
    else
    {
        const mp_limb_t borrow = mpn_sub_1(this->mont_repr.data, params->modulus->data, n, (mp_limb_t)-x);
    }
    
    mul_reduce(*params->Rsquared);
}

template<mp_size_t n>
bool Fp_model_host<n>::operator==(const Fp_model_host<n>& other) const
{
    return (this->mont_repr == other.mont_repr);
}

template<mp_size_t n>
bool Fp_model_host<n>::operator!=(const Fp_model_host<n>& other) const
{
    return (this->mont_repr != other.mont_repr);
}

template<mp_size_t n>
Fp_model_host<n>& Fp_model_host<n>::operator+=(const Fp_model_host<n>& other)
{
    mp_limb_t scratch[n+1];
    const mp_limb_t carry = mpn_add_n(scratch, this->mont_repr.data, other.mont_repr.data, n);
    scratch[n] = carry;

    if (carry || mpn_cmp(scratch, params->modulus->data, n) >= 0)
    {
        const mp_limb_t borrow = mpn_sub(scratch, scratch, n+1, params->modulus->data, n);
    }

    mpn_copyi(this->mont_repr.data, scratch, n);
    return *this;
}

template<mp_size_t n>
Fp_model_host<n>& Fp_model_host<n>::operator-=(const Fp_model_host<n>& other)
{
    mp_limb_t scratch[n+1];
    if (mpn_cmp(this->mont_repr.data, other.mont_repr.data, n) < 0)
    {
        const mp_limb_t carry = mpn_add_n(scratch, this->mont_repr.data, params->modulus->data, n);
        scratch[n] = carry;
    }
    else
    {
        mpn_copyi(scratch, this->mont_repr.data, n);
        scratch[n] = 0;
    }
    const mp_limb_t borrow = mpn_sub(scratch, scratch, n+1, other.mont_repr.data, n);
    mpn_copyi(this->mont_repr.data, scratch, n);
    return *this;
}
   
template<mp_size_t n>
Fp_model_host<n>& Fp_model_host<n>::operator*=(const bigint_host<n>& other)
{
    mul_reduce(other);
    return *this;
}

template<mp_size_t n>
Fp_model_host<n>& Fp_model_host<n>::operator*=(const Fp_model_host<n>& other)
{
    return *this *= other.mont_repr;
}



template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::operator+(Fp_model_host<n> other) const
{
    Fp_model_host<n> r(*this);
    return (r += other);
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::operator-(Fp_model_host<n> other) const
{
    Fp_model_host<n> r(*this);
    return (r -= other);
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::operator*(const Fp_model_host<n>& other) const
{
    Fp_model_host<n> r(*this);
    return (r *= other);
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::operator-() const
{
    if (this->is_zero())
    {
        return (*this);
    }
    else
    {
        Fp_model_host<n> r;
        mpn_sub_n(r.mont_repr.data, params->modulus->data, this->mont_repr.data, n);
        return r;
    }
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::dbl() const
{
    return (*this) + (*this);
}


template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::squared() const
{
    return (*this) * (*this);
}

template<mp_size_t n>
Fp_model_host<n>& Fp_model_host<n>::invert()
{
    bigint_host<n> g; /* gp should have room for vn = n limbs */
    
    mp_limb_t s[n+1]; /* sp should have room for vn+1 limbs */
    mp_size_t sn;

    bigint_host<n> v = *params->modulus; // both source operands are destroyed by mpn_gcdext

    /* computes gcd(u, v) = g = u*s + v*t, so s*u will be 1 (mod v) */
    const mp_size_t gn = mpn_gcdext(g.data, s, &sn, this->mont_repr.data, n, v.data, n);

    mp_limb_t q; /* division result fits into q, as sn <= n+1 */
    /* sn < 0 indicates negative sn; will fix up later */

    if (std::abs(sn) >= n)
    {
        /* if sn could require modulus reduction, do it here */
        mpn_tdiv_qr(&q, this->mont_repr.data, 0, s, std::abs(sn), params->modulus->data, n);
    }
    else
    {
        /* otherwise just copy it over */
        mpn_zero(this->mont_repr.data, n);
        mpn_copyi(this->mont_repr.data, s, std::abs(sn));
    }

    /* fix up the negative sn */
    if (sn < 0)
    {
        const mp_limb_t borrow = mpn_sub_n(this->mont_repr.data, params->modulus->data, this->mont_repr.data, n);
    }

    mul_reduce(*params->Rcubed);
    return *this;
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::inverse() const
{
    Fp_model_host<n> r(*this);
    return (r.invert());
}



template<mp_size_t n>
void Fp_model_host<n>::init_zero()
{
    mont_repr.clear();
}

template<mp_size_t n>
void Fp_model_host<n>::init_one()
{
    mont_repr.clear();
    mont_repr.data[0] = 1;
    *this *= *params->Rsquared;
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::zero() const 
{
    return *params->zero;
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::one() const 
{
    return *params->one;
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::random_element() const
{
    Fp_model_host<n> r(params);
    r.mont_repr.randomize();
    r *= *params->Rsquared;
    return r;
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::geometric_generator()
{
    Fp_model_host<n> res(params);
    res.mont_repr.data[0] = 2;
    res *= *params->Rsquared;
    return res;
}

template<mp_size_t n>
Fp_model_host<n> Fp_model_host<n>::arithmetic_generator()
{
    Fp_model_host<n> res(params);
    res.mont_repr.data[0] = 1;
    res *= *params->Rsquared;
    return res;
}


template<mp_size_t n>
void Fp_model_host<n>::set_ulong(const unsigned long x)
{
    this->mont_repr.clear();
    this->mont_repr.data[0] = x;
    *this *= *params->Rsquared;
}

template<mp_size_t n>
bigint_host<n> Fp_model_host<n>::as_bigint() const
{
    bigint_host<n> one;
    one.data[0] = 1;
    
    Fp_model_host<n> res(*this);
    res *= one;
    return res.mont_repr;
}

template<mp_size_t n>
unsigned long Fp_model_host<n>::as_ulong() const
{
    return as_bigint().as_ulong();
}

template<mp_size_t n>
bool Fp_model_host<n>::is_zero() const
{
    return mont_repr.is_zero(); 
}

template<mp_size_t n>
void Fp_model_host<n>::clear()
{
    mont_repr.clear();
}

template<mp_size_t n>
void Fp_model_host<n>::set_params(Fp_params_host<n>* params)
{
    this->params = params;
}





}


#endif