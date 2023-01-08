#ifndef __FP2_HOST_CU__
#define __FP2_HOST_CU__

namespace libff {

template<mp_size_t n> 
bool Fp2_model_host<n>::operator==(const Fp2_model_host<n>& other) const
{
    return (this->c0 == other.c0 && this->c1 == other.c1);
}

template<mp_size_t n> 
bool Fp2_model_host<n>::operator!=(const Fp2_model_host<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t n>
Fp2_model_host<n> Fp2_model_host<n>::operator+(const Fp2_model_host<n>& other) const
{
    return Fp2_model_host<n>(params, this->c0 + other.c0, this->c1 + other.c1);
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::operator-(const Fp2_model_host<n>& other) const
{
    return Fp2_model_host<n>(params, this->c0 - other.c0, this->c1 - other.c1);
}

template<mp_size_t n> 
Fp2_model_host<n> operator*(const Fp_model_host<n>& lhs, const Fp2_model_host<n>& rhs)
{
    return Fp2_model_host<n>(rhs.params, lhs * rhs.c0, lhs * rhs.c1);
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::operator*(const Fp2_model_host<n>& other) const
{
    const my_Fp &A = other.c0, &B = other.c1, &a = this->c0, &b = this->c1;
    const my_Fp aA = a * A;
    const my_Fp bB = b * B;

    if (n == 5)
        return Fp2_model_host<n>(params, aA + bB * my_Fp(params->fp_params, *params->non_residue), (a + b) * (A + B) - aA - bB);
    else if(n == 12)
        return Fp2_model_host<n>(params, aA + bB * my_Fp(params->fp_params, *params->non_residue), (a + b) * (A + B) - aA - bB);
    else
        return Fp2_model_host<n>(params, aA - bB, (a + b) * (A + B) - aA - bB);
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::operator-() const
{
    return Fp2_model_host<n>(params, -this->c0, -this->c1);
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::dbl() const
{
    return Fp2_model_host<n>(params, this->c0.dbl(), this->c1.dbl());
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::squared() const
{
    const my_Fp &a = this->c0, &b = this->c1;
    const my_Fp ab = a * b;
    if (n == 5)
        return Fp2_model_host<n>(params, a.squared() + b.squared() * my_Fp(params->fp_params, *params->non_residue), ab.dbl());
    else if(n == 12)
        return Fp2_model_host<n>(params, a.squared() + b.squared() * my_Fp(params->fp_params, *params->non_residue), ab.dbl());
    else
        return Fp2_model_host<n>(params, (a + b) * (a - b), ab.dbl());
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::squared_karatsuba() const
{
    const my_Fp &a = this->c0, &b = this->c1;
    const my_Fp asq = a.squared();
    const my_Fp bsq = b.squared();

    return Fp2_model_host<n>(params, asq - bsq, (a + b).squared() - asq - bsq);
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::squared_complex() const
{
    const my_Fp &a = this->c0, &b = this->c1;
    const my_Fp ab = a * b;

    return Fp2_model_host<n>(params, (a + b) * (a - b), ab.dbl());
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::inverse() const
{
    const my_Fp &a = this->c0, &b = this->c1;

    /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 8 */
    const my_Fp t0 = a.squared();
    const my_Fp t1 = b.squared();
    const my_Fp t2 = (n == 5) ? t0 - t1 * my_Fp(params->fp_params, *params->non_residue) : t0 + t1;
    const my_Fp t3 = t2.inverse();
    const my_Fp c0 = a * t3;
    const my_Fp c1 = - (b * t3);

    return Fp2_model_host<n>(params, c0, c1);
}


template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::Frobenius_map(unsigned long power) const
{
    my_Fp Frobenius_coeffs_c1[2] = { my_Fp(params->fp_params, *params->Frobenius_coeffs_c1[0]), my_Fp(params->fp_params, *params->Frobenius_coeffs_c1[1]) };
    return Fp2_model_host<n>(params, c0, Frobenius_coeffs_c1[power % 2] * c1);
}

template<mp_size_t n> 
void Fp2_model_host<n>::set_params(Fp2_params_host<n>* params)
{
    this->params = params;
    c0.set_params(params->fp_params);
    c1.set_params(params->fp_params);
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::zero() const 
{
    my_Fp t(params->fp_params);
    return Fp2_model_host<n>(params, t.zero(), t.zero());
}

template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::one() const
{
    my_Fp t(params->fp_params);
    return Fp2_model_host<n>(params, t.one(), t.zero());
}


template<mp_size_t n> 
Fp2_model_host<n> Fp2_model_host<n>::random_element()
{
    my_Fp t(params->fp_params);

    Fp2_model_host<n> r(params);
    r.c0 = t.random_element();
    r.c1 = t.random_element();

    return r;
}

template<mp_size_t n>
size_t Fp2_model_host<n>::size_in_bits()
{
    my_Fp t(params->fp_params);
    return 2 * t.size_in_bits();
}


}


#endif