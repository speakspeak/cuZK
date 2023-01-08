#ifndef __FP2_CU__
#define __FP2_CU__

#include <assert.h>
#include <stdio.h>

namespace libff {

template<mp_size_t_ n> 
__device__ bool Fp2_model<n>::operator==(const Fp2_model<n>& other) const
{
    return (this->c0 == other.c0 && this->c1 == other.c1);
}

template<mp_size_t_ n> 
__device__ bool Fp2_model<n>::operator!=(const Fp2_model<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t_ n>
__device__ Fp2_model<n> Fp2_model<n>::operator+(const Fp2_model<n>& other) const
{
    return Fp2_model<n>(params, this->c0 + other.c0, this->c1 + other.c1);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::operator-(const Fp2_model<n>& other) const
{
    return Fp2_model<n>(params, this->c0 - other.c0, this->c1 - other.c1);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> operator*(const Fp_model<n>& lhs, const Fp2_model<n>& rhs)
{
    return Fp2_model<n>(rhs.params, lhs * rhs.c0, lhs * rhs.c1);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::operator*(const Fp2_model<n>& other) const
{
    const my_Fp &A = other.c0, &B = other.c1, &a = this->c0, &b = this->c1;
    const my_Fp aA = a * A;
    const my_Fp bB = b * B;
    
    if (n == 5)
        return Fp2_model<n>(params, aA + bB * my_Fp(params->fp_params, *params->non_residue), (a + b) * (A + B) - aA - bB);
    else if(n == 12)
        return Fp2_model<n>(params, aA + bB * my_Fp(params->fp_params, *params->non_residue), (a + b) * (A + B) - aA - bB);
    else
        return Fp2_model<n>(params, aA - bB, (a + b) * (A + B) - aA - bB);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::operator-() const
{
    return Fp2_model<n>(params, -this->c0, -this->c1);
}

template<mp_size_t_ n>
template<mp_size_t_ m>
__noinline__ __device__ Fp2_model<n> Fp2_model<n>::operator^(const bigint<m>& pow) const
{
    return power<Fp2_model<n>, m>(*this, pow);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::dbl() const
{
    return Fp2_model<n>(params, this->c0.dbl(), this->c1.dbl());
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::squared() const
{
    const my_Fp &a = this->c0, &b = this->c1;
    const my_Fp ab = a * b;

    if (n == 5)
        return Fp2_model<n>(params, a.squared() + b.squared() * my_Fp(params->fp_params, *params->non_residue), ab.dbl());
    else if(n == 12)
        return Fp2_model<n>(params, a.squared() + b.squared() * my_Fp(params->fp_params, *params->non_residue), ab.dbl());
    else
        return Fp2_model<n>(params, (a + b) * (a - b), ab.dbl());
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::squared_karatsuba() const
{
    const my_Fp &a = this->c0, &b = this->c1;
    const my_Fp asq = a.squared();
    const my_Fp bsq = b.squared();

    return Fp2_model<n>(params, asq - bsq, (a + b).squared() - asq - bsq);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::squared_complex() const
{
    const my_Fp &a = this->c0, &b = this->c1;
    const my_Fp ab = a * b;

    return Fp2_model<n>(params, (a + b) * (a - b), ab.dbl());
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::inverse() const
{
    const my_Fp &a = this->c0, &b = this->c1;

    /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 8 */
    const my_Fp t0 = a.squared();
    const my_Fp t1 = b.squared();
    const my_Fp t2 = (n == 5) ? t0 - t1 * my_Fp(params->fp_params, *params->non_residue) : t0 + t1;
    const my_Fp t3 = t2.inverse();
    const my_Fp c0 = a * t3;
    const my_Fp c1 = - (b * t3);

    return Fp2_model<n>(params, c0, c1);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::Frobenius_map(unsigned long power) const
{
    my_Fp Frobenius_coeffs_c1[2] = { my_Fp(params->fp_params, *params->Frobenius_coeffs_c1[0]), my_Fp(params->fp_params, *params->Frobenius_coeffs_c1[1]) };
    return Fp2_model<n>(params, c0, Frobenius_coeffs_c1[power % 2] * c1);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::sqrt() const  // HAS TO BE A SQUARE (else does not terminate)
{
    Fp2_model<n> t(params);
    Fp2_model<n> one = t.one();

    size_t v = params->s;
    Fp2_model<n> z(params, *params->nqr_to_t_c0, *params->nqr_to_t_c1);
    Fp2_model<n> w = (*this) ^ (*params->t_minus_1_over_2);
    Fp2_model<n> x = (*this) * w;
    Fp2_model<n> b = x * w; // b = (*this)^t
    
    // check if square with euler's criterion
    Fp2_model<n> check = b;
    for (size_t i = 0; i < v - 1; ++i)
        check = check.squared();
    
    if (check != one)
        assert(0);

    // compute square root with Tonelli--Shanks
    // (does not terminate if not a square!)
    while (b != one)
    {
        size_t m = 0;
        Fp2_model<n> b2m = b;
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
__device__ void Fp2_model<n>::set_params(Fp2_params<n>* params)
{
    this->params = params;
    c0.set_params(params->fp_params);
    c1.set_params(params->fp_params);
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::zero() const 
{
    my_Fp t(params->fp_params);
    return Fp2_model<n>(params, t.zero(), t.zero());
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::one() const
{
    my_Fp t(params->fp_params);
    return Fp2_model<n>(params, t.one(), t.zero());
}

template<mp_size_t_ n> 
__device__ Fp2_model<n> Fp2_model<n>::random_element()
{
    my_Fp t(params->fp_params);

    Fp2_model<n> r(params);
    r.c0 = t.random_element();
    r.c1 = t.random_element();

    return r;
}

template<mp_size_t_ n>
__device__ size_t Fp2_model<n>::size_in_bits()
{
    my_Fp t(params->fp_params);
    return 2 * t.size_in_bits();
}

}

#endif