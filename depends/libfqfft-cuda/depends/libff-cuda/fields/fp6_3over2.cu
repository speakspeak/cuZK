#ifndef __FP6_3OVER2_CU__
#define __FP6_3OVER2_CU__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"
#include "fp2.cuh"

namespace libff {

template<mp_size_t_ n> 
__noinline__ __device__ bool Fp6_3over2_model<n>::operator==(const Fp6_3over2_model<n>& other) const
{
    return (this->c0 == other.c0 && this->c1 == other.c1 && this->c2 == other.c2);
}

template<mp_size_t_ n> 
__noinline__ __device__ bool Fp6_3over2_model<n>::operator!=(const Fp6_3over2_model<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::operator+(const Fp6_3over2_model<n>& other) const
{
    return Fp6_3over2_model<n>(params, this->c0 + other.c0, 
                                       this->c1 + other.c1, 
                                       this->c2 + other.c2);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::operator-(const Fp6_3over2_model<n>& other) const
{
    return Fp6_3over2_model<n>(params, this->c0 - other.c0, 
                                       this->c1 - other.c1, 
                                       this->c2 - other.c2);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> operator*(const Fp_model<n>& lhs, const Fp6_3over2_model<n>& rhs)
{
    return Fp6_3over2_model<n>(rhs.params, lhs * rhs.c0, 
                                           lhs * rhs.c1,
                                           lhs * rhs.c2);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> operator*(const Fp2_model<n>& lhs, const Fp6_3over2_model<n>& rhs)
{
    return Fp6_3over2_model<n>(rhs.params, lhs * rhs.c0, 
                                           lhs * rhs.c1, 
                                           lhs * rhs.c2);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::operator*(const Fp6_3over2_model<n>& other) const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 4 (Karatsuba) */

    const my_Fp2 &A = other.c0, &B = other.c1, &C = other.c2,
                 &a = this->c0, &b = this->c1, &c = this->c2;
    const my_Fp2 aA = a*A;
    const my_Fp2 bB = b*B;
    const my_Fp2 cC = c*C;

    return Fp6_3over2_model<n>(params, aA + this->mul_by_non_residue((b + c) * (B + C) - bB - cC),
                                       (a + b) * (A + B) - aA - bB + this->mul_by_non_residue(cC),
                                       (a + c) * (A + C) - aA + bB - cC);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::operator-() const
{
    return Fp6_3over2_model<n>(params, -this->c0, -this->c1, -this->c2);
}

template<mp_size_t_ n> 
template<mp_size_t_ m>
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::operator^(const bigint<m>& pow) const
{
    return power<Fp6_3over2_model<n>, m>(*this, pow);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp2_model<n> Fp6_3over2_model<n>::mul_by_non_residue(const Fp2_model<n>& elt) const 
{
    Fp2_model<n> non_residue(params->fp2_params, *params->non_residue_c0, *params->non_residue_c1);
    return Fp2_model<n>(non_residue * elt);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::dbl() const
{
    return Fp6_3over2_model<n>(params, this->c0.dbl(), 
                                       this->c1.dbl(), 
                                       this->c2.dbl());
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::squared() const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 4 (CH-SQR2) */

    const my_Fp2 &a = this->c0, &b = this->c1, &c = this->c2;
    const my_Fp2 s0 = a.squared();
    const my_Fp2 ab = a*b;
    const my_Fp2 s1 = ab + ab;
    const my_Fp2 s2 = (a - b + c).squared();
    const my_Fp2 bc = b*c;
    const my_Fp2 s3 = bc + bc;
    const my_Fp2 s4 = c.squared();

    return Fp6_3over2_model<n>(params, s0 + this->mul_by_non_residue(s3),
                                       s1 + this->mul_by_non_residue(s4),
                                       s1 + s2 + s3 - s0 - s4);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::inverse() const
{
    /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 17 */
    const my_Fp2 &a = this->c0, &b = this->c1, &c = this->c2;
    const my_Fp2 t0 = a.squared();
    const my_Fp2 t1 = b.squared();
    const my_Fp2 t2 = c.squared();
    const my_Fp2 t3 = a * b;
    const my_Fp2 t4 = a * c;
    const my_Fp2 t5 = b * c;
    const my_Fp2 c0 = t0 - this->mul_by_non_residue(t5);
    const my_Fp2 c1 = this->mul_by_non_residue(t2) - t3;
    const my_Fp2 c2 = t1 - t4; // typo in paper referenced above. should be "-" as per Scott, but is "*"
    const my_Fp2 t6 = (a * c0 + this->mul_by_non_residue((c * c1 + b * c2))).inverse();
    return Fp6_3over2_model<n>(params, t6 * c0, t6 * c1, t6 * c2);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::Frobenius_map(unsigned long power) const
{
    my_Fp2 Frobenius_coeffs_c1[6] = 
    {
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c1_c0[0], *params->Frobenius_coeffs_c1_c1[0]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c1_c0[1], *params->Frobenius_coeffs_c1_c1[1]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c1_c0[2], *params->Frobenius_coeffs_c1_c1[2]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c1_c0[3], *params->Frobenius_coeffs_c1_c1[3]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c1_c0[4], *params->Frobenius_coeffs_c1_c1[4]),  
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c1_c0[5], *params->Frobenius_coeffs_c1_c1[5])
    };

    my_Fp2 Frobenius_coeffs_c2[6] = 
    {
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c2_c0[0], *params->Frobenius_coeffs_c2_c1[0]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c2_c0[1], *params->Frobenius_coeffs_c2_c1[1]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c2_c0[2], *params->Frobenius_coeffs_c2_c1[2]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c2_c0[3], *params->Frobenius_coeffs_c2_c1[3]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c2_c0[4], *params->Frobenius_coeffs_c2_c1[4]),
        my_Fp2(params->fp2_params, *params->Frobenius_coeffs_c2_c0[5], *params->Frobenius_coeffs_c2_c1[5])
    };                                

    return Fp6_3over2_model<n>(params, c0.Frobenius_map(power),
                                       Frobenius_coeffs_c1[power % 6] * c1.Frobenius_map(power),
                                       Frobenius_coeffs_c2[power % 6] * c2.Frobenius_map(power));
}


template<mp_size_t_ n> 
__noinline__ __device__ void Fp6_3over2_model<n>::set_params(Fp6_3over2_params<n>* params)
{
    this->params = params;
    this->c0.set_params(params->fp2_params);
    this->c1.set_params(params->fp2_params);
    this->c2.set_params(params->fp2_params);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::zero()
{
    Fp2_model<n> t(params->fp2_params);
    return Fp6_3over2_model<n>(params, t.zero(), t.zero(), t.zero());
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::one()
{
    Fp2_model<n> t(params->fp2_params);
    return Fp6_3over2_model<n>(params, t.one(), t.zero(), t.zero());
}


template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> Fp6_3over2_model<n>::random_element()
{
    Fp2_model<n> t(params->fp2_params);
    Fp6_3over2_model<n> r(params);
    r.c0 = t.random_element();
    r.c1 = t.random_element();
    r.c2 = t.random_element();

    return r;
}

}

#endif