#ifndef __FP3_CU_
#define __FP3_CU__

#include <assert.h>
#include <stdio.h>

namespace libff {

template<mp_size_t_ n>
__noinline__ __device__ bool Fp3_model<n>::operator==(const Fp3_model<n>& other) const
{
    return (this->c0 == other.c0 && this->c1 == other.c1 && this->c2 == other.c2);
}

template<mp_size_t_ n>
__noinline__ __device__ bool Fp3_model<n>::operator!=(const Fp3_model<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::operator+(Fp3_model<n> other) const
{
    return Fp3_model<n>(fp3_params, this->c0 + other.c0, 
                                    this->c1 + other.c1, 
                                    this->c2 + other.c2);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::operator-(Fp3_model<n> other) const
{
    return Fp3_model<n>(fp3_params, this->c0 - other.c0, 
                                    this->c1 - other.c1, 
                                    this->c2 - other.c2);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> operator*(const Fp_model<n>& lhs, const Fp3_model<n>& rhs)
{
    return Fp3_model<n>(fp3_params, lhs * rhs.c0, 
                                    lhs * rhs.c1, 
                                    lhs * rhs.c2);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::operator*(const Fp3_model<n>& other) const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 4 (Karatsuba) */
    const my_Fp 
        &A = other.c0, &B = other.c1, &C = other.c2,
        &a = this->c0, &b = this->c1, &c = this->c2;

    const my_Fp aA = a * A;
    const my_Fp bB = b * B;
    const my_Fp cC = c * C;
    my_Fp non_residue(fp_params, *fp3_params.non_residue);

    return Fp3_model<n>(fp3_params, aA + non_residue * ((b + c) * (B + C) - bB - cC),
                                    (a + b) * (A + B) - aA - bB + non_residue * cC,
                                    (a + c) * (A + C) - aA + bB - cC);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::operator-() const
{
    return Fp3_model<n>(fp3_params, -this->c0, -this->c1, -this->c2);
}

template<mp_size_t_ n>
template<mp_size_t_ m>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::operator^(const bigint<m>& pow) const
{
    return power<Fp3_model<n>, m>(*this, pow);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::dbl() const
{
    return Fp3_model<n>(fp3_params, this->c0.dbl(), 
                                    this->c1.dbl(), 
                                    this->c2.dbl());
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::squared() const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 4 (CH-SQR2) */
    const my_Fp &a = this->c0, &b = this->c1, &c = this->c2;
    const my_Fp s0 = a.squared();
    const my_Fp ab = a * b;
    const my_Fp s1 = ab + ab;
    const my_Fp s2 = (a - b + c).squared();
    const my_Fp bc = b*c;
    const my_Fp s3 = bc + bc;
    const my_Fp s4 = c.squared();
    my_Fp non_residue(fp_params, *fp3_params.non_residue);

    return Fp3_model<n>(fp3_params, s0 + non_residue * s3,
                                    s1 + non_residue * s4,
                                    s1 + s2 + s3 - s0 - s4);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::inverse() const
{
    const my_Fp &a = this->c0, &b = this->c1, &c = this->c2;
    my_Fp non_residue(fp_params, *fp3_params.non_residue);

    /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 17 */
    const my_Fp t0 = a.squared();
    const my_Fp t1 = b.squared();
    const my_Fp t2 = c.squared();
    const my_Fp t3 = a * b;
    const my_Fp t4 = a * c;
    const my_Fp t5 = b * c;
    const my_Fp c0 = t0 - non_residue * t5;
    const my_Fp c1 = non_residue * t2 - t3;
    const my_Fp c2 = t1 - t4; // typo in paper referenced above. should be "-" as per Scott, but is "*"
    const my_Fp t6 = (a * c0 + non_residue * (c * c1 + b * c2)).inverse();

    return Fp3_model<n>(fp3_params, t6 * c0, t6 * c1, t6 * c2);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::Frobenius_map(unsigned long power) const
{
    my_Fp Frobenius_coeffs_c1[3] = { my_Fp(fp_params, *fp3_params.Frobenius_coeffs_c1[0]), my_Fp(fp_params, *fp3_params.Frobenius_coeffs_c1[1]), my_Fp(fp_params, *fp3_params.Frobenius_coeffs_c1[2]) };
    my_Fp Frobenius_coeffs_c2[3] = { my_Fp(fp_params, *fp3_params.Frobenius_coeffs_c2[0]), my_Fp(fp_params, *fp3_params.Frobenius_coeffs_c2[1]), my_Fp(fp_params, *fp3_params.Frobenius_coeffs_c2[2]) };

    return Fp3_model<n>(fp3_params, c0,
                                    Frobenius_coeffs_c1[power % 3] * c1,
                                    Frobenius_coeffs_c2[power % 3] * c2);
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::sqrt() const
{
    Fp3_model<n> t(fp3_params);
    Fp3_model<n> one = t.one();

    size_t v = fp3_params.s;
    Fp3_model<n> z(fp3_params, *fp3_params.nqr_to_t_c0, *fp3_params.nqr_to_t_c1, *fp3_params.nqr_to_t_c2);

    Fp3_model<n> w = (*this) ^ (*fp3_params.t_minus_1_over_2);
    Fp3_model<n> x = (*this) * w;
    Fp3_model<n> b = x * w; // b = (*this)^t

    // check if square with euler's criterion
    Fp3_model<n> check = b;
    for (size_t i = 0; i < v-1; ++i)
        check = check.squared();
    
    if (check != one)
        assert(0);

    // compute square root with Tonelli--Shanks
    // (does not terminate if not a square!)

    while (b != one)
    {
        size_t m = 0;
        Fp3_model<n> b2m = b;
        while (b2m != one)
        {
            /* invariant: b2m = b^(2^m) after entering this loop */
            b2m = b2m.squared();
            m += 1;
        }

        int j = v-m-1;
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
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::zero()
{
    my_Fp t(fp_params);
    return Fp3_model<n>(fp3_params, t.zero(), t.zero(), t.zero());
}

template<mp_size_t_ n>
__noinline__ __device__ Fp3_model<n> Fp3_model<n>::one()
{
    my_Fp t(fp_params);
    return Fp3_model<n>(fp3_params, t.one(), t.zero(), t.zero());
}

template<mp_size_t_ n>
__noinline__ __device__ size_t Fp3_model<n>::size_in_bits()
{
    my_Fp t(fp_params);
    return 3 * t.size_in_bits();
}

}

#endif