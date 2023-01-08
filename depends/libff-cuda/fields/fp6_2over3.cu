#ifndef __FP6_2OVER3_CU__
#define __FP6_2OVER3_CU__

namespace libff {
    
template<mp_size_t_ n> 
__noinline__ __device__ Fp3_model<n> Fp6_2over3_model<n>::mul_by_non_residue(const Fp3_model<n>& elem) const
{
    Fp_model<n> non_residue(fp3_params, *params.non_residue);
    return Fp3_model<n>(fp3_params, non_residue * elem.c2, elem.c0, elem.c1);
}

template<mp_size_t_ n> 
__noinline__ __device__ bool Fp6_2over3_model<n>::operator==(const Fp6_2over3_model<n>& other) const
{
    return (this->c0 == other.c0 && this->c1 == other.c1);
}

template<mp_size_t_ n> 
__noinline__ __device__ bool Fp6_2over3_model<n>::operator!=(const Fp6_2over3_model<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::operator+(const Fp6_2over3_model<n>& other) const
{
    return Fp6_2over3_model<n>(fp6_params, this->c0 + other.c0, this->c1 + other.c1);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::operator-(const Fp6_2over3_model<n>& other) const
{
    return Fp6_2over3_model<n>(fp6_params, this->c0 - other.c0, this->c1 - other.c1);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> operator*(const Fp_model<n>& lhs, const Fp6_2over3_model<n>& rhs)
{
    return Fp6_2over3_model<n>(fp6_params, lhs * rhs.c0, lhs * rhs.c1);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::operator*(const Fp6_2over3_model<n>& other) const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Karatsuba) */

    const my_Fp3 &B = other.c1, &A = other.c0, &b = this->c1, &a = this->c0;
    const my_Fp3 aA = a*A;
    const my_Fp3 bB = b*B;
    const my_Fp3 beta_bB = this->mul_by_non_residue(bB);

    return Fp6_2over3_model<n>(fp6_params, aA + beta_bB, (a + b) * (A + B) - aA  - bB);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::operator-() const
{
    return Fp6_2over3_model<n>(fp6_params, -this->c0, -this->c1);
}

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp6_2over3_model<n> operator^(const Fp6_2over3_model<n>& self, const bigint<m>& exponent)
{
    return power<Fp6_2over3_model<n>, m>(self, exponent);
}

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp6_2over3_model<n> operator^(const Fp6_2over3_model<n>& self, const Fp_model<m>& exponent)
{
    return self ^ (exponent.as_bigint());
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::mul_by_2345(const Fp6_2over3_model<n>& other) const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Karatsuba) */
    assert(other.c0.c0.is_zero());
    assert(other.c0.c1.is_zero());
    Fp_model<n> non_residue(fp_params, *fp6_params.non_residue);

    const my_Fp3 &B = other.c1, &A = other.c0, &b = this->c1, &a = this->c0;
    const my_Fp3 aA = my_Fp3(fp3_params, a.c1 * A.c2 * non_residue, a.c2 * A.c2 * non_residue, a.c0 * A.c2);
    const my_Fp3 bB = b * B;
    const my_Fp3 beta_bB = this->mul_by_non_residue(bB);

    return Fp6_2over3_model<n>(fp6_params, aA + beta_bB, (a + b) * (A + B) - aA  - bB);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::dbl() const
{
    return Fp6_2over3_model<n>(fp6_params, this->c0.dbl(), this->c1.dbl());
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::squared() const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Complex) */
    const my_Fp3 &b = this->c1, &a = this->c0;
    const my_Fp3 ab = a * b;

    return Fp6_2over3_model<n>(fp6_params, (a + b) * (a + this->mul_by_non_residue(b)) - ab - this->mul_by_non_residue(ab), ab.dbl());
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::inverse() const
{
    /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 8 */

    const my_Fp3 &b = this->c1, &a = this->c0;
    const my_Fp3 t1 = b.squared();
    const my_Fp3 t0 = a.squared() - this->mul_by_non_residue(t1);
    const my_Fp3 new_t1 = t0.inverse();

    return Fp6_2over3_model<n>(fp6_params, a * new_t1, - (b * new_t1));
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::Frobenius_map(unsigned long power) const
{
    my_Fp Frobenius_coeffs_c1[6] = 
    {   
        my_Fp(fp_params, *params.Frobenius_coeffs_c1[0]), my_Fp(fp_params, *params.Frobenius_coeffs_c1[1]), 
        my_Fp(fp_params, *params.Frobenius_coeffs_c1[2]), my_Fp(fp_params, *params.Frobenius_coeffs_c1[3]),
        my_Fp(fp_params, *params.Frobenius_coeffs_c1[4]), my_Fp(fp_params, *params.Frobenius_coeffs_c1[5])
    };

    return Fp6_2over3_model<n>(fp6_params, c0.Frobenius_map(power), Frobenius_coeffs_c1[power % 6] * c1.Frobenius_map(power));
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::unitary_inverse() const
{
    return Fp6_2over3_model<n>(fp6_params, this->c0, -this->c1);
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::cyclotomic_squared() const
{
    my_Fp2 a = my_Fp2(fp2_params, c0.c0, c1.c1);
    //my_Fp a_a = c0.c0; // a = Fp2([c0[0],c1[1]])
    //my_Fp a_b = c1.c1;

    my_Fp2 b = my_Fp2(fp2_params, c1.c0, c0.c2);
    //my_Fp b_a = c1.c0; // b = Fp2([c1[0],c0[2]])
    //my_Fp b_b = c0.c2;

    my_Fp2 c = my_Fp2(fp2_params, c0.c1, c1.c2);
    //my_Fp c_a = c0.c1; // c = Fp2([c0[1],c1[2]])
    //my_Fp c_b = c1.c2;

    my_Fp2 asq = a.squared();
    my_Fp2 bsq = b.squared();
    my_Fp2 csq = c.squared();

    // A = vector(3*a^2 - 2*Fp2([vector(a)[0],-vector(a)[1]]))
    //my_Fp A_a = my_Fp(3l) * asq_a - my_Fp(2l) * a_a;
    my_Fp A_a = asq.c0 - a.c0;
    A_a = A_a + A_a + asq.c0;
    //my_Fp A_b = my_Fp(3l) * asq_b + my_Fp(2l) * a_b;
    my_Fp A_b = asq.c1 + a.c1;
    A_b = A_b + A_b + asq.c1;

    // B = vector(3*Fp2([non_residue*c2[1],c2[0]]) + 2*Fp2([vector(b)[0],-vector(b)[1]]))
    //my_Fp B_a = my_Fp(3l) * my_Fp3::non_residue * csq_b + my_Fp(2l) * b_a;
    my_Fp3 non_residue(fp3_params, *fp3_params.non_residue);
    my_Fp B_tmp = non_residue * csq.c1;
    my_Fp B_a = B_tmp + b.c0;
    B_a = B_a + B_a + B_tmp;

    //my_Fp B_b = my_Fp(3l) * csq_a - my_Fp(2l) * b_b;
    my_Fp B_b = csq.c0 - b.c1;
    B_b = B_b + B_b + csq.c0;

    // C = vector(3*b^2 - 2*Fp2([vector(c)[0],-vector(c)[1]]))
    //my_Fp C_a = my_Fp(3l) * bsq_a - my_Fp(2l) * c_a;
    my_Fp C_a = bsq.c0 - c.c0;
    C_a = C_a + C_a + bsq.c0;
    // my_Fp C_b = my_Fp(3l) * bsq_b + my_Fp(2l) * c_b;
    my_Fp C_b = bsq.c1 + c.c1;
    C_b = C_b + C_b + bsq.c1;

    // e0 = Fp3([A[0],C[0],B[1]])
    // e1 = Fp3([B[0],A[1],C[1]])
    // fin = Fp6e([e0,e1])
    // return fin

    return Fp6_2over3_model<n>(fp6_params, my_Fp3(fp3_params, A_a, C_a, B_b), my_Fp3(fp3_params, B_a, A_b, C_b));
}


template<mp_size_t_ n>
template<mp_size_t_ m>
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::cyclotomic_exp(const bigint<m>& exponent) const
{
    Fp6_2over3_model<n> t(fp6_params);
    Fp6_2over3_model<n> res = t.one();
    Fp6_2over3_model<n> this_inverse = this->unitary_inverse();

    bool found_nonzero = false;

    long* NAF;
    size_t NAF_size;
    find_wnaf(NAF, NAF_size, 1, exponent);

    for (long i = NAF_size - 1; i >= 0; --i)
    {
        if (found_nonzero)
            res = res.cyclotomic_squared();

        if (NAF[i] != 0)
        {
            found_nonzero = true;
            res = NAF[i] > 0 ? res * (*this) : res * this_inverse;
    }

    delete[] NAF;
    return res;
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::zero()
{
    Fp3_model<n> t(p3_params);
    return Fp6_2over3_model<n>(fp6_params, t.zero(), t.zero());
}

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> Fp6_2over3_model<n>::one()
{
    Fp3_model<n> t(p3_params);
    return Fp6_2over3_model<n>(fp6_params, t.one(), t.zero());
}

}

#endif