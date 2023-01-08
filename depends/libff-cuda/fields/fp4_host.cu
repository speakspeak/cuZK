#ifndef __FP4_HOST_CU__
#define __FP4_HOST_CU__

namespace libff {

template<mp_size_t n> 
Fp2_model_host<n> Fp4_model_host<n>::mul_by_non_residue(const Fp2_model_host<n>& elt) const 
{
    Fp_model_host<n> non_residue(params->fp2_params->fp_params, *params->non_residue);
    return Fp2_model_host<n>(params->fp2_params, non_residue * elt.c1, elt.c0);
}

template<mp_size_t n> 
bool Fp4_model_host<n>::operator==(const Fp4_model_host<n>& other) const
{
    return (this->c0 == other.c0 && this->c1 == other.c1);
}

template<mp_size_t n> 
bool Fp4_model_host<n>::operator!=(const Fp4_model_host<n>& other) const
{
    return !(operator==(other));
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::operator+(const Fp4_model_host<n>& other) const
{
    return Fp4_model_host<n>(params, this->c0 + other.c0, this->c1 + other.c1);
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::operator-(const Fp4_model_host<n>& other) const
{
    return Fp4_model_host<n>(params, this->c0 - other.c0, this->c1 - other.c1);
}

template<mp_size_t n> 
Fp4_model_host<n> operator*(const Fp_model_host<n>& lhs, const Fp4_model_host<n>& rhs)
{
    return Fp4_model_host<n>(rhs.params, lhs * rhs.c0, lhs * rhs.c1);
}

template<mp_size_t n> 
Fp4_model_host<n> operator*(const Fp2_model_host<n>& lhs, const Fp4_model_host<n>& rhs)
{
    return Fp4_model_host<n>(rhs.params, lhs * rhs.c0, lhs * rhs.c1);
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::operator*(const Fp4_model_host<n>& other) const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Karatsuba) */
    const my_Fp2 &B = other.c1, &A = other.c0, &b = this->c1, &a = this->c0;
    const my_Fp2 aA = a * A;
    const my_Fp2 bB = b * B;

    const my_Fp2 beta_bB = this->mul_by_non_residue(bB);
    return Fp4_model_host<n>(params, aA + beta_bB, (a + b) * (A + B) - aA  - bB);
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::operator-() const
{
    return Fp4_model_host<n>(params, -this->c0, -this->c1);
}

template<mp_size_t n, mp_size_t m>
Fp4_model_host<n> operator^(const Fp4_model_host<n>& self, const bigint_host<m>& exponent)
{
    return power_host<Fp4_model_host<n>, m>(self, exponent);
}

template<mp_size_t n, mp_size_t m>
Fp4_model_host<n> operator^(const Fp4_model_host<n>& self, const Fp_model_host<m>& exponent)
{
    return self ^ (exponent.as_bigint());
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::mul_by_023(const Fp4_model_host<n>& other) const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Karatsuba) */

    const my_Fp2 &B = other.c1, &A = other.c0, &b = this->c1, &a = this->c0;
    const my_Fp2 aA = my_Fp2(params->fp2_params, a.c0 * A.c0, a.c1 * A.c0);
    const my_Fp2 bB = b * B;

    const my_Fp2 beta_bB = this->mul_by_non_residue(bB);
    return Fp4_model_host<n>(params, aA + beta_bB, (a + b) * (A + B) - aA  - bB);
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::dbl() const
{
    return Fp4_model_host<n>(params, this->c0.dbl(), this->c1.dbl());
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::squared() const
{
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Complex) */

    const my_Fp2 &b = this->c1, &a = this->c0;
    const my_Fp2 ab = a * b;

    return Fp4_model_host<n>(params, (a + b) * (a + this->mul_by_non_residue(b)) - ab - this->mul_by_non_residue(ab), ab.dbl());
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::inverse() const
{
    /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 8 */
    const my_Fp2 &b = this->c1, &a = this->c0;
    const my_Fp2 t1 = b.squared();
    const my_Fp2 t0 = a.squared() - this->mul_by_non_residue(t1);
    const my_Fp2 new_t1 = t0.inverse();

    return Fp4_model_host<n>(params, a * new_t1, - (b * new_t1));
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::Frobenius_map(unsigned long power) const
{
    my_Fp Frobenius_coeffs_c1[4] = 
    {
        my_Fp(params->fp2_params->fp_params, *params->Frobenius_coeffs_c1[0]), 
        my_Fp(params->fp2_params->fp_params, *params->Frobenius_coeffs_c1[1]), 
        my_Fp(params->fp2_params->fp_params, *params->Frobenius_coeffs_c1[2]), 
        my_Fp(params->fp2_params->fp_params, *params->Frobenius_coeffs_c1[3]) 
    };

    return Fp4_model_host<n>(params, c0.Frobenius_map(power), Frobenius_coeffs_c1[power % 4] * c1.Frobenius_map(power));
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::unitary_inverse() const
{
    return Fp4_model_host<n>(params, this->c0, -this->c1);
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::cyclotomic_squared() const
{
    my_Fp2 t(params->fp2_params);
    const my_Fp2 A = this->c1.squared();
    const my_Fp2 B = this->c1 + this->c0;
    const my_Fp2 C = B.squared() - A;
    const my_Fp2 D = this->mul_by_non_residue(A); // Fp2(A.c1 * non_residue, A.c0)
    const my_Fp2 E = C - D;
    const my_Fp2 F = D + D + t.one();
    const my_Fp2 G = E - t.one();

    return Fp4_model_host<n>(params, F, G);
}

template<mp_size_t n, mp_size_t m>
Fp4_model_host<n> cyclotomic_exp(Fp4_model_host<n>& self, bigint_host<m>& exponent)
{
    Fp4_model_host<n> t(self.params);
    Fp4_model_host<n> res = t.one();
    Fp4_model_host<n> this_inverse = self.unitary_inverse();

    bool found_nonzero = false;

    long* NAF;
    size_t NAF_size;
    find_wnaf(&NAF, &NAF_size, 1, exponent);

    for (long i = NAF_size - 1; i >= 0; --i)
    {
        if (found_nonzero)
            res = res.cyclotomic_squared();

        if (NAF[i] != 0)
        {
            found_nonzero = true;
            res = NAF[i] > 0 ? res * self : res * this_inverse;
        }
    }

    delete[] NAF;
    return res;
}

template<mp_size_t n> 
void Fp4_model_host<n>::set_params(Fp4_params_host<n>* params)
{
    this->params = params;
    this->c0.set_params(params->fp2_params);
    this->c1.set_params(params->fp2_params);
}


template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::zero()
{
    Fp2_model_host<n> t(params->fp2_params);
    return Fp4_model_host<n>(params, t.zero(), t.zero());
}

template<mp_size_t n> 
Fp4_model_host<n> Fp4_model_host<n>::one()
{
    Fp2_model_host<n> t(params->fp2_params);
    return Fp4_model_host<n>(params, t.one(), t.zero());
}

}

#endif