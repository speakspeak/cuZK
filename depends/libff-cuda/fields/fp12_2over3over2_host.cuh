#ifndef __FP12_2OVER3OVER2_HOST_CUH__
#define __FP12_2OVER3OVER2_HOST_CUH__

#include "bigint_host.cuh"
#include "fp_host.cuh"
#include "fp2_host.cuh"
#include "fp6_3over2_host.cuh"


namespace libff {

template<mp_size_t n>
struct Fp12_params_host
{
    Fp6_3over2_params_host<n>* fp6_params;
    bigint_host<n>* non_residue_c0;
    bigint_host<n>* non_residue_c1;
    bigint_host<n>* Frobenius_coeffs_c1_c0[12];
    bigint_host<n>* Frobenius_coeffs_c1_c1[12];
};


template<mp_size_t n> 
class Fp12_2over3over2_model_host {
public:
    typedef Fp_model_host<n> my_Fp;
    typedef Fp2_model_host<n> my_Fp2;
    typedef Fp6_3over2_model_host<n> my_Fp6;

    Fp12_params_host<n>* params;

    my_Fp6 c0, c1;

    Fp12_2over3over2_model_host() : params(nullptr) {}
    Fp12_2over3over2_model_host(Fp12_params_host<n>* params) : params(params), c0(params->fp6_params), c1(params->fp6_params)  {}
    Fp12_2over3over2_model_host(Fp12_params_host<n>* params, const my_Fp6& c0, const my_Fp6& c1) : params(params), c0(c0), c1(c1) {}
    Fp12_2over3over2_model_host(const Fp12_2over3over2_model_host<n>& other) : params(other.params), c0(other.c0), c1(other.c1) {}
    
    Fp12_2over3over2_model_host<n>& operator=(const Fp12_2over3over2_model_host<n>& other) = default;
    
    bool operator==(const Fp12_2over3over2_model_host<n>& other) const;
    bool operator!=(const Fp12_2over3over2_model_host<n>& other) const;

    Fp12_2over3over2_model_host<n> operator+(const Fp12_2over3over2_model_host<n>& other) const;
    Fp12_2over3over2_model_host<n> operator-(const Fp12_2over3over2_model_host<n>& other) const;
    Fp12_2over3over2_model_host<n> operator*(const Fp12_2over3over2_model_host<n>& other) const;
    Fp12_2over3over2_model_host<n> operator-() const;

    my_Fp6 mul_by_non_residue(const my_Fp6& elt) const;
    Fp12_2over3over2_model_host<n> mul_by_024(const my_Fp2& ell_0, const my_Fp2& ell_VW, const my_Fp2& ell_VV) const;
    Fp12_2over3over2_model_host<n> mul_by_045(const my_Fp2& ell_0, const my_Fp2& ell_VW, const my_Fp2& ell_VV) const;
   
    Fp12_2over3over2_model_host<n> dbl() const;                    
    Fp12_2over3over2_model_host<n> squared() const;                    // default is squared_complex
    Fp12_2over3over2_model_host<n> squared_karatsuba() const;
    Fp12_2over3over2_model_host<n> squared_complex() const;
    Fp12_2over3over2_model_host<n> inverse() const;
    Fp12_2over3over2_model_host<n> Frobenius_map(unsigned long power) const;
    Fp12_2over3over2_model_host<n> unitary_inverse() const;
    Fp12_2over3over2_model_host<n> cyclotomic_squared() const;

    // template<mp_size_t_ m>
    //   Fp12_2over3over2_model<n> cyclotomic_exp(const bigint<m>& exponent) const;

    Fp12_2over3over2_model_host<n> zero();
    Fp12_2over3over2_model_host<n> one();
    Fp12_2over3over2_model_host<n> random_element();

    bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
    void clear() { c0.clear(); c1.clear(); }

    void set_params(Fp12_params_host<n>* params);

    bigint_host<n> base_field_char() { return *params->fp6_params->fp2_params->fp_params->modulus; }
    size_t extension_degree() { return 12; }

};

template<mp_size_t n> 
Fp12_2over3over2_model_host<n> operator*(const Fp_model_host<n>& lhs, const Fp12_2over3over2_model_host<n>& rhs);

template<mp_size_t n> 
Fp12_2over3over2_model_host<n> operator*(const Fp2_model_host<n>& lhs, const Fp12_2over3over2_model_host<n>& rhs);

template<mp_size_t n> 
Fp12_2over3over2_model_host<n> operator*(const Fp6_3over2_model_host<n>& lhs, const Fp12_2over3over2_model_host<n>& rhs);

// template<mp_size_t_ n, mp_size_t_ m>
//   Fp12_2over3over2_model<n> operator^(const Fp12_2over3over2_model<n>& self, const bigint<m>& exponent);

// template<mp_size_t_ n, mp_size_t_ m>
//   Fp12_2over3over2_model<n> operator^(const Fp12_2over3over2_model<n>& self, const Fp_model<m>& exponent);


}

#include "fp12_2over3over2_host.cu"

#endif