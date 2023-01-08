#ifndef __FP6_3OVER2_HOST_CUH__
#define __FP6_3OVER2_HOST_CUH__

#include "bigint_host.cuh"
#include "fp_host.cuh"
#include "fp2_host.cuh"

namespace libff {

template<mp_size_t n>
struct Fp6_3over2_params_host
{
    Fp2_params_host<n>* fp2_params;
    bigint_host<n>* non_residue_c0;
    bigint_host<n>* non_residue_c1;
    bigint_host<n>* Frobenius_coeffs_c1_c0[6];
    bigint_host<n>* Frobenius_coeffs_c1_c1[6];
    bigint_host<n>* Frobenius_coeffs_c2_c0[6];
    bigint_host<n>* Frobenius_coeffs_c2_c1[6];
};


template<mp_size_t n> 
class Fp6_3over2_model_host {
public:
    typedef Fp_model_host<n> my_Fp;
    typedef Fp2_model_host<n> my_Fp2;

    Fp6_3over2_params_host<n>* params;

    my_Fp2 c0, c1, c2;

    Fp6_3over2_model_host() : params(nullptr){}
    Fp6_3over2_model_host(Fp6_3over2_params_host<n>* params) : params(params), c0(params->fp2_params), c1(params->fp2_params), c2(params->fp2_params) {};
    Fp6_3over2_model_host(Fp6_3over2_params_host<n>* params, const my_Fp2& c0, const my_Fp2& c1, const my_Fp2& c2) : params(params), c0(c0), c1(c1), c2(c2) {};
    Fp6_3over2_model_host(const Fp6_3over2_model_host<n>& other) :params(other.params), c0(other.c0), c1(other.c1), c2(other.c2) {};

    Fp6_3over2_model_host<n>& operator=(const Fp6_3over2_model_host<n>& other) = default;

    bool operator==(const Fp6_3over2_model_host<n>& other) const;
    bool operator!=(const Fp6_3over2_model_host<n>& other) const;

    Fp6_3over2_model_host<n> operator+(const Fp6_3over2_model_host<n>& other) const;
    Fp6_3over2_model_host<n> operator-(const Fp6_3over2_model_host<n>& other) const;
    Fp6_3over2_model_host<n> operator*(const Fp6_3over2_model_host<n>& other) const;
    Fp6_3over2_model_host<n> operator-() const;

    // template<mp_size_t m>
    // Fp6_3over2_model_host<n> operator^(const bigint_host<m>& other) const;

    my_Fp2 mul_by_non_residue(const my_Fp2& elt) const;
    Fp6_3over2_model_host<n> dbl() const;
    Fp6_3over2_model_host<n> squared() const;
    Fp6_3over2_model_host<n> inverse() const;
    Fp6_3over2_model_host<n> Frobenius_map(unsigned long power) const;

    Fp6_3over2_model_host<n> zero();
    Fp6_3over2_model_host<n> one();
    Fp6_3over2_model_host<n> random_element();

    bool is_zero() const { return c0.is_zero() && c1.is_zero() && c2.is_zero(); }
    void clear() { c0.clear(); c1.clear(); c2.clear(); }

    void set_params(Fp6_3over2_params_host<n>* params);

    bigint_host<n> base_field_char() { return *params->fp2_params->fp_params->modulus; }
    size_t extension_degree() { return 6; }
};

template<mp_size_t n> 
Fp6_3over2_model_host<n> operator*(const Fp_model_host<n>& lhs, const Fp6_3over2_model_host<n>& rhs);

template<mp_size_t n> 
Fp6_3over2_model_host<n> operator*(const Fp2_model_host<n>& lhs, const Fp6_3over2_model_host<n>& rhs);

}

#include "fp6_3over2_host.cu"

#endif