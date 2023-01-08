#ifndef __FP2_HOST_CUH__
#define __FP2_HOST_CUH__

#include "bigint_host.cuh"
#include "fp_host.cuh"

namespace libff {

template<mp_size_t n>
struct Fp2_params_host
{
    Fp_params_host<n>* fp_params;
    bigint_host<2 * n>* euler;
    size_t s;
    bigint_host<2 * n>* t;
    bigint_host<2 * n>* t_minus_1_over_2;
    bigint_host<n>* non_residue;
    bigint_host<n>* nqr_c0;
    bigint_host<n>* nqr_c1;
    bigint_host<n>* nqr_to_t_c0;
    bigint_host<n>* nqr_to_t_c1;
    bigint_host<n>* Frobenius_coeffs_c1[2];
};


template<mp_size_t n>
class Fp2_model_host {
public: 
    typedef Fp_model_host<n> my_Fp;

    Fp2_params_host<n>* params;

    my_Fp c0, c1;

    Fp2_model_host() : params(nullptr) {}
    Fp2_model_host(Fp2_params_host<n>* params) : params(params), c0(params->fp_params), c1(params->fp_params) {}
    Fp2_model_host(Fp2_params_host<n>* params, const my_Fp& c0, const my_Fp& c1) : params(params), c0(c0), c1(c1) {}
    Fp2_model_host(Fp2_params_host<n>* params, const bigint_host<n>& c0, const bigint_host<n>& c1): params(params), c0(params->fp_params, c0), c1(params->fp_params, c1) {};
    Fp2_model_host(const Fp2_model_host<n>& other) = default;

    Fp2_model_host<n>& operator=(const Fp2_model_host<n>& other) = default;

    bool operator==(const Fp2_model_host<n>& other) const;
    bool operator!=(const Fp2_model_host<n>& other) const;

    Fp2_model_host<n> operator+(const Fp2_model_host<n>& other) const;
    Fp2_model_host<n> operator-(const Fp2_model_host<n>& other) const;
    Fp2_model_host<n> operator*(const Fp2_model_host<n>& other) const;
    Fp2_model_host<n> operator-() const;

    // template<mp_size_t m>
    // Fp2_model_host<n> operator^(const bigint<m>& other) const;

    Fp2_model_host<n> dbl() const;
    Fp2_model_host<n> squared() const;    // default is squared_complex
    Fp2_model_host<n> inverse() const;
    Fp2_model_host<n> Frobenius_map(unsigned long power) const;
    Fp2_model_host<n> sqrt() const;  // HAS TO BE A SQUARE (else does not terminate)
    Fp2_model_host<n> squared_karatsuba() const;
    Fp2_model_host<n> squared_complex() const;

    Fp2_model_host<n> zero() const;
    Fp2_model_host<n> one() const;
    Fp2_model_host<n> random_element();

    bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
    void clear() { c0.clear(); c1.clear(); }

    void set_params(Fp2_params_host<n>* params);

    size_t size_in_bits();
    bigint_host<n> base_field_char() { return *params->fp_params->modulus; }
};

template<mp_size_t n> 
Fp2_model_host<n> operator*(const Fp_model_host<n>& lhs, const Fp2_model_host<n>& rhs);


}


#include "fp2_host.cu"

#endif
