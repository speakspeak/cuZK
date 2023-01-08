#ifndef __FP4_HOST_CUH__
#define __FP4_HOST_CUH__

#include "bigint_host.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp_host.cuh"
#include "fp2_host.cuh"

namespace libff {

template<mp_size_t n>
struct Fp4_params_host
{
    Fp2_params_host<n>* fp2_params;
    bigint_host<n>* non_residue;
    bigint_host<n>* Frobenius_coeffs_c1[4];
};


template<mp_size_t n> 
class Fp4_model_host {
public:
    typedef Fp_model_host<n> my_Fp;
    typedef Fp2_model_host<n> my_Fp2;

    Fp4_params_host<n>* params;

    my_Fp2 c0, c1;
    
    Fp4_model_host() : params(nullptr){}
    Fp4_model_host(Fp4_params_host<n>* params) : params(params), c0(params->fp2_params), c1(params->fp2_params) {};
    Fp4_model_host(Fp4_params_host<n>* params, const my_Fp2& c0, const my_Fp2& c1) : params(params), c0(c0), c1(c1) {};
    Fp4_model_host(const Fp4_model_host<n>& other) :params(other.params), c0(other.c0), c1(other.c1) {};

    Fp4_model_host<n>& operator=(const Fp4_model_host<n>& other) = default;

    bool operator==(const Fp4_model_host<n>& other) const;
    bool operator!=(const Fp4_model_host<n>& other) const;

    Fp4_model_host<n> operator+(const Fp4_model_host<n>& other) const;
    Fp4_model_host<n> operator-(const Fp4_model_host<n>& other) const;
    Fp4_model_host<n> operator*(const Fp4_model_host<n>& other) const;
    Fp4_model_host<n> operator-() const;

    // template<mp_size_t_ m>
    // __noinline__ __device__ Fp4_model<n> operator^(const bigint<m>& other) const;

    my_Fp2 mul_by_non_residue(const my_Fp2& elt) const;
    Fp4_model_host<n> mul_by_023(const Fp4_model_host& other) const;

    Fp4_model_host<n> dbl() const;
    Fp4_model_host<n> squared() const;
    Fp4_model_host<n> inverse() const;
    Fp4_model_host<n> Frobenius_map(unsigned long power) const;
    Fp4_model_host<n> unitary_inverse() const;
    Fp4_model_host<n> cyclotomic_squared() const;

    Fp4_model_host<n> zero();
    Fp4_model_host<n> one();

    bool is_zero() const { return c0.is_zero() && c1.is_zero();  }
    void clear() { c0.clear(); c1.clear(); }

    void set_params(Fp4_params_host<n>* params);

    bigint_host<n> base_field_char() { return *params->fp2_params->fp_params->modulus; }
    size_t extension_degree() { return 4; }
};

template<mp_size_t n> 
Fp4_model_host<n> operator*(const Fp_model_host<n>& lhs, const Fp4_model_host<n>& rhs);

template<mp_size_t n> 
Fp4_model_host<n> operator*(const Fp2_model_host<n>& lhs, const Fp4_model_host<n>& rhs);

template<mp_size_t n, mp_size_t m>
Fp4_model_host<n> cyclotomic_exp(const Fp4_model_host<n>& self, bigint_host<m>& exponent);

template<mp_size_t n, mp_size_t m>
Fp4_model_host<n> operator^(const Fp4_model_host<n>& self, const bigint_host<m>& exponent);

template<mp_size_t n, mp_size_t m>
Fp4_model_host<n> operator^(const Fp4_model_host<n>& self, const Fp_model_host<m>& exponent);

}

#include "fp4_host.cu"

#endif