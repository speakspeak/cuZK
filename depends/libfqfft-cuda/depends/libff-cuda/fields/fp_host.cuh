#ifndef __FP_HOST_CUH__
#define __FP_HOST_CUH__

#include "bigint_host.cuh"

namespace libff {

template<mp_size_t n>
class Fp_model_host;

template<mp_size_t n>
struct Fp_params_host
{
    mp_size_t num_limbs;
    bigint_host<n>* modulus;
    size_t num_bits;
    bigint_host<n>* euler;
    size_t s;
    bigint_host<n>* t;
    bigint_host<n>* t_minus_1_over_2;
    bigint_host<n>* nqr;
    bigint_host<n>* nqr_to_t;
    bigint_host<n>* multiplicative_generator;
    bigint_host<n>* root_of_unity;
    mp_limb_t inv;
    bigint_host<n>* Rsquared;
    bigint_host<n>* Rcubed;
    Fp_model_host<n>* zero;
    Fp_model_host<n>* one;
};

template<mp_size_t n>
class Fp_model_host {
public:
    Fp_params_host<n>* params;

    bigint_host<n> mont_repr;

     Fp_model_host() : params(nullptr) {}
     Fp_model_host(Fp_params_host<n>* params) : params(params) {}
     Fp_model_host(Fp_params_host<n>* params, const bigint_host<n>& b);
     Fp_model_host(Fp_params_host<n>* params, const long x, const bool is_unsigned = false);
     Fp_model_host(const Fp_model_host<n>& other) = default;

     void mul_reduce(const bigint_host<n> &other);
    
     Fp_model_host<n>& operator=(const Fp_model_host<n>& other) = default;

     bool operator==(const Fp_model_host<n>& other) const;
     bool operator!=(const Fp_model_host<n>& other) const;

     Fp_model_host<n>& operator+=(const Fp_model_host<n>& other);
     Fp_model_host<n>& operator-=(const Fp_model_host<n>& other); 

     Fp_model_host<n>& operator*=(const bigint_host<n>& other);
     Fp_model_host<n>& operator*=(const Fp_model_host<n>& other); 

    //  Fp_model_host<n>& operator^=(const unsigned long pow);
    // template<mp_size_t_ m>
    //  Fp_model_host<n>& operator^=(const bigint_host<m> pow);

     Fp_model_host<n> operator+(Fp_model_host<n> other) const; 
     Fp_model_host<n> operator-(Fp_model_host<n> other) const;
     Fp_model_host<n> operator*(const Fp_model_host<n>& other) const;
     Fp_model_host<n> operator-() const;

    // Fp_model_host<n> operator^(const unsigned long pow) const;
    // template<mp_size_t_ m>
    // Fp_model_host<n> operator^(const bigint_host<m>& pow) const;

     Fp_model_host<n> dbl() const;
     Fp_model_host<n> squared() const;                        
     Fp_model_host<n>& invert();                      
     Fp_model_host<n> inverse() const;
    //  Fp_model_host<n> sqrt() const;  // HAS TO BE A SQUARE (else does not terminate)   //
    
     void init_zero();
     void init_one();

     Fp_model_host<n> zero() const;
     Fp_model_host<n> one() const;
     Fp_model_host<n> random_element() const;
     Fp_model_host<n> geometric_generator(); // generator^k, for k = 1 to m, domain size m
     Fp_model_host<n> arithmetic_generator();// generator++, for k = 1 to m, domain size m

     void set_ulong(const unsigned long x);
     bigint_host<n> as_bigint() const; 
     unsigned long as_ulong() const;

     bool is_zero() const;
     void clear();

     void set_params(Fp_params_host<n>* params);

     size_t size_in_bits() const { return params->num_bits; }
     size_t capacity() const { return params->num_bits - 1; }
     bigint_host<n> field_char() { return *params->modulus; }

     bool modulus_is_valid() { return params->modulus->data[n - 1] != 0; }

};

}

#include "fp_host.cu"

#endif