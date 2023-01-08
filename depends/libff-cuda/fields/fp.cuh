#ifndef __FP_CUH__
#define __FP_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"

namespace libff {

template<mp_size_t_ n>
class Fp_model;

template<mp_size_t_ n>
struct Fp_params
{
    mp_size_t_ num_limbs;
    bigint<n>* modulus;
    size_t num_bits;
    bigint<n>* euler;
    size_t s;
    bigint<n>* t;
    bigint<n>* t_minus_1_over_2;
    bigint<n>* nqr;
    bigint<n>* nqr_to_t;
    bigint<n>* multiplicative_generator;
    bigint<n>* root_of_unity;
    mp_limb_t_ inv;
    bigint<n>* Rsquared;
    bigint<n>* Rcubed;
    Fp_model<n>* zero;
    Fp_model<n>* one;
};

template<mp_size_t_ n>
class Fp_model {
public:
    Fp_params<n>* params;

    bigint<n> mont_repr;

    __device__ Fp_model() : params(nullptr) {}
    __device__ Fp_model(Fp_params<n>* params) : params(params) {}
    __device__ Fp_model(Fp_params<n>* params, const bigint<n>& b);
    __device__ Fp_model(Fp_params<n>* params, const long x, const bool is_unsigned = false);
    __device__ Fp_model(const Fp_model<n>& other) = default;
    
    __device__ Fp_model<n>& operator=(const Fp_model<n>& other) = default;

    __device__ bool operator==(const Fp_model<n>& other) const;
    __device__ bool operator!=(const Fp_model<n>& other) const;

    __device__ Fp_model<n>& operator+=(const Fp_model<n>& other);
    __device__ Fp_model<n>& operator-=(const Fp_model<n>& other); 

    __device__ Fp_model<n>& operator*=(const bigint<n>& other);
    __device__ Fp_model<n>& operator*=(const Fp_model<n>& other); 

    __noinline__ __device__ Fp_model<n>& operator^=(const unsigned long pow);
    template<mp_size_t_ m>
    __noinline__ __device__ Fp_model<n>& operator^=(const bigint<m> pow);

    __device__ Fp_model<n> operator+(Fp_model<n> other) const; 
    __device__ Fp_model<n> operator-(Fp_model<n> other) const;
    __noinline__ __device__ Fp_model<n> operator*(const Fp_model<n>& other) const;
    __device__ Fp_model<n> operator-() const;

    __noinline__ __device__ Fp_model<n> operator^(const unsigned long pow) const;
    template<mp_size_t_ m>
    __noinline__ __device__ Fp_model<n> operator^(const bigint<m>& pow) const;

    __device__ Fp_model<n> dbl() const;
    __device__ Fp_model<n> squared() const;                        
    __device__ Fp_model<n>& invert();                      
    __device__ Fp_model<n> inverse() const;
    __host__ Fp_model<n>* inverse_host();

    __device__ Fp_model<n> sqrt() const;  // HAS TO BE A SQUARE (else does not terminate)   //
    
    __device__ void init_zero();
    __device__ void init_one();

    __device__ Fp_model<n> zero() const;
    __host__ Fp_model<n>* zero_host();
    __device__ Fp_model<n> one() const;
    __host__ Fp_model<n>* one_host();
    
    __device__ Fp_model<n> random_element() const;
    __device__ Fp_model<n> geometric_generator(); // generator^k, for k = 1 to m, domain size m
    __device__ Fp_model<n> arithmetic_generator();// generator++, for k = 1 to m, domain size m

    __device__ void set_ulong(const unsigned long x);
    __device__ bigint<n> as_bigint() const; 
    __device__ unsigned long as_ulong() const;

    __device__ bool is_zero() const;
    __device__ void clear();

    __device__ void set_params(Fp_params<n>* params);

    __device__ size_t size_in_bits() const { return params->num_bits; }
    __device__ size_t capacity() const { return params->num_bits - 1; }
    __device__ bigint<n> field_char() { return *params->modulus; }

    __device__ bool modulus_is_valid() { return params->modulus->data[n - 1] != 0; }

};

}

#include "fp.cu"

#endif