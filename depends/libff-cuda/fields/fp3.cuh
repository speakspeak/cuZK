#ifndef __FP3_CUH__
#define __FP3_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"

namespace libff {

template<mp_size_t_ n>
struct Fp3_params
{
    Fp_params<n> fp_params;
    bigint<3 * n>* euler; 
    size_t s;      
    bigint<3 * n>* t;  
    bigint<3 * n>* t_minus_1_over_2; 
    bigint<n>* non_residue; 
    bigint<n>* nqr_c0;
    bigint<n>* nqr_c1;
    bigint<n>* nqr_c2;
    bigint<n>* nqr_to_t_c0;
    bigint<n>* nqr_to_t_c1;
    bigint<n>* nqr_to_t_c2;
    bigint<n>* Frobenius_coeffs_c1[3]; 
    bigint<n>* Frobenius_coeffs_c2[3];
};


template<mp_size_t_ n> 
class Fp3_model {
public:
    typedef Fp_model<n> my_Fp;

    const Fp_params<n>& fp_params;
    const Fp3_params<n>& fp3_params;
    const Fp3_params<n>& params;
    
    my_Fp c0, c1, c2;

    __noinline__ __device__ Fp3_model(const Fp3_params<n>& fp3_params) : fp_params(fp3_params.fp_params), fp3_params(fp3_params), params(fp3_params), c0(fp3_params.fp_params), c1(fp3_params.fp_params), c2(fp3_params.fp_params) {};
    __noinline__ __device__ Fp3_model(const Fp3_params<n>& fp3_params, const my_Fp& c0, const my_Fp& c1, const my_Fp& c2) :fp_params(fp3_params.fp_params), fp3_params(fp3_params), params(fp3_params), c0(c0), c1(c1), c2(c2) {};
    __noinline__ __device__ Fp3_model(const Fp3_params<n>& fp3_params, const bigint<n>& c0, const bigint<n>& c1, const bigint<n>& c2) :fp_params(fp3_params.fp_params), fp3_params(fp3_params), params(fp3_params), c0(fp3_params.fp_params, c0), c1(fp3_params.fp_params, c1), c2(fp3_params.fp_params, c2) {};
    __noinline__ __device__ Fp3_model(const Fp3_model<n>& other) = default;

    __noinline__ __device__ Fp3_model<n>& operator=(const Fp3_model<n>& other) = default;

    __noinline__ __device__ bool operator==(const Fp3_model<n>& other) const;
    __noinline__ __device__ bool operator!=(const Fp3_model<n>& other) const;

    __noinline__ __device__ Fp3_model<n> operator+(Fp3_model<n> other) const;
    __noinline__ __device__ Fp3_model<n> operator-(Fp3_model<n> other) const;
    __noinline__ __device__ Fp3_model<n> operator*(const Fp3_model<n>& other) const;
    __noinline__ __device__ Fp3_model<n> operator-() const;

    template<mp_size_t_ m>
    __noinline__ __device__ Fp3_model<n> operator^(const bigint<m>& other) const;

    __noinline__ __device__ Fp3_model<n> dbl() const;
    __noinline__ __device__ Fp3_model<n> squared() const;
    __noinline__ __device__ Fp3_model<n> inverse() const;
    __noinline__ __device__ Fp3_model<n> Frobenius_map(unsigned long power) const;
    __noinline__ __device__ Fp3_model<n> sqrt() const; // HAS TO BE A SQUARE (else does not terminate)
    
    __noinline__ __device__ Fp3_model<n> zero();
    __noinline__ __device__ Fp3_model<n> one();
    // __noinline__ __device__ Fp3_model random_element();

    __noinline__ __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero() && c2.is_zero(); }
    __noinline__ __device__ void clear() { c0.clear(); c1.clear(); c2.clear(); }

    __noinline__ __device__ size_t size_in_bits();
    __noinline__ __device__ bigint<n> base_field_char() { return *fp_params.modulus; }

};

template<mp_size_t_ n> 
__noinline__ __device__ Fp3_model<n> operator*(const Fp_model<n>& lhs, const Fp3_model<n>& rhs);

}

#include "fp3.cu"

#endif