#ifndef __FP2_CUH__
#define __FP2_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"

namespace libff {

template<mp_size_t_ n>
struct Fp2_params
{
    Fp_params<n>* fp_params;
    bigint<2 * n>* euler;
    size_t s;
    bigint<2 * n>* t;
    bigint<2 * n>* t_minus_1_over_2;
    bigint<n>* non_residue;
    bigint<n>* nqr_c0;
    bigint<n>* nqr_c1;
    bigint<n>* nqr_to_t_c0;
    bigint<n>* nqr_to_t_c1;
    bigint<n>* Frobenius_coeffs_c1[2];
};


template<mp_size_t_ n>
class Fp2_model {
public: 
    typedef Fp_model<n> my_Fp;

    Fp2_params<n>* params;

    my_Fp c0, c1;

    __device__ Fp2_model() : params(nullptr) {}
    __device__ Fp2_model(Fp2_params<n>* params) : params(params), c0(params->fp_params), c1(params->fp_params) {}
    __device__ Fp2_model(Fp2_params<n>* params, const my_Fp& c0, const my_Fp& c1) : params(params), c0(c0), c1(c1) {}
    __device__ Fp2_model(Fp2_params<n>* params, const bigint<n>& c0, const bigint<n>& c1): params(params), c0(params->fp_params, c0), c1(params->fp_params, c1) {};
    __device__ Fp2_model(const Fp2_model<n>& other) = default;

    __device__ Fp2_model<n>& operator=(const Fp2_model<n>& other) = default;

    __device__ bool operator==(const Fp2_model<n>& other) const;
    __device__ bool operator!=(const Fp2_model<n>& other) const;

    __noinline__ __device__ Fp2_model<n> operator+(const Fp2_model<n>& other) const;
    __noinline__ __device__ Fp2_model<n> operator-(const Fp2_model<n>& other) const;
    __device__ Fp2_model<n> operator*(const Fp2_model<n>& other) const;
    __noinline__ __device__ Fp2_model<n> operator-() const;

    template<mp_size_t_ m>
    __noinline__ __device__ Fp2_model<n> operator^(const bigint<m>& other) const;

    __noinline__ __device__ Fp2_model<n> dbl() const;
    __device__ Fp2_model<n> squared() const;    // default is squared_complex
    __noinline__ __device__ Fp2_model<n> inverse() const;
    __noinline__ __device__ Fp2_model<n> Frobenius_map(unsigned long power) const;
    __noinline__ __device__ Fp2_model<n> sqrt() const;  // HAS TO BE A SQUARE (else does not terminate)
    __noinline__ __device__ Fp2_model<n> squared_karatsuba() const;
    __noinline__ __device__ Fp2_model<n> squared_complex() const;

    __device__ Fp2_model<n> zero() const;
    __device__ Fp2_model<n> one() const;
    __device__ Fp2_model<n> random_element();

    __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
    __device__ void clear() { c0.clear(); c1.clear(); }

    __device__ void set_params(Fp2_params<n>* params);

    __device__ size_t size_in_bits();
    __device__ bigint<n> base_field_char() { return *params->fp_params->modulus; }
};


template<mp_size_t_ n> 
__noinline__ __device__ Fp2_model<n> operator*(const Fp_model<n>& lhs, const Fp2_model<n>& rhs);

}

#include "fp2.cu"

#endif