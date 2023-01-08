#ifndef __FP12_2OVER3OVER2_CUH__
#define __FP12_2OVER3OVER2_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"
#include "fp2.cuh"
#include "fp6_3over2.cuh"


namespace libff {

template<mp_size_t_ n>
struct Fp12_params
{
    Fp6_3over2_params<n>* fp6_params;
    bigint<n>* non_residue_c0;
    bigint<n>* non_residue_c1;
    bigint<n>* Frobenius_coeffs_c1_c0[12];
    bigint<n>* Frobenius_coeffs_c1_c1[12];
};


template<mp_size_t_ n> 
class Fp12_2over3over2_model {
public:
    typedef Fp_model<n> my_Fp;
    typedef Fp2_model<n> my_Fp2;
    typedef Fp6_3over2_model<n> my_Fp6;

    Fp12_params<n>* params;

    my_Fp6 c0, c1;

    __noinline__ __device__ Fp12_2over3over2_model() : params(nullptr) {}
    __noinline__ __device__ Fp12_2over3over2_model(Fp12_params<n>* params) : params(params), c0(params->fp6_params), c1(params->fp6_params)  {}
    __noinline__ __device__ Fp12_2over3over2_model(Fp12_params<n>* params, const my_Fp6& c0, const my_Fp6& c1) : params(params), c0(c0), c1(c1) {}
    __noinline__ __device__ Fp12_2over3over2_model(const Fp12_2over3over2_model<n>& other) : params(other.params), c0(other.c0), c1(other.c1) {}
    
    __noinline__ __device__ Fp12_2over3over2_model<n>& operator=(const Fp12_2over3over2_model<n>& other) = default;
    
    __noinline__ __device__ bool operator==(const Fp12_2over3over2_model<n>& other) const;
    __noinline__ __device__ bool operator!=(const Fp12_2over3over2_model<n>& other) const;

    __noinline__ __device__ Fp12_2over3over2_model<n> operator+(const Fp12_2over3over2_model<n>& other) const;
    __noinline__ __device__ Fp12_2over3over2_model<n> operator-(const Fp12_2over3over2_model<n>& other) const;
    __noinline__ __device__ Fp12_2over3over2_model<n> operator*(const Fp12_2over3over2_model<n>& other) const;
    __noinline__ __device__ Fp12_2over3over2_model<n> operator-() const;

    __noinline__ __device__ my_Fp6 mul_by_non_residue(const my_Fp6& elt) const;
    __noinline__ __device__ Fp12_2over3over2_model<n> mul_by_024(const my_Fp2& ell_0, const my_Fp2& ell_VW, const my_Fp2& ell_VV) const;
    __noinline__ __device__ Fp12_2over3over2_model<n> mul_by_045(const my_Fp2& ell_0, const my_Fp2& ell_VW, const my_Fp2& ell_VV) const;
   
    __noinline__ __device__ Fp12_2over3over2_model<n> dbl() const;                    
    __noinline__ __device__ Fp12_2over3over2_model<n> squared() const;                    // default is squared_complex
    __noinline__ __device__ Fp12_2over3over2_model<n> squared_karatsuba() const;
    __noinline__ __device__ Fp12_2over3over2_model<n> squared_complex() const;
    __noinline__ __device__ Fp12_2over3over2_model<n> inverse() const;
    __noinline__ __device__ Fp12_2over3over2_model<n> Frobenius_map(unsigned long power) const;
    __noinline__ __device__ Fp12_2over3over2_model<n> unitary_inverse() const;
    __noinline__ __device__ Fp12_2over3over2_model<n> cyclotomic_squared() const;

    template<mp_size_t_ m>
    __noinline__ __device__ Fp12_2over3over2_model<n> cyclotomic_exp(const bigint<m>& exponent) const;

    __noinline__ __device__ Fp12_2over3over2_model<n> zero();
    __noinline__ __device__ Fp12_2over3over2_model<n> one();
    __noinline__ __device__ Fp12_2over3over2_model<n> random_element();

    __noinline__ __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
    __noinline__ __device__ void clear() { c0.clear(); c1.clear(); }

    __noinline__ __device__ void set_params(Fp12_params<n>* params);

    __noinline__ __device__ bigint<n> base_field_char() { return *params->fp6_params->fp2_params->fp_params->modulus; }
    __noinline__ __device__ size_t extension_degree() { return 12; }

};

template<mp_size_t_ n> 
__noinline__ __device__ Fp12_2over3over2_model<n> operator*(const Fp_model<n>& lhs, const Fp12_2over3over2_model<n>& rhs);

template<mp_size_t_ n> 
__noinline__ __device__ Fp12_2over3over2_model<n> operator*(const Fp2_model<n>& lhs, const Fp12_2over3over2_model<n>& rhs);

template<mp_size_t_ n> 
__noinline__ __device__ Fp12_2over3over2_model<n> operator*(const Fp6_3over2_model<n>& lhs, const Fp12_2over3over2_model<n>& rhs);

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp12_2over3over2_model<n> operator^(const Fp12_2over3over2_model<n>& self, const bigint<m>& exponent);

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp12_2over3over2_model<n> operator^(const Fp12_2over3over2_model<n>& self, const Fp_model<m>& exponent);


}

#include "fp12_2over3over2.cu"

#endif