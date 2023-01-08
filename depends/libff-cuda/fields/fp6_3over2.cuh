#ifndef __FP6_3OVER2_CUH__
#define __FP6_3OVER2_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"
#include "fp2.cuh"

namespace libff {

template<mp_size_t_ n>
struct Fp6_3over2_params
{
    Fp2_params<n>* fp2_params;
    bigint<n>* non_residue_c0;
    bigint<n>* non_residue_c1;
    bigint<n>* Frobenius_coeffs_c1_c0[6];
    bigint<n>* Frobenius_coeffs_c1_c1[6];
    bigint<n>* Frobenius_coeffs_c2_c0[6];
    bigint<n>* Frobenius_coeffs_c2_c1[6];
};


template<mp_size_t_ n> 
class Fp6_3over2_model {
public:
    typedef Fp_model<n> my_Fp;
    typedef Fp2_model<n> my_Fp2;

    Fp6_3over2_params<n>* params;

    my_Fp2 c0, c1, c2;
    
    __noinline__ __device__ Fp6_3over2_model() : params(nullptr){}
    __noinline__ __device__ Fp6_3over2_model(Fp6_3over2_params<n>* params) : params(params), c0(params->fp2_params), c1(params->fp2_params), c2(params->fp2_params) {};
    __noinline__ __device__ Fp6_3over2_model(Fp6_3over2_params<n>* params, const my_Fp2& c0, const my_Fp2& c1, const my_Fp2& c2) : params(params), c0(c0), c1(c1), c2(c2) {};
    __noinline__ __device__ Fp6_3over2_model(const Fp6_3over2_model<n>& other) :params(other.params), c0(other.c0), c1(other.c1), c2(other.c2) {};

    __noinline__ __device__ Fp6_3over2_model<n>& operator=(const Fp6_3over2_model<n>& other) = default;

    __noinline__ __device__ bool operator==(const Fp6_3over2_model<n>& other) const;
    __noinline__ __device__ bool operator!=(const Fp6_3over2_model<n>& other) const;

    __noinline__ __device__ Fp6_3over2_model<n> operator+(const Fp6_3over2_model<n>& other) const;
    __noinline__ __device__ Fp6_3over2_model<n> operator-(const Fp6_3over2_model<n>& other) const;
    __noinline__ __device__ Fp6_3over2_model<n> operator*(const Fp6_3over2_model<n>& other) const;
    __noinline__ __device__ Fp6_3over2_model<n> operator-() const;

    template<mp_size_t_ m>
    __noinline__ __device__ Fp6_3over2_model<n> operator^(const bigint<m>& other) const;

    __noinline__ __device__ my_Fp2 mul_by_non_residue(const my_Fp2& elt) const;

    __noinline__ __device__ Fp6_3over2_model<n> dbl() const;
    __noinline__ __device__ Fp6_3over2_model<n> squared() const;
    __noinline__ __device__ Fp6_3over2_model<n> inverse() const;
    __noinline__ __device__ Fp6_3over2_model<n> Frobenius_map(unsigned long power) const;

    __noinline__ __device__ Fp6_3over2_model<n> zero();
    __noinline__ __device__ Fp6_3over2_model<n> one();
    __noinline__ __device__ Fp6_3over2_model<n> random_element();

    __noinline__ __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero() && c2.is_zero(); }
    __noinline__ __device__ void clear() { c0.clear(); c1.clear(); c2.clear(); }

    __noinline__ __device__ void set_params(Fp6_3over2_params<n>* params);

    __noinline__ __device__ bigint<n> base_field_char() { return *params->fp2_params->fp_params->modulus; }
    __noinline__ __device__ size_t extension_degree() { return 6; }
};

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> operator*(const Fp_model<n>& lhs, const Fp6_3over2_model<n>& rhs);

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_3over2_model<n> operator*(const Fp2_model<n>& lhs, const Fp6_3over2_model<n>& rhs);

}

#include "fp6_3over2.cu"

#endif