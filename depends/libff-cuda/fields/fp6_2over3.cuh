#ifndef __FP6_2OVER3_CUH__
#define __FP6_2OVER3_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"
#include "fp2.cuh"
#include "fp3.cuh"

namespace libff {


template<mp_size_t_ n>
struct Fp6_2over3_params
{
    Fp2_params<n> fp2_params;
    Fp3_params<n> fp3_params;
    bigint<n>* non_residue;
    bigint<n>* Frobenius_coeffs_c1[6];
};

template<mp_size_t_ n> 
class Fp6_2over3_model {
public:
    typedef Fp_model<n> my_Fp;
    typedef Fp2_model<n> my_Fp2;
    typedef Fp3_model<n> my_Fp3;
    typedef my_Fp3 my_Fpe;

    const Fp_params<n>& fp_params;
    const Fp2_params<n>& fp2_params;
    const Fp3_params<n>& fp3_params;
    const Fp6_2over3_params<n>& fp6_params;
    const Fp6_2over3_params<n>& params;
    
    my_Fp3 c0, c1;

    __noinline__ __device__ Fp6_2over3_model(const Fp6_2over3_params<n>& fp6_params) : fp_params(fp6_params.fp2_params.fp_params), fp2_params(fp6_params.fp2_params), fp3_params(fp6_params.fp3_params), fp6_params(fp6_params), params(fp6_params), c0(fp3_params), c1(fp3_params) {};
    __noinline__ __device__ Fp6_2over3_model(const Fp6_2over3_params<n>& fp6_params, const my_Fp3& c0, const my_Fp3& c1) : fp_params(fp6_params.fp2_params.fp_params), fp2_params(fp6_params.fp2_params), fp3_params(fp6_params.fp3_params), fp6_params(fp6_params), params(fp6_params), c0(c0), c1(c1) {};
    __noinline__ __device__ Fp6_2over3_model(const Fp6_2over3_model<n>& other) = default;
    
    __noinline__ __device__ Fp6_2over3_model& operator=(const Fp6_2over3_model<n>& other) = default;

    __noinline__ __device__ bool operator==(const Fp6_2over3_model<n>& other) const;
    __noinline__ __device__ bool operator!=(const Fp6_2over3_model<n>& other) const;

    __noinline__ __device__ Fp6_2over3_model<n> operator+(const Fp6_2over3_model<n>& other) const;
    __noinline__ __device__ Fp6_2over3_model<n> operator-(const Fp6_2over3_model<n>& other) const;
    __noinline__ __device__ Fp6_2over3_model<n> operator*(const Fp6_2over3_model<n>& other) const;
    __noinline__ __device__ Fp6_2over3_model<n> operator-() const;

    __noinline__ __device__ my_Fp3 mul_by_non_residue(const my_Fp3& elem) const;
    __noinline__ __device__ Fp6_2over3_model<n> mul_by_2345(const Fp6_2over3_model<n>& other) const;
    
    __noinline__ __device__ Fp6_2over3_model<n> dbl() const;
    __noinline__ __device__ Fp6_2over3_model<n> squared() const;
    __noinline__ __device__ Fp6_2over3_model<n> inverse() const;
    __noinline__ __device__ Fp6_2over3_model<n> Frobenius_map(unsigned long power) const;
    __noinline__ __device__ Fp6_2over3_model<n> unitary_inverse() const;
    __noinline__ __device__ Fp6_2over3_model<n> cyclotomic_squared() const;

    template<mp_size_t_ m>
    __noinline__ __device__ Fp6_2over3_model<n> cyclotomic_exp(const bigint<m>& exponent) const;

    __noinline__ __device__ Fp6_2over3_model<n> zero();
    __noinline__ __device__ Fp6_2over3_model<n> one();

    __noinline__ __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
    __noinline__ __device__ void clear() { c0.clear(); c1.clear(); }

    __noinline__ __device__ bigint<n> base_field_char() { return *fp_params.modulus; }
    __noinline__ __device__size_t extension_degree() { return 6; }
};

template<mp_size_t_ n> 
__noinline__ __device__ Fp6_2over3_model<n> operator*(const Fp_model<n>& lhs, const Fp6_2over3_model<n>& rhs);

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp6_2over3_model<n> operator^(const Fp6_2over3_model<n>& self, const bigint<m>& exponent);

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp6_2over3_model<n> operator^(const Fp6_2over3_model<n>& self, const Fp_model<m>& exponent);

}

#include "fp6_2over3.cu"

#endif