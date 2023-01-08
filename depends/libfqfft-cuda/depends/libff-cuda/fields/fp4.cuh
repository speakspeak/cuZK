#ifndef __FP4_CUH__
#define __FP4_CUH__

#include "bigint.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"
#include "../exponentiation/exponentiation.cuh"
#include "fp.cuh"
#include "fp2.cuh"

namespace libff {

template<mp_size_t_ n>
struct Fp4_params
{
    Fp2_params<n>* fp2_params;
    bigint<n>* non_residue;
    bigint<n>* Frobenius_coeffs_c1[4];
};


template<mp_size_t_ n> 
class Fp4_model {
public:
    typedef Fp_model<n> my_Fp;
    typedef Fp2_model<n> my_Fp2;

    Fp4_params<n>* params;

    my_Fp2 c0, c1;
    
    __noinline__ __device__ Fp4_model() : params(nullptr){}
    __noinline__ __device__ Fp4_model(Fp4_params<n>* params) : params(params), c0(params->fp2_params), c1(params->fp2_params) {};
    __noinline__ __device__ Fp4_model(Fp4_params<n>* params, const my_Fp2& c0, const my_Fp2& c1) : params(params), c0(c0), c1(c1) {};
    __noinline__ __device__ Fp4_model(const Fp4_model<n>& other) :params(other.params), c0(other.c0), c1(other.c1) {};

    __noinline__ __device__ Fp4_model<n>& operator=(const Fp4_model<n>& other) = default;

    __noinline__ __device__ bool operator==(const Fp4_model<n>& other) const;
    __noinline__ __device__ bool operator!=(const Fp4_model<n>& other) const;

    __noinline__ __device__ Fp4_model<n> operator+(const Fp4_model<n>& other) const;
    __noinline__ __device__ Fp4_model<n> operator-(const Fp4_model<n>& other) const;
    __noinline__ __device__ Fp4_model<n> operator*(const Fp4_model<n>& other) const;
    __noinline__ __device__ Fp4_model<n> operator-() const;

    // template<mp_size_t_ m>
    // __noinline__ __device__ Fp4_model<n> operator^(const bigint<m>& other) const;

    __noinline__ __device__ my_Fp2 mul_by_non_residue(const my_Fp2& elt) const;
    __noinline__ __device__ Fp4_model<n> mul_by_023(const Fp4_model& other) const;

    __noinline__ __device__ Fp4_model<n> dbl() const;
    __noinline__ __device__ Fp4_model<n> squared() const;
    __noinline__ __device__ Fp4_model<n> inverse() const;
    __noinline__ __device__ Fp4_model<n> Frobenius_map(unsigned long power) const;
    __noinline__ __device__ Fp4_model<n> unitary_inverse() const;
    __noinline__ __device__ Fp4_model<n> cyclotomic_squared() const;

    template<mp_size_t_ m>
    __noinline__ __device__ Fp4_model<n> cyclotomic_exp(const bigint<m>& exponent) const;

    __noinline__ __device__ Fp4_model<n> zero();
    __noinline__ __device__ Fp4_model<n> one();

    __noinline__ __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero();  }
    __noinline__ __device__ void clear() { c0.clear(); c1.clear(); }

    __noinline__ __device__ void set_params(Fp4_params<n>* params);

    __noinline__ __device__ bigint<n> base_field_char() { return *params->fp2_params->fp_params->modulus; }
    __noinline__ __device__ size_t extension_degree() { return 4; }
};

template<mp_size_t_ n> 
__noinline__ __device__ Fp4_model<n> operator*(const Fp_model<n>& lhs, const Fp4_model<n>& rhs);

template<mp_size_t_ n> 
__noinline__ __device__ Fp4_model<n> operator*(const Fp2_model<n>& lhs, const Fp4_model<n>& rhs);

// template<mp_size_t_ n, mp_size_t_ m>
// __noinline__ __device__ Fp4_model<n> cyclotomic_exp(const Fp4_model<n>& self, bigint<m>& exponent);

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp4_model<n> operator^(const Fp4_model<n>& self, const bigint<m>& exponent);

template<mp_size_t_ n, mp_size_t_ m>
__noinline__ __device__ Fp4_model<n> operator^(const Fp4_model<n>& self, const Fp_model<m>& exponent);

}

#include "fp4.cu"

#endif