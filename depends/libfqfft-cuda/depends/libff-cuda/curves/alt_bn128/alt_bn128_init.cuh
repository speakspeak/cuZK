#ifndef __ALT_BN128_INIT_CUH__
#define __ALT_BN128_INIT_CUH__

#include "../../fields/fp.cuh"
#include "../../fields/fp2.cuh"
#include "../../fields/fp6_3over2.cuh"
#include "../../fields/fp12_2over3over2.cuh"

namespace libff {

__device__ static const mp_size_t_ alt_bn128_r_bitcount = 254;
__device__ static const mp_size_t_ alt_bn128_q_bitcount = 254;

__device__ static const mp_size_t_ GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;

__device__ static const mp_size_t_ alt_bn128_r_limbs = (alt_bn128_r_bitcount+GMP_NUMB_BITS_-1)/GMP_NUMB_BITS_;
__device__ static const mp_size_t_ alt_bn128_q_limbs = (alt_bn128_q_bitcount+GMP_NUMB_BITS_-1)/GMP_NUMB_BITS_;

extern __device__ Fp_params<alt_bn128_r_limbs> alt_bn128_fp_params_r;
extern __device__ Fp_params<alt_bn128_q_limbs> alt_bn128_fp_params_q;
extern __device__ Fp2_params<alt_bn128_q_limbs> alt_bn128_fp2_params_q;
extern __device__ Fp6_3over2_params<alt_bn128_q_limbs> alt_bn128_fp6_params_q;
extern __device__ Fp12_params<alt_bn128_q_limbs> alt_bn128_fp12_params_q;

typedef Fp_model<alt_bn128_r_limbs> alt_bn128_Fr;
typedef Fp_model<alt_bn128_q_limbs> alt_bn128_Fq;
typedef Fp2_model<alt_bn128_q_limbs> alt_bn128_Fq2;
typedef Fp6_3over2_model<alt_bn128_q_limbs> alt_bn128_Fq6;
typedef Fp12_2over3over2_model<alt_bn128_q_limbs> alt_bn128_Fq12;
typedef alt_bn128_Fq12 alt_bn128_GT;

// parameters for Barreto--Naehrig curve E/Fq : y^2 = x^3 + b
extern __device__ alt_bn128_Fq* alt_bn128_coeff_b;
// parameters for twisted Barreto--Naehrig curve E'/Fq2 : y^2 = x^3 + b/xi
extern __device__ alt_bn128_Fq2* alt_bn128_twist;
extern __device__ alt_bn128_Fq2* alt_bn128_twist_coeff_b;
extern __device__ alt_bn128_Fq* alt_bn128_twist_mul_by_b_c0;
extern __device__ alt_bn128_Fq* alt_bn128_twist_mul_by_b_c1;
extern __device__ alt_bn128_Fq2* alt_bn128_twist_mul_by_q_X;
extern __device__ alt_bn128_Fq2* alt_bn128_twist_mul_by_q_Y;

struct alt_bn128_G1_params;
struct alt_bn128_G2_params;

extern __device__ alt_bn128_G1_params g1_params;
extern __device__ alt_bn128_G2_params g2_params;

// parameters for pairing
extern __device__ bigint<alt_bn128_q_limbs>* alt_bn128_ate_loop_count;
extern __device__ bool alt_bn128_ate_is_loop_count_neg;
extern __device__ bigint<12*alt_bn128_q_limbs>* alt_bn128_final_exponent;
extern __device__ bigint<alt_bn128_q_limbs>* alt_bn128_final_exponent_z;
extern __device__ bool alt_bn128_final_exponent_is_z_neg;

__device__ void init_alt_bn128_params();

class alt_bn128_G1;
class alt_bn128_G2;

}

#include "alt_bn128_init.cu"

#endif