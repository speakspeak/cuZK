#ifndef __BLS12_381_INIT_CUH__
#define __BLS12_381_INIT_CUH__

#include "../../fields/fp.cuh"
#include "../../fields/fp2.cuh"
#include "../../fields/fp6_3over2.cuh"
#include "../../fields/fp12_2over3over2.cuh"

namespace libff {

__device__ static const mp_size_t_ bls12_381_r_bitcount = 255;
__device__ static const mp_size_t_ bls12_381_q_bitcount = 381;

__device__ static const mp_size_t_ GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;

__device__ static const mp_size_t_ bls12_381_r_limbs = (bls12_381_r_bitcount + GMP_NUMB_BITS_ - 1) / GMP_NUMB_BITS_;
__device__ static const mp_size_t_ bls12_381_q_limbs = (bls12_381_q_bitcount + GMP_NUMB_BITS_ - 1) / GMP_NUMB_BITS_;

extern __device__ Fp_params<bls12_381_r_limbs> bls12_381_fp_params_r;
extern __device__ Fp_params<bls12_381_q_limbs> bls12_381_fp_params_q;
extern __device__ Fp2_params<bls12_381_q_limbs> bls12_381_fp2_params_q;
extern __device__ Fp6_3over2_params<bls12_381_q_limbs> bls12_381_fp6_params_q;
extern __device__ Fp12_params<bls12_381_q_limbs> bls12_381_fp12_params_q;

typedef Fp_model<bls12_381_r_limbs> bls12_381_Fr;
typedef Fp_model<bls12_381_q_limbs> bls12_381_Fq;
typedef Fp2_model<bls12_381_q_limbs> bls12_381_Fq2;
typedef Fp6_3over2_model<bls12_381_q_limbs> bls12_381_Fq6;
typedef Fp12_2over3over2_model<bls12_381_q_limbs> bls12_381_Fq12;
typedef bls12_381_Fq12 bls12_381_GT;

// parameters for the curve E/Fq : y^2 = x^3 + b
extern __device__ bls12_381_Fq* bls12_381_coeff_b;
// parameters for the twisted curve E'/Fq2 : y^2 = x^3 + b/xi
extern __device__ bls12_381_Fq2* bls12_381_twist;
extern __device__ bls12_381_Fq2* bls12_381_twist_coeff_b;
extern __device__ bls12_381_Fq* bls12_381_twist_mul_by_b_c0;
extern __device__ bls12_381_Fq* bls12_381_twist_mul_by_b_c1;
extern __device__ bls12_381_Fq2* bls12_381_twist_mul_by_q_X;
extern __device__ bls12_381_Fq2* bls12_381_twist_mul_by_q_Y;

struct bls12_381_G1_params;
struct bls12_381_G2_params;

extern __device__ bls12_381_G1_params g1_params;
extern __device__ bls12_381_G2_params g2_params;

// parameters for pairing
extern __device__ bigint<bls12_381_q_limbs>* bls12_381_ate_loop_count;
extern __device__ bool bls12_381_ate_is_loop_count_neg;
extern __device__ bigint<12 * bls12_381_q_limbs>* bls12_381_final_exponent;
extern __device__ bigint<bls12_381_q_limbs>* bls12_381_final_exponent_z;
extern __device__ bool bls12_381_final_exponent_is_z_neg;

__device__ void init_bls12_381_params();

class bls12_381_G1;
class bls12_381_G2;

}

#include "bls12_381_init.cu"

#endif
