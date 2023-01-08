#ifndef __MNT4_INIT_CUH__
#define __MNT4_INIT_CUH__

#include "../../fields/fp.cuh"
#include "../../fields/fp2.cuh"
#include "../../fields/fp4.cuh"

namespace libff {

__device__ static const mp_size_t_ mnt4_r_bitcount = 298;
__device__ static const mp_size_t_ mnt4_q_bitcount = 298;

__device__ static const mp_size_t_ GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;

__device__ static const mp_size_t_ mnt4_r_limbs = (mnt4_r_bitcount+GMP_NUMB_BITS_-1)/GMP_NUMB_BITS_;
__device__ static const mp_size_t_ mnt4_q_limbs = (mnt4_q_bitcount+GMP_NUMB_BITS_-1)/GMP_NUMB_BITS_;

extern __device__ Fp_params<mnt4_r_limbs> mnt4_fp_params_r;
extern __device__ Fp_params<mnt4_q_limbs> mnt4_fp_params_q;
extern __device__ Fp2_params<mnt4_q_limbs> mnt4_fp2_params_q;
extern __device__ Fp4_params<mnt4_q_limbs> mnt4_fp4_params_q;

typedef Fp_model<mnt4_r_limbs> mnt4_Fr;
typedef Fp_model<mnt4_q_limbs> mnt4_Fq;
typedef Fp2_model<mnt4_q_limbs> mnt4_Fq2;
typedef Fp4_model<mnt4_q_limbs> mnt4_Fq4;
typedef mnt4_Fq4 mnt4_GT;

// parameters for twisted short Weierstrass curve E'/Fq2 : y^2 = x^3 + (a * twist^2) * x + (b * twist^3)
extern __device__ mnt4_Fq2* mnt4_twist;
extern __device__ mnt4_Fq2* mnt4_twist_coeff_a;
extern __device__ mnt4_Fq2* mnt4_twist_coeff_b;
extern __device__ mnt4_Fq* mnt4_twist_mul_by_a_c0;
extern __device__ mnt4_Fq* mnt4_twist_mul_by_a_c1;
extern __device__ mnt4_Fq* mnt4_twist_mul_by_b_c0;
extern __device__ mnt4_Fq* mnt4_twist_mul_by_b_c1;
extern __device__ mnt4_Fq* mnt4_twist_mul_by_q_X;
extern __device__ mnt4_Fq* mnt4_twist_mul_by_q_Y;

// #include "mnt4_g1.cuh"
// #include "mnt4_g2.cuh"

struct mnt4_G1_params;
struct mnt4_G2_params;

extern __device__ mnt4_G1_params g1_params;
extern __device__ mnt4_G2_params g2_params;

// parameters for pairing
extern __device__ bigint<mnt4_q_limbs>* mnt4_ate_loop_count;
extern __device__ bool mnt4_ate_is_loop_count_neg;
extern __device__ bigint<4*mnt4_q_limbs>* mnt4_final_exponent;
extern __device__ bigint<mnt4_q_limbs>* mnt4_final_exponent_last_chunk_abs_of_w0;
extern __device__ bool mnt4_final_exponent_last_chunk_is_w0_neg;
extern __device__ bigint<mnt4_q_limbs>* mnt4_final_exponent_last_chunk_w1;

__device__ void init_mnt4_params();

class mnt4_G1;
class mnt4_G2;

}

#include "mnt4_init.cu"

#endif