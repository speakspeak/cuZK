#ifndef __MNT4753_INIT_CUH__
#define __MNT4753_INIT_CUH__

#include "../../fields/fp.cuh"
#include "../../fields/fp2.cuh"
#include "../../fields/fp4.cuh"

namespace libff {

__device__ static const mp_size_t_ mnt4753_r_bitcount = 753;
__device__ static const mp_size_t_ mnt4753_q_bitcount = 753;

__device__ static const mp_size_t_ GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;

__device__ static const mp_size_t_ mnt4753_r_limbs = (mnt4753_r_bitcount+GMP_NUMB_BITS_-1)/GMP_NUMB_BITS_;
__device__ static const mp_size_t_ mnt4753_q_limbs = (mnt4753_q_bitcount+GMP_NUMB_BITS_-1)/GMP_NUMB_BITS_;

extern __device__ Fp_params<mnt4753_r_limbs> mnt4753_fp_params_r;
extern __device__ Fp_params<mnt4753_q_limbs> mnt4753_fp_params_q;
extern __device__ Fp2_params<mnt4753_q_limbs> mnt4753_fp2_params_q;
extern __device__ Fp4_params<mnt4753_q_limbs> mnt4753_fp4_params_q;

typedef Fp_model<mnt4753_r_limbs> mnt4753_Fr;
typedef Fp_model<mnt4753_q_limbs> mnt4753_Fq;
typedef Fp2_model<mnt4753_q_limbs> mnt4753_Fq2;
typedef Fp4_model<mnt4753_q_limbs> mnt4753_Fq4;
typedef mnt4753_Fq4 mnt4753_GT;

// parameters for twisted short Weierstrass curve E'/Fq2 : y^2 = x^3 + (a * twist^2) * x + (b * twist^3)
extern __device__ mnt4753_Fq2* mnt4753_twist;
extern __device__ mnt4753_Fq2* mnt4753_twist_coeff_a;
extern __device__ mnt4753_Fq2* mnt4753_twist_coeff_b;
extern __device__ mnt4753_Fq* mnt4753_twist_mul_by_a_c0;
extern __device__ mnt4753_Fq* mnt4753_twist_mul_by_a_c1;
extern __device__ mnt4753_Fq* mnt4753_twist_mul_by_b_c0;
extern __device__ mnt4753_Fq* mnt4753_twist_mul_by_b_c1;
extern __device__ mnt4753_Fq* mnt4753_twist_mul_by_q_X;
extern __device__ mnt4753_Fq* mnt4753_twist_mul_by_q_Y;

// #include "mnt4753_g1.cuh"
// #include "mnt4753_g2.cuh"

struct mnt4753_G1_params;
struct mnt4753_G2_params;

extern __device__ mnt4753_G1_params g1_params;
extern __device__ mnt4753_G2_params g2_params;

// parameters for pairing
extern __device__ bigint<mnt4753_q_limbs>* mnt4753_ate_loop_count;
extern __device__ bool mnt4753_ate_is_loop_count_neg;
extern __device__ bigint<4*mnt4753_q_limbs>* mnt4753_final_exponent;
extern __device__ bigint<mnt4753_q_limbs>* mnt4753_final_exponent_last_chunk_abs_of_w0;
extern __device__ bool mnt4753_final_exponent_last_chunk_is_w0_neg;
extern __device__ bigint<mnt4753_q_limbs>* mnt4753_final_exponent_last_chunk_w1;

__device__ void init_mnt4753_params();

class mnt4753_G1;
class mnt4753_G2;

}

#include "mnt4753_init.cu"

#endif