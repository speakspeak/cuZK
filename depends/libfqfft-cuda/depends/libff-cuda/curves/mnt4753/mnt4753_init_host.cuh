#ifndef __MNT4753_INIT_HOST_CUH__
#define __MNT4753_INIT_HOST_CUH__

#include "../../fields/fp_host.cuh"
#include "../../fields/fp2_host.cuh"
#include "../../fields/fp4_host.cuh"

namespace libff {

 static const mp_size_t mnt4753_r_bitcount_host = 753;
 static const mp_size_t mnt4753_q_bitcount_host = 753;

// __device__ static const mp_size_t GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;

 static const mp_size_t mnt4753_r_limbs_host = (mnt4753_r_bitcount_host + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
 static const mp_size_t mnt4753_q_limbs_host = (mnt4753_q_bitcount_host + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;

extern  Fp_params_host<mnt4753_r_limbs_host> mnt4753_fp_params_r_host;
extern  Fp_params_host<mnt4753_q_limbs_host> mnt4753_fp_params_q_host;
extern  Fp2_params_host<mnt4753_q_limbs_host> mnt4753_fp2_params_q_host;
extern  Fp4_params_host<mnt4753_q_limbs_host> mnt4753_fp4_params_q_host;

typedef Fp_model_host<mnt4753_r_limbs_host> mnt4753_Fr_host;
typedef Fp_model_host<mnt4753_q_limbs_host> mnt4753_Fq_host;
typedef Fp2_model_host<mnt4753_q_limbs_host> mnt4753_Fq2_host;
typedef Fp4_model_host<mnt4753_q_limbs_host> mnt4753_Fq4_host;
typedef mnt4753_Fq4_host mnt4753_GT_host;

// parameters for twisted short Weierstrass curve E'/Fq2 : y^2 = x^3 + (a * twist^2) * x + (b * twist^3)
extern  mnt4753_Fq2_host* mnt4753_twist_host;
extern  mnt4753_Fq2_host* mnt4753_twist_coeff_a_host;
extern  mnt4753_Fq2_host* mnt4753_twist_coeff_b_host;
extern  mnt4753_Fq_host* mnt4753_twist_mul_by_a_c0_host;
extern  mnt4753_Fq_host* mnt4753_twist_mul_by_a_c1_host;
extern  mnt4753_Fq_host* mnt4753_twist_mul_by_b_c0_host;
extern  mnt4753_Fq_host* mnt4753_twist_mul_by_b_c1_host;
extern  mnt4753_Fq_host* mnt4753_twist_mul_by_q_X_host;
extern  mnt4753_Fq_host* mnt4753_twist_mul_by_q_Y_host;

// #include "mnt4753_g1.cuh"
// #include "mnt4753_g2.cuh"

struct mnt4753_G1_params_host;
struct mnt4753_G2_params_host;

extern  mnt4753_G1_params_host g1_params_host;
extern  mnt4753_G2_params_host g2_params_host;

// // parameters for pairing
// extern __device__ bigint<mnt4753_q_limbs>* mnt4753_ate_loop_count;
// extern __device__ bool mnt4753_ate_is_loop_count_neg;
// extern __device__ bigint<4*mnt4753_q_limbs>* mnt4753_final_exponent;
// extern __device__ bigint<mnt4753_q_limbs>* mnt4753_final_exponent_last_chunk_abs_of_w0;
// extern __device__ bool mnt4753_final_exponent_last_chunk_is_w0_neg;
// extern __device__ bigint<mnt4753_q_limbs>* mnt4753_final_exponent_last_chunk_w1;

void init_mnt4753_params_host();

class mnt4753_G1_host;
class mnt4753_G2_host;

}

#endif