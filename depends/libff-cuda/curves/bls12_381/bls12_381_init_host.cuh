#ifndef __BLS12_381_INIT_HOST_CUH__
#define __BLS12_381_INIT_HOST_CUH__

#include "../../fields/fp_host.cuh"
#include "../../fields/fp2_host.cuh"
#include "../../fields/fp6_3over2_host.cuh"
#include "../../fields/fp12_2over3over2_host.cuh"


namespace libff {
static const mp_size_t bls12_381_r_bitcount_host = 255;
static const mp_size_t bls12_381_q_bitcount_host = 381;

// __device__ static const mp_size_t_ GMP_NUMB_BITS_ = sizeof(mp_limb_t_) * 8;
static const mp_size_t bls12_381_r_limbs_host = (bls12_381_r_bitcount_host + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
static const mp_size_t bls12_381_q_limbs_host = (bls12_381_q_bitcount_host + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;

extern Fp_params_host<bls12_381_r_limbs_host> bls12_381_fp_params_r_host;
extern Fp_params_host<bls12_381_q_limbs_host> bls12_381_fp_params_q_host;
extern Fp2_params_host<bls12_381_q_limbs_host> bls12_381_fp2_params_q_host;
extern Fp6_3over2_params_host<bls12_381_q_limbs_host> bls12_381_fp6_params_q_host;
extern Fp12_params_host<bls12_381_q_limbs_host> bls12_381_fp12_params_q_host;

typedef Fp_model_host<bls12_381_r_limbs_host> bls12_381_Fr_host;
typedef Fp_model_host<bls12_381_q_limbs_host> bls12_381_Fq_host;
typedef Fp2_model_host<bls12_381_q_limbs_host> bls12_381_Fq2_host;
typedef Fp6_3over2_model_host<bls12_381_q_limbs_host> bls12_381_Fq6_host;
typedef Fp12_2over3over2_model_host<bls12_381_q_limbs_host> bls12_381_Fq12_host;
typedef bls12_381_Fq12_host bls12_381_GT_host;

// parameters for the curve E/Fq : y^2 = x^3 + b
extern  bls12_381_Fq_host* bls12_381_coeff_b_host;
// parameters for the twisted curve E'/Fq2 : y^2 = x^3 + b/xi
extern  bls12_381_Fq2_host* bls12_381_twist_host;
extern  bls12_381_Fq2_host* bls12_381_twist_coeff_b_host;
extern  bls12_381_Fq_host* bls12_381_twist_mul_by_b_c0_host;
extern  bls12_381_Fq_host* bls12_381_twist_mul_by_b_c1_host;
extern  bls12_381_Fq2_host* bls12_381_twist_mul_by_q_X_host;
extern  bls12_381_Fq2_host* bls12_381_twist_mul_by_q_Y_host;

struct bls12_381_G1_params_host;
struct bls12_381_G2_params_host;

extern  bls12_381_G1_params_host g1_params_host;
extern  bls12_381_G2_params_host g2_params_host;

// // parameters for pairing
// extern __device__ bigint<bls12_381_q_limbs>* bls12_381_ate_loop_count;
// extern __device__ bool bls12_381_ate_is_loop_count_neg;
// extern __device__ bigint<12 * bls12_381_q_limbs>* bls12_381_final_exponent;
// extern __device__ bigint<bls12_381_q_limbs>* bls12_381_final_exponent_z;
// extern __device__ bool bls12_381_final_exponent_is_z_neg;

void init_bls12_381_params_host();

class bls12_381_G1_host;
class bls12_381_G2_host;
}

#endif