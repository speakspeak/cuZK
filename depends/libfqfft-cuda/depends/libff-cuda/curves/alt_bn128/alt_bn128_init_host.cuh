#ifndef __ALT_BN128_INIT_HOST_CUH__
#define __ALT_BN128_INIT_HOST_CUH__

#include "../../fields/fp_host.cuh"
#include "../../fields/fp2_host.cuh"
#include "../../fields/fp6_3over2_host.cuh"
#include "../../fields/fp12_2over3over2_host.cuh"

namespace libff {

static const mp_size_t alt_bn128_r_bitcount_host = 254;
static const mp_size_t alt_bn128_q_bitcount_host = 254;

// __device__ static const mp_size_t GMP_NUMB_BITS_ = sizeof(mp_limb_t) * 8;
static const mp_size_t alt_bn128_r_limbs_host = (alt_bn128_r_bitcount_host + GMP_NUMB_BITS-1)/GMP_NUMB_BITS;
static const mp_size_t alt_bn128_q_limbs_host = (alt_bn128_q_bitcount_host + GMP_NUMB_BITS-1)/GMP_NUMB_BITS;

extern  Fp_params_host<alt_bn128_r_limbs_host> alt_bn128_fp_params_r_host;
extern  Fp_params_host<alt_bn128_q_limbs_host> alt_bn128_fp_params_q_host;
extern  Fp2_params_host<alt_bn128_q_limbs_host> alt_bn128_fp2_params_q_host;
extern  Fp6_3over2_params_host<alt_bn128_q_limbs_host> alt_bn128_fp6_params_q_host;
extern  Fp12_params_host<alt_bn128_q_limbs_host> alt_bn128_fp12_params_q_host;

typedef Fp_model_host<alt_bn128_r_limbs_host> alt_bn128_Fr_host;
typedef Fp_model_host<alt_bn128_q_limbs_host> alt_bn128_Fq_host;
typedef Fp2_model_host<alt_bn128_q_limbs_host> alt_bn128_Fq2_host;
typedef Fp6_3over2_model_host<alt_bn128_q_limbs_host> alt_bn128_Fq6_host;
typedef Fp12_2over3over2_model_host<alt_bn128_q_limbs_host> alt_bn128_Fq12_host;
typedef alt_bn128_Fq12_host alt_bn128_GT_host;

// parameters for Barreto--Naehrig curve E/Fq : y^2 = x^3 + b
extern  alt_bn128_Fq_host* alt_bn128_coeff_b_host;
// parameters for twisted Barreto--Naehrig curve E'/Fq2 : y^2 = x^3 + b/xi
extern  alt_bn128_Fq2_host* alt_bn128_twist_host;
extern  alt_bn128_Fq2_host* alt_bn128_twist_coeff_b_host;
extern  alt_bn128_Fq_host* alt_bn128_twist_mul_by_b_c0_host;
extern  alt_bn128_Fq_host* alt_bn128_twist_mul_by_b_c1_host;
extern  alt_bn128_Fq2_host* alt_bn128_twist_mul_by_q_X_host;
extern  alt_bn128_Fq2_host* alt_bn128_twist_mul_by_q_Y_host;

struct alt_bn128_G1_params_host;
struct alt_bn128_G2_params_host;

extern  alt_bn128_G1_params_host g1_params_host;
extern  alt_bn128_G2_params_host g2_params_host;

// // parameters for pairing
// extern  bigint<alt_bn128_q_limbs>* alt_bn128_ate_loop_count;
// extern  bool alt_bn128_ate_is_loop_count_neg;
// extern  bigint<12*alt_bn128_q_limbs>* alt_bn128_final_exponent;
// extern  bigint<alt_bn128_q_limbs>* alt_bn128_final_exponent_z;
// extern  bool alt_bn128_final_exponent_is_z_neg;

void init_alt_bn128_params_host();

class alt_bn128_G1_host;
class alt_bn128_G2_host;

}

#endif