#include "../scalar_multiplication/wnaf.cuh"
#include "../scalar_multiplication/multiexp.cuh"
#include "../curves/alt_bn128/alt_bn128_pairing.cuh"
#include "../curves/alt_bn128/alt_bn128_init.cuh"
#include "../curves/alt_bn128/alt_bn128_g1.cuh"
#include "../curves/alt_bn128/alt_bn128_g2.cuh"
#include "../curves/alt_bn128/alt_bn128_pp.cuh"
#include "../fields/bigint.cuh"
#include "../fields/fp.cuh"
#include "../fields/fp2.cuh"
#include "../fields/fp3.cuh"
#include "../fields/fp4.cuh"
#include "../fields/fp6_2over3.cuh"
#include "../fields/fp6_3over2.cuh"
#include "../fields/fp12_2over3over2.cuh"

#include <stdio.h>

__device__ static const mp_size_t_ n = 8;

__device__ Fp_params<n> fp_params;
__device__ Fp2_params<n> fp2_params;
__device__ Fp3_params<n> fp3_params;
__device__ Fp4_params<n> fp4_params;
__device__ Fp6_2over3_params<n> fp6_2over3_params;
__device__ Fp6_3over2_params<n> fp6_3over2_params;

// 
// {8,
//     "21888242871839275222246405745257275088548364400416034343698204186575808495617",
//     254,  \
//     "10944121435919637611123202872628637544274182200208017171849102093287904247808", \
//     28, \
//     "81540058820840996586704275553141814055101440848469862132140264610111", \
//     "40770029410420498293352137776570907027550720424234931066070132305055",   \
//     "5",  \
//     "19103219067921713944291392827692070036145651957329286315305642004821462161904",  \
//     "5",  \
//     "19103219067921713944291392827692070036145651957329286315305642004821462161904",  \
//     0xefffffff,     \
//     "944936681149208446651664254269745548490766851729442924617792859073125903783",   \
//     "5866548545943845227489894872040244720403868105578784105281690076696998248512" \
//     };

__device__ void init()
{

    bigint<n>* modulus= new bigint<n>("21888242871839275222246405745257275088548364400416034343698204186575808495617");
    bigint<n>* euler = new bigint<n>("10944121435919637611123202872628637544274182200208017171849102093287904247808");
    bigint<n>* t = new bigint<n>("81540058820840996586704275553141814055101440848469862132140264610111");
    bigint<n>* t_minus_1_over_2 = new bigint<n>("40770029410420498293352137776570907027550720424234931066070132305055");
    bigint<n>* nqr = new bigint<n>("5");
    bigint<n>* nqr_to_t = new bigint<n>("19103219067921713944291392827692070036145651957329286315305642004821462161904");
    bigint<n>* multiplicative_generator = new bigint<n>("5");
    bigint<n>* root_of_unity = new bigint<n>("19103219067921713944291392827692070036145651957329286315305642004821462161904");
    bigint<n>* Rsquared = new bigint<n>("944936681149208446651664254269745548490766851729442924617792859073125903783");
    bigint<n>* Rcubed = new bigint<n>("5866548545943845227489894872040244720403868105578784105281690076696998248512");
    
    fp_params.modulus = modulus;    //
    fp_params.euler = euler;
    fp_params.t = t;
    fp_params.t_minus_1_over_2 = t_minus_1_over_2;
    fp_params.nqr = nqr;
    fp_params.nqr_to_t = nqr_to_t;
    fp_params.multiplicative_generator = multiplicative_generator;
    fp_params.root_of_unity = root_of_unity;
    fp_params.Rsquared = Rsquared;
    fp_params.Rcubed = Rcubed;

    fp_params.num_limbs = 8;
    fp_params.num_bits = 254;
    fp_params.s = 28;
    fp_params.inv = 0xefffffff;
}

template<mp_size_t_ n, const Fp_params<n>& fp_params>
__device__ void axy()
{

    Fp12_2over3over2_model<alt_bn128_q_limbs, alt_bn128_fp_params_q, alt_bn128_fp2_params_q, alt_bn128_fp6_params_q, alt_bn128_fp12_params_q> tt;
    // Fp4_model<n, fp_params, fp2_params, fp4_params> tt;
    // Fp6_2over3_model<n, fp_params, fp2_params, fp3_params, fp6_2over3_params> tt;
    tt.inverse();

    Fp_model<n, fp_params> f(-7);
    printf("final value: %d\n", f.as_ulong());
    Fp_model<n, fp_params> s = f.inverse();
    Fp_model<n, fp_params> t = s * f;

    Fp_model<n, fp_params> tf = -f;
    // tf = -tf;

    printf("%d\n", s.as_ulong());
    printf("%d\n", t.as_ulong());
    printf("%d\n", tf.as_ulong());
}


template<mp_size_t_ n, const Fp_params<n>& fp_params, const Fp2_params<n>& fp2_params> 
__device__ void bxy()
{

    Fp2_model<n, fp_params, fp2_params> f;
}



__global__ void warmup(void) {
    init();
    // init_alt_bn128_params();
    // alt_bn128_pp t;
    alt_bn128_pp::init_public_params();
    printf("%d\n", alt_bn128_fp_params_r.num_bits);  
    axy<n, fp_params>();
    // bxy<n, fp_params, fp2_params>();
}


int main()
{
    warmup<<<1, 1>>>();
}
