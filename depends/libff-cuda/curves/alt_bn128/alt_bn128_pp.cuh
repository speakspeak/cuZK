#ifndef __ALT_BN128_PP_CUH__
#define __ALT_BN128_PP_CUH__

#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
#include "alt_bn128_init.cuh"
#include "alt_bn128_pairing.cuh"
#include "../public_params.cuh"

namespace libff {

class alt_bn128_pp {
public:
    typedef alt_bn128_Fr Fp_type;
    typedef alt_bn128_G1 G1_type;
    typedef alt_bn128_G2 G2_type;
    typedef alt_bn128_G1_precomp G1_precomp_type;
    typedef alt_bn128_G2_precomp G2_precomp_type;
    typedef alt_bn128_Fq Fq_type;
    typedef alt_bn128_Fq2 Fqe_type;
    typedef alt_bn128_Fq12 Fqk_type;
    typedef alt_bn128_GT GT_type;

    static const bool has_affine_pairing = false;

    __device__ static void init_public_params();
    __device__ static alt_bn128_GT final_exponentiation(const alt_bn128_Fq12 &elt);
    __device__ static alt_bn128_G1_precomp precompute_G1(const alt_bn128_G1 &P);
    __device__ static alt_bn128_G2_precomp precompute_G2(const alt_bn128_G2 &Q);
    __device__ static alt_bn128_Fq12 miller_loop(const alt_bn128_G1_precomp &prec_P,const alt_bn128_G2_precomp &prec_Q);
                                      
    __device__ static alt_bn128_Fq12 double_miller_loop(const alt_bn128_G1_precomp &prec_P1,
                                                        const alt_bn128_G2_precomp &prec_Q1,
                                                        const alt_bn128_G1_precomp &prec_P2,
                                                        const alt_bn128_G2_precomp &prec_Q2);
    __device__ static alt_bn128_Fq12 pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);
    __device__ static alt_bn128_Fq12 reduced_pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);
};

}

#include "alt_bn128_pp.cu"

#endif
