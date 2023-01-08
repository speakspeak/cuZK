#ifndef __ALT_BN128_PP_HOST_CUH__
#define __ALT_BN128_PP_HOST_CUH__

#include "alt_bn128_g1_host.cuh"
#include "alt_bn128_g2_host.cuh"
#include "alt_bn128_init_host.cuh"
// #include "alt_bn128_pairing_host.cuh"
#include "../public_params.cuh"

namespace libff {

class alt_bn128_pp_host {
public:
    typedef alt_bn128_Fr_host Fp_type;
    typedef alt_bn128_G1_host G1_type;
    typedef alt_bn128_G2_host G2_type;
    // typedef alt_bn128_G1_precomp_host G1_precomp_type;
    // typedef alt_bn128_G2_precomp_host G2_precomp_type;
    typedef alt_bn128_Fq_host Fq_type;
    typedef alt_bn128_Fq2_host Fqe_type;
    typedef alt_bn128_Fq12_host Fqk_type;
    typedef alt_bn128_GT_host GT_type;

    // static const bool has_affine_pairing = false;

    static void init_public_params();
    // __device__ static alt_bn128_GT final_exponentiation(const alt_bn128_Fq12 &elt);
    // __device__ static alt_bn128_G1_precomp precompute_G1(const alt_bn128_G1 &P);
    // __device__ static alt_bn128_G2_precomp precompute_G2(const alt_bn128_G2 &Q);
    // __device__ static alt_bn128_Fq12 miller_loop(const alt_bn128_G1_precomp &prec_P,const alt_bn128_G2_precomp &prec_Q);
                                      
    // __device__ static alt_bn128_Fq12 double_miller_loop(const alt_bn128_G1_precomp &prec_P1,
    //                                                     const alt_bn128_G2_precomp &prec_Q1,
    //                                                     const alt_bn128_G1_precomp &prec_P2,
    //                                                     const alt_bn128_G2_precomp &prec_Q2);
    // __device__ static alt_bn128_Fq12 pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);
    // __device__ static alt_bn128_Fq12 reduced_pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);
};

}

#endif
