#ifndef __BLS12_381_PP_HOST_CUH__
#define __BLS12_381_PP_HOST_CUH__

#include "bls12_381_g1_host.cuh"
#include "bls12_381_g2_host.cuh"
#include "bls12_381_init_host.cuh"
// #include "bls12_381_pairing_host.cuh"
#include "../public_params.cuh"

namespace libff {

class bls12_381_pp_host {
public:
    typedef bls12_381_Fr_host Fp_type;
    typedef bls12_381_G1_host G1_type;
    typedef bls12_381_G2_host G2_type;
    // typedef bls12_381_G1_precomp_host G1_precomp_type;
    // typedef bls12_381_G2_precomp_host G2_precomp_type;
    typedef bls12_381_Fq_host Fq_type;
    typedef bls12_381_Fq2_host Fqe_type;
    typedef bls12_381_Fq12_host Fqk_type;
    typedef bls12_381_GT_host GT_type;

    // static const bool has_affine_pairing = false;

    static void init_public_params();
    // __device__ static bls12_381_GT final_exponentiation(const bls12_381_Fq12 &elt);
    // __device__ static bls12_381_G1_precomp precompute_G1(const bls12_381_G1 &P);
    // __device__ static bls12_381_G2_precomp precompute_G2(const bls12_381_G2 &Q);
    // __device__ static bls12_381_Fq12 miller_loop(const bls12_381_G1_precomp &prec_P,const bls12_381_G2_precomp &prec_Q);
                                      
    // __device__ static bls12_381_Fq12 double_miller_loop(const bls12_381_G1_precomp &prec_P1,
    //                                                     const bls12_381_G2_precomp &prec_Q1,
    //                                                     const bls12_381_G1_precomp &prec_P2,
    //                                                     const bls12_381_G2_precomp &prec_Q2);
    // __device__ static bls12_381_Fq12 pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);
    // __device__ static bls12_381_Fq12 reduced_pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);
};

}

#endif
