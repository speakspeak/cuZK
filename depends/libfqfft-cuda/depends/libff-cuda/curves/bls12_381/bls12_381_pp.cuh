#ifndef __BLS12_381_PP_CUH__
#define __BLS12_381_PP_CUH__

#include "bls12_381_g1.cuh"
#include "bls12_381_g2.cuh"
#include "bls12_381_init.cuh"
#include "bls12_381_pairing.cuh"
#include "../public_params.cuh"

namespace libff {

class bls12_381_pp {
public:
    typedef bls12_381_Fr Fp_type;
    typedef bls12_381_G1 G1_type;
    typedef bls12_381_G2 G2_type;
    typedef bls12_381_G1_precomp G1_precomp_type;
    typedef bls12_381_G2_precomp G2_precomp_type;
    typedef bls12_381_Fq Fq_type;
    typedef bls12_381_Fq2 Fqe_type;
    typedef bls12_381_Fq12 Fqk_type;
    typedef bls12_381_GT GT_type;

    static const bool has_affine_pairing = false;

    __device__ static void init_public_params();
    __device__ static bls12_381_GT final_exponentiation(const bls12_381_Fq12 &elt);
    __device__ static bls12_381_G1_precomp precompute_G1(const bls12_381_G1 &P);
    __device__ static bls12_381_G2_precomp precompute_G2(const bls12_381_G2 &Q);
    __device__ static bls12_381_Fq12 miller_loop(const bls12_381_G1_precomp &prec_P,const bls12_381_G2_precomp &prec_Q);
                                      
    __device__ static bls12_381_Fq12 double_miller_loop(const bls12_381_G1_precomp &prec_P1,
                                                        const bls12_381_G2_precomp &prec_Q1,
                                                        const bls12_381_G1_precomp &prec_P2,
                                                        const bls12_381_G2_precomp &prec_Q2);
    __device__ static bls12_381_Fq12 pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);
    __device__ static bls12_381_Fq12 reduced_pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);
};

}

#include "bls12_381_pp.cu"

#endif
