#ifndef __BLS12_381_PP_CU__
#define __BLS12_381_PP_CU__

namespace libff {

__device__ void bls12_381_pp::init_public_params()
{
    init_bls12_381_params();
}

__device__ bls12_381_GT bls12_381_pp::final_exponentiation(const bls12_381_Fq12 &elt)
{
    return bls12_381_final_exponentiation(elt);
}

__device__ bls12_381_G1_precomp bls12_381_pp::precompute_G1(const bls12_381_G1 &P)
{
    return bls12_381_precompute_G1(P);
}

__device__ bls12_381_G2_precomp bls12_381_pp::precompute_G2(const bls12_381_G2 &Q)
{
    return bls12_381_precompute_G2(Q);
}

__device__ bls12_381_Fq12 bls12_381_pp::miller_loop(const bls12_381_G1_precomp &prec_P, const bls12_381_G2_precomp &prec_Q)
{
    return bls12_381_miller_loop(prec_P, prec_Q);
}

__device__ bls12_381_Fq12 bls12_381_pp::double_miller_loop(const bls12_381_G1_precomp &prec_P1,
                                                            const bls12_381_G2_precomp &prec_Q1,
                                                            const bls12_381_G1_precomp &prec_P2,
                                                            const bls12_381_G2_precomp &prec_Q2)
{
    return bls12_381_double_miller_loop(prec_P1, prec_Q1, prec_P2, prec_Q2);
}

__device__ bls12_381_Fq12 bls12_381_pp::pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q)
{
    return bls12_381_pairing(P, Q);
}

__device__ bls12_381_Fq12 bls12_381_pp::reduced_pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q)
{
    return bls12_381_reduced_pairing(P, Q);
}

}

#endif
