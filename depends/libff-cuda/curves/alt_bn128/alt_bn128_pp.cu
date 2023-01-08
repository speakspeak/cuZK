#ifndef __ALT_BN128_PP_CU__
#define __ALT_BN128_PP_CU__

namespace libff {

__device__ void alt_bn128_pp::init_public_params()
{
    init_alt_bn128_params();
}

__device__ alt_bn128_GT alt_bn128_pp::final_exponentiation(const alt_bn128_Fq12 &elt)
{
    return alt_bn128_final_exponentiation(elt);
}

__device__ alt_bn128_G1_precomp alt_bn128_pp::precompute_G1(const alt_bn128_G1 &P)
{
    return alt_bn128_precompute_G1(P);
}

__device__ alt_bn128_G2_precomp alt_bn128_pp::precompute_G2(const alt_bn128_G2 &Q)
{
    return alt_bn128_precompute_G2(Q);
}

__device__ alt_bn128_Fq12 alt_bn128_pp::miller_loop(const alt_bn128_G1_precomp &prec_P, const alt_bn128_G2_precomp &prec_Q)
{
    return alt_bn128_miller_loop(prec_P, prec_Q);
}

__device__ alt_bn128_Fq12 alt_bn128_pp::double_miller_loop(const alt_bn128_G1_precomp &prec_P1,
                                                            const alt_bn128_G2_precomp &prec_Q1,
                                                            const alt_bn128_G1_precomp &prec_P2,
                                                            const alt_bn128_G2_precomp &prec_Q2)
{
    return alt_bn128_double_miller_loop(prec_P1, prec_Q1, prec_P2, prec_Q2);
}

__device__ alt_bn128_Fq12 alt_bn128_pp::pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q)
{
    return alt_bn128_pairing(P, Q);
}

__device__ alt_bn128_Fq12 alt_bn128_pp::reduced_pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q)
{
    return alt_bn128_reduced_pairing(P, Q);
}

}

#endif
