#ifndef __MNT4_PP_CU__
#define __MNT4_PP_CU__

namespace libff {

__device__ void mnt4_pp::init_public_params()
{
    init_mnt4_params();
}

__device__ mnt4_GT mnt4_pp::final_exponentiation(const mnt4_Fq4 &elt)
{
    return mnt4_final_exponentiation(elt);
}

__device__ mnt4_G1_precomp mnt4_pp::precompute_G1(const mnt4_G1 &P)
{
    return mnt4_precompute_G1(P);
}

__device__ mnt4_G2_precomp mnt4_pp::precompute_G2(const mnt4_G2 &Q)
{
    return mnt4_precompute_G2(Q);
}

__device__ mnt4_Fq4 mnt4_pp::miller_loop(const mnt4_G1_precomp &prec_P, const mnt4_G2_precomp &prec_Q)
{
    return mnt4_miller_loop(prec_P, prec_Q);
}

__device__ mnt4_affine_ate_G1_precomputation mnt4_pp::affine_ate_precompute_G1(const mnt4_G1 &P)
{
    return mnt4_affine_ate_precompute_G1(P);
}

__device__ mnt4_affine_ate_G2_precomputation mnt4_pp::affine_ate_precompute_G2(const mnt4_G2 &Q)
{
    return mnt4_affine_ate_precompute_G2(Q);
}

__device__ mnt4_Fq4 mnt4_pp::affine_ate_miller_loop(const mnt4_affine_ate_G1_precomputation &prec_P,
                                                    const mnt4_affine_ate_G2_precomputation &prec_Q)
{
    return mnt4_affine_ate_miller_loop(prec_P, prec_Q);
}

__device__ mnt4_Fq4 mnt4_pp::affine_ate_e_over_e_miller_loop(const mnt4_affine_ate_G1_precomputation &prec_P1,
                                                                const mnt4_affine_ate_G2_precomputation &prec_Q1,
                                                                const mnt4_affine_ate_G1_precomputation &prec_P2,
                                                                const mnt4_affine_ate_G2_precomputation &prec_Q2)
{
    return mnt4_affine_ate_miller_loop(prec_P1, prec_Q1) * mnt4_affine_ate_miller_loop(prec_P2, prec_Q2).unitary_inverse();
}

__device__  mnt4_Fq4 mnt4_pp::affine_ate_e_times_e_over_e_miller_loop(const mnt4_affine_ate_G1_precomputation &prec_P1,
                                                                        const mnt4_affine_ate_G2_precomputation &prec_Q1,
                                                                        const mnt4_affine_ate_G1_precomputation &prec_P2,
                                                                        const mnt4_affine_ate_G2_precomputation &prec_Q2,
                                                                        const mnt4_affine_ate_G1_precomputation &prec_P3,
                                                                        const mnt4_affine_ate_G2_precomputation &prec_Q3)
{
    return ((mnt4_affine_ate_miller_loop(prec_P1, prec_Q1) * mnt4_affine_ate_miller_loop(prec_P2, prec_Q2)) *
            mnt4_affine_ate_miller_loop(prec_P3, prec_Q3).unitary_inverse());
}

__device__  mnt4_Fq4 mnt4_pp::double_miller_loop(const mnt4_G1_precomp &prec_P1,
                                                    const mnt4_G2_precomp &prec_Q1,
                                                    const mnt4_G1_precomp &prec_P2,
                                                    const mnt4_G2_precomp &prec_Q2)
{
    return mnt4_double_miller_loop(prec_P1, prec_Q1, prec_P2, prec_Q2);
}

__device__  mnt4_Fq4 mnt4_pp::pairing(const mnt4_G1 &P, const mnt4_G2 &Q)
{
    return mnt4_pairing(P, Q);
}

__device__  mnt4_Fq4 mnt4_pp::reduced_pairing(const mnt4_G1 &P, const mnt4_G2 &Q)
{
    return mnt4_reduced_pairing(P, Q);
}

__device__  mnt4_Fq4 mnt4_pp::affine_reduced_pairing(const mnt4_G1 &P, const mnt4_G2 &Q)
{
    return mnt4_affine_reduced_pairing(P, Q);
}

}

#endif