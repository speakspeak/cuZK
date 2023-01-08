#ifndef __MNT4753_PP_CUH__
#define __MNT4753_PP_CUH__

#include "mnt4753_init.cuh"
#include "mnt4753_pairing.cuh"
#include "mnt4753_g1.cuh"
#include "mnt4753_g2.cuh"
#include "../public_params.cuh"

namespace libff {

class mnt4753_pp {
public:
    typedef mnt4753_Fr Fp_type;
    typedef mnt4753_G1 G1_type;
    typedef mnt4753_G2 G2_type;
    typedef mnt4753_G1_precomp G1_precomp_type;
    typedef mnt4753_G2_precomp G2_precomp_type;
    typedef mnt4753_affine_ate_G1_precomputation affine_ate_G1_precomp_type;
    typedef mnt4753_affine_ate_G2_precomputation affine_ate_G2_precomp_type;
    typedef mnt4753_Fq Fq_type;
    typedef mnt4753_Fq2 Fqe_type;
    typedef mnt4753_Fq4 Fqk_type;
    typedef mnt4753_GT GT_type;

    static const bool has_affine_pairing = true;

    __device__ static void init_public_params();
    __device__ static mnt4753_GT final_exponentiation(const mnt4753_Fq4 &elt);

    __device__ static mnt4753_G1_precomp precompute_G1(const mnt4753_G1 &P);
    __device__ static mnt4753_G2_precomp precompute_G2(const mnt4753_G2 &Q);

    __device__ static mnt4753_Fq4 miller_loop(const mnt4753_G1_precomp &prec_P, const mnt4753_G2_precomp &prec_Q);

    __device__ static mnt4753_affine_ate_G1_precomputation affine_ate_precompute_G1(const mnt4753_G1 &P);
    __device__ static mnt4753_affine_ate_G2_precomputation affine_ate_precompute_G2(const mnt4753_G2 &Q);

    __device__ static mnt4753_Fq4 affine_ate_miller_loop(const mnt4753_affine_ate_G1_precomputation &prec_P,
                                                        const mnt4753_affine_ate_G2_precomputation &prec_Q);

    __device__ static mnt4753_Fq4 affine_ate_e_over_e_miller_loop(const mnt4753_affine_ate_G1_precomputation &prec_P1,
                                                                const mnt4753_affine_ate_G2_precomputation &prec_Q1,
                                                                const mnt4753_affine_ate_G1_precomputation &prec_P2,
                                                                const mnt4753_affine_ate_G2_precomputation &prec_Q2);

    __device__ static mnt4753_Fq4 affine_ate_e_times_e_over_e_miller_loop(const mnt4753_affine_ate_G1_precomputation &prec_P1,
                                                                        const mnt4753_affine_ate_G2_precomputation &prec_Q1,
                                                                        const mnt4753_affine_ate_G1_precomputation &prec_P2,
                                                                        const mnt4753_affine_ate_G2_precomputation &prec_Q2,
                                                                        const mnt4753_affine_ate_G1_precomputation &prec_P3,
                                                                        const mnt4753_affine_ate_G2_precomputation &prec_Q3);

    __device__ static mnt4753_Fq4 double_miller_loop(const mnt4753_G1_precomp &prec_P1,
                                                    const mnt4753_G2_precomp &prec_Q1,
                                                    const mnt4753_G1_precomp &prec_P2,
                                                    const mnt4753_G2_precomp &prec_Q2);

    __device__ static mnt4753_Fq4 pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q);
    __device__ static mnt4753_Fq4 reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q);
    __device__ static mnt4753_Fq4 affine_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q);

};

}

#include "mnt4753_pp.cu"

#endif