#ifndef __BLS12_381_PAIRING_CUH__
#define __BLS12_381_PAIRING_CUH__

#include "bls12_381_init.cuh"
#include "bls12_381_g1.cuh"
#include "bls12_381_g2.cuh"

namespace libff {

__device__ bls12_381_GT bls12_381_final_exponentiation(const bls12_381_Fq12 &elt);

struct bls12_381_ate_G1_precomp {
    bls12_381_Fq PX;
    bls12_381_Fq PY;

    __device__ bls12_381_ate_G1_precomp();
    __device__ bool operator==(const bls12_381_ate_G1_precomp &other) const;
};

struct bls12_381_ate_ell_coeffs {
    bls12_381_Fq2 ell_0;
    bls12_381_Fq2 ell_VW;
    bls12_381_Fq2 ell_VV;

    __device__ bls12_381_ate_ell_coeffs();
    __device__ bool operator==(const bls12_381_ate_ell_coeffs &other) const;
};

struct bls12_381_ate_G2_precomp {
    bls12_381_Fq2 QX;
    bls12_381_Fq2 QY;
    bls12_381_ate_ell_coeffs* coeffs;
    size_t coeffs_size;

    __device__ bls12_381_ate_G2_precomp();
    __device__ ~bls12_381_ate_G2_precomp();
    __device__ bool operator==(const bls12_381_ate_G2_precomp &other) const;
};

__device__ bls12_381_ate_G1_precomp bls12_381_ate_precompute_G1(const bls12_381_G1& P);

__device__ bls12_381_ate_G2_precomp bls12_381_ate_precompute_G2(const bls12_381_G2& Q);

__device__ bls12_381_Fq12 bls12_381_ate_miller_loop(const bls12_381_ate_G1_precomp &prec_P, const bls12_381_ate_G2_precomp &prec_Q);

__device__ bls12_381_Fq12 bls12_381_ate_double_miller_loop(const bls12_381_ate_G1_precomp &prec_P1,
                                                            const bls12_381_ate_G2_precomp &prec_Q1,
                                                            const bls12_381_ate_G1_precomp &prec_P2,
                                                            const bls12_381_ate_G2_precomp &prec_Q2);

__device__ bls12_381_Fq12 bls12_381_ate_pairing(const bls12_381_G1& P, const bls12_381_G2 &Q);

__device__ bls12_381_GT bls12_381_ate_reduced_pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);

/* choice of pairing */

typedef bls12_381_ate_G1_precomp bls12_381_G1_precomp;
typedef bls12_381_ate_G2_precomp bls12_381_G2_precomp;

__device__ bls12_381_G1_precomp bls12_381_precompute_G1(const bls12_381_G1& P);

__device__ bls12_381_G2_precomp bls12_381_precompute_G2(const bls12_381_G2& Q);

__device__ bls12_381_Fq12 bls12_381_miller_loop(const bls12_381_G1_precomp &prec_P, const bls12_381_G2_precomp &prec_Q);

__device__ bls12_381_Fq12 bls12_381_double_miller_loop(const bls12_381_G1_precomp &prec_P1,
                                                        const bls12_381_G2_precomp &prec_Q1,
                                                        const bls12_381_G1_precomp &prec_P2,
                                                        const bls12_381_G2_precomp &prec_Q2);

__device__ bls12_381_Fq12 bls12_381_pairing(const bls12_381_G1& P, const bls12_381_G2 &Q);

__device__ bls12_381_GT bls12_381_reduced_pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);

__device__ bls12_381_GT bls12_381_affine_reduced_pairing(const bls12_381_G1 &P, const bls12_381_G2 &Q);

}

#include "bls12_381_pairing.cu"

#endif 
