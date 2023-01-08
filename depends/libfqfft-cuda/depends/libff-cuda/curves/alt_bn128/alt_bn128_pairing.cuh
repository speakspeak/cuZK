#ifndef __ALT_BN128_PAIRING_CUH__
#define __ALT_BN128_PAIRING_CUH__

#include "alt_bn128_init.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"

namespace libff {

__device__ alt_bn128_GT alt_bn128_final_exponentiation(const alt_bn128_Fq12 &elt);

struct alt_bn128_ate_G1_precomp {
    alt_bn128_Fq PX;
    alt_bn128_Fq PY;

    __device__ alt_bn128_ate_G1_precomp();
    __device__ bool operator==(const alt_bn128_ate_G1_precomp &other) const;
};

struct alt_bn128_ate_ell_coeffs {
    alt_bn128_Fq2 ell_0;
    alt_bn128_Fq2 ell_VW;
    alt_bn128_Fq2 ell_VV;

    __device__ alt_bn128_ate_ell_coeffs();
    __device__ bool operator==(const alt_bn128_ate_ell_coeffs &other) const;
};

struct alt_bn128_ate_G2_precomp {
    alt_bn128_Fq2 QX;
    alt_bn128_Fq2 QY;
    alt_bn128_ate_ell_coeffs* coeffs;
    size_t coeffs_size;

    __device__ alt_bn128_ate_G2_precomp();
    __device__ ~alt_bn128_ate_G2_precomp();
    __device__ bool operator==(const alt_bn128_ate_G2_precomp &other) const;
};

__device__ alt_bn128_ate_G1_precomp alt_bn128_ate_precompute_G1(const alt_bn128_G1& P);

__device__ alt_bn128_ate_G2_precomp alt_bn128_ate_precompute_G2(const alt_bn128_G2& Q);

__device__ alt_bn128_Fq12 alt_bn128_ate_miller_loop(const alt_bn128_ate_G1_precomp &prec_P, const alt_bn128_ate_G2_precomp &prec_Q);

__device__ alt_bn128_Fq12 alt_bn128_ate_double_miller_loop(const alt_bn128_ate_G1_precomp &prec_P1,
                                                            const alt_bn128_ate_G2_precomp &prec_Q1,
                                                            const alt_bn128_ate_G1_precomp &prec_P2,
                                                            const alt_bn128_ate_G2_precomp &prec_Q2);

__device__ alt_bn128_Fq12 alt_bn128_ate_pairing(const alt_bn128_G1& P, const alt_bn128_G2 &Q);

__device__ alt_bn128_GT alt_bn128_ate_reduced_pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);

/* choice of pairing */

typedef alt_bn128_ate_G1_precomp alt_bn128_G1_precomp;
typedef alt_bn128_ate_G2_precomp alt_bn128_G2_precomp;

__device__ alt_bn128_G1_precomp alt_bn128_precompute_G1(const alt_bn128_G1& P);

__device__ alt_bn128_G2_precomp alt_bn128_precompute_G2(const alt_bn128_G2& Q);

__device__ alt_bn128_Fq12 alt_bn128_miller_loop(const alt_bn128_G1_precomp &prec_P, const alt_bn128_G2_precomp &prec_Q);

__device__ alt_bn128_Fq12 alt_bn128_double_miller_loop(const alt_bn128_G1_precomp &prec_P1,
                                                        const alt_bn128_G2_precomp &prec_Q1,
                                                        const alt_bn128_G1_precomp &prec_P2,
                                                        const alt_bn128_G2_precomp &prec_Q2);

__device__ alt_bn128_Fq12 alt_bn128_pairing(const alt_bn128_G1& P, const alt_bn128_G2 &Q);

__device__ alt_bn128_GT alt_bn128_reduced_pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);

__device__ alt_bn128_GT alt_bn128_affine_reduced_pairing(const alt_bn128_G1 &P, const alt_bn128_G2 &Q);

}

#include "alt_bn128_pairing.cu"

#endif 
