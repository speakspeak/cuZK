#ifndef __MNT4753_PAIRING_CUH__
#define __MNT4753_PAIRING_CUH__

#include "mnt4753_init.cuh"
#include "mnt4753_g1.cuh"
#include "mnt4753_g2.cuh"

namespace libff {

__device__ mnt4753_GT mnt4753_final_exponentiation(const mnt4753_Fq4 &elt);

/* affine ate miller loop */

struct mnt4753_affine_ate_G1_precomputation {
    mnt4753_Fq PX;
    mnt4753_Fq PY;
    mnt4753_Fq2 PY_twist_squared;
};

struct mnt4753_affine_ate_coeffs {
    mnt4753_Fq2 old_RX;
    mnt4753_Fq2 old_RY;
    mnt4753_Fq2 gamma;
    mnt4753_Fq2 gamma_twist;
    mnt4753_Fq2 gamma_X;
};

struct mnt4753_affine_ate_G2_precomputation {
    mnt4753_Fq2 QX;
    mnt4753_Fq2 QY;
    mnt4753_affine_ate_coeffs* coeffs;
    size_t coeffs_size;

    __device__ mnt4753_affine_ate_G2_precomputation() : coeffs_size(0) {}
};

__device__ mnt4753_affine_ate_G1_precomputation mnt4753_affine_ate_precompute_G1(const mnt4753_G1& P);
__device__ mnt4753_affine_ate_G2_precomputation mnt4753_affine_ate_precompute_G2(const mnt4753_G2& Q);

__device__ mnt4753_Fq4 mnt4753_affine_ate_miller_loop(const mnt4753_affine_ate_G1_precomputation &prec_P,
                                                const mnt4753_affine_ate_G2_precomputation &prec_Q);

/* ate pairing */

struct mnt4753_ate_G1_precomp {
    mnt4753_Fq PX;
    mnt4753_Fq PY;
    mnt4753_Fq2 PX_twist;
    mnt4753_Fq2 PY_twist;

    __device__ bool operator==(const mnt4753_ate_G1_precomp &other) const;
};


struct mnt4753_ate_dbl_coeffs {
    mnt4753_Fq2 c_H;
    mnt4753_Fq2 c_4C;
    mnt4753_Fq2 c_J;
    mnt4753_Fq2 c_L;

    __device__ bool operator==(const mnt4753_ate_dbl_coeffs &other) const;
};

struct mnt4753_ate_add_coeffs {
    mnt4753_Fq2 c_L1;
    mnt4753_Fq2 c_RZ;

    __device__ bool operator==(const mnt4753_ate_add_coeffs &other) const;
};

struct mnt4753_ate_G2_precomp {
    mnt4753_Fq2 QX;
    mnt4753_Fq2 QY;
    mnt4753_Fq2 QY2;
    mnt4753_Fq2 QX_over_twist;
    mnt4753_Fq2 QY_over_twist;

    mnt4753_ate_dbl_coeffs* dbl_coeffs;
    mnt4753_ate_add_coeffs* add_coeffs;

    size_t dbl_coeffs_size;
    size_t add_coeffs_size;

    __device__ mnt4753_ate_G2_precomp() : dbl_coeffs_size(0), add_coeffs_size(0) {}
    __device__ bool operator==(const mnt4753_ate_G2_precomp &other) const;
};

__device__ mnt4753_ate_G1_precomp mnt4753_ate_precompute_G1(const mnt4753_G1& P);
__device__ mnt4753_ate_G2_precomp mnt4753_ate_precompute_G2(const mnt4753_G2& Q);

__device__ mnt4753_Fq4 mnt4753_ate_miller_loop(const mnt4753_ate_G1_precomp &prec_P,
                                         const mnt4753_ate_G2_precomp &prec_Q);

__device__ mnt4753_Fq4 mnt4753_ate_double_miller_loop(const mnt4753_ate_G1_precomp &prec_P1,
                                                const mnt4753_ate_G2_precomp &prec_Q1,
                                                const mnt4753_ate_G1_precomp &prec_P2,
                                                const mnt4753_ate_G2_precomp &prec_Q2);

__device__ mnt4753_Fq4 mnt4753_ate_pairing(const mnt4753_G1& P, const mnt4753_G2 &Q);

__device__ mnt4753_GT mnt4753_ate_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q);



// choice of pairing
typedef mnt4753_ate_G1_precomp mnt4753_G1_precomp;
typedef mnt4753_ate_G2_precomp mnt4753_G2_precomp;


__device__ mnt4753_G1_precomp mnt4753_precompute_G1(const mnt4753_G1& P);

__device__ mnt4753_G2_precomp mnt4753_precompute_G2(const mnt4753_G2& Q);

__device__ mnt4753_Fq4 mnt4753_miller_loop(const mnt4753_G1_precomp &prec_P,
                                     const mnt4753_G2_precomp &prec_Q);

__device__ mnt4753_Fq4 mnt4753_double_miller_loop(const mnt4753_G1_precomp &prec_P1,
                                            const mnt4753_G2_precomp &prec_Q1,
                                            const mnt4753_G1_precomp &prec_P2,
                                            const mnt4753_G2_precomp &prec_Q2);

__device__ mnt4753_Fq4 mnt4753_pairing(const mnt4753_G1& P, const mnt4753_G2 &Q);

__device__ mnt4753_GT mnt4753_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q);

__device__ mnt4753_GT mnt4753_affine_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q);

}

#include "mnt4753_pairing.cu"

#endif