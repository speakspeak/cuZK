#ifndef __MNT4_PAIRING_CUH__
#define __MNT4_PAIRING_CUH__

#include "mnt4_init.cuh"
#include "mnt4_g1.cuh"
#include "mnt4_g2.cuh"

namespace libff {

__device__ mnt4_GT mnt4_final_exponentiation(const mnt4_Fq4 &elt);

/* affine ate miller loop */

struct mnt4_affine_ate_G1_precomputation {
    mnt4_Fq PX;
    mnt4_Fq PY;
    mnt4_Fq2 PY_twist_squared;
};

struct mnt4_affine_ate_coeffs {
    mnt4_Fq2 old_RX;
    mnt4_Fq2 old_RY;
    mnt4_Fq2 gamma;
    mnt4_Fq2 gamma_twist;
    mnt4_Fq2 gamma_X;
};

struct mnt4_affine_ate_G2_precomputation {
    mnt4_Fq2 QX;
    mnt4_Fq2 QY;
    mnt4_affine_ate_coeffs* coeffs;
    size_t coeffs_size;

    __device__ mnt4_affine_ate_G2_precomputation() : coeffs_size(0) {}
};

__device__ mnt4_affine_ate_G1_precomputation mnt4_affine_ate_precompute_G1(const mnt4_G1& P);
__device__ mnt4_affine_ate_G2_precomputation mnt4_affine_ate_precompute_G2(const mnt4_G2& Q);

__device__ mnt4_Fq4 mnt4_affine_ate_miller_loop(const mnt4_affine_ate_G1_precomputation &prec_P,
                                                const mnt4_affine_ate_G2_precomputation &prec_Q);

/* ate pairing */

struct mnt4_ate_G1_precomp {
    mnt4_Fq PX;
    mnt4_Fq PY;
    mnt4_Fq2 PX_twist;
    mnt4_Fq2 PY_twist;

    __device__ bool operator==(const mnt4_ate_G1_precomp &other) const;
};


struct mnt4_ate_dbl_coeffs {
    mnt4_Fq2 c_H;
    mnt4_Fq2 c_4C;
    mnt4_Fq2 c_J;
    mnt4_Fq2 c_L;

    __device__ bool operator==(const mnt4_ate_dbl_coeffs &other) const;
};

struct mnt4_ate_add_coeffs {
    mnt4_Fq2 c_L1;
    mnt4_Fq2 c_RZ;

    __device__ bool operator==(const mnt4_ate_add_coeffs &other) const;
};

struct mnt4_ate_G2_precomp {
    mnt4_Fq2 QX;
    mnt4_Fq2 QY;
    mnt4_Fq2 QY2;
    mnt4_Fq2 QX_over_twist;
    mnt4_Fq2 QY_over_twist;

    mnt4_ate_dbl_coeffs* dbl_coeffs;
    mnt4_ate_add_coeffs* add_coeffs;

    size_t dbl_coeffs_size;
    size_t add_coeffs_size;

    __device__ mnt4_ate_G2_precomp() : dbl_coeffs_size(0), add_coeffs_size(0) {}
    __device__ bool operator==(const mnt4_ate_G2_precomp &other) const;
};

__device__ mnt4_ate_G1_precomp mnt4_ate_precompute_G1(const mnt4_G1& P);
__device__ mnt4_ate_G2_precomp mnt4_ate_precompute_G2(const mnt4_G2& Q);

__device__ mnt4_Fq4 mnt4_ate_miller_loop(const mnt4_ate_G1_precomp &prec_P,
                                         const mnt4_ate_G2_precomp &prec_Q);

__device__ mnt4_Fq4 mnt4_ate_double_miller_loop(const mnt4_ate_G1_precomp &prec_P1,
                                                const mnt4_ate_G2_precomp &prec_Q1,
                                                const mnt4_ate_G1_precomp &prec_P2,
                                                const mnt4_ate_G2_precomp &prec_Q2);

__device__ mnt4_Fq4 mnt4_ate_pairing(const mnt4_G1& P, const mnt4_G2 &Q);

__device__ mnt4_GT mnt4_ate_reduced_pairing(const mnt4_G1 &P, const mnt4_G2 &Q);



// choice of pairing
typedef mnt4_ate_G1_precomp mnt4_G1_precomp;
typedef mnt4_ate_G2_precomp mnt4_G2_precomp;


__device__ mnt4_G1_precomp mnt4_precompute_G1(const mnt4_G1& P);

__device__ mnt4_G2_precomp mnt4_precompute_G2(const mnt4_G2& Q);

__device__ mnt4_Fq4 mnt4_miller_loop(const mnt4_G1_precomp &prec_P,
                                     const mnt4_G2_precomp &prec_Q);

__device__ mnt4_Fq4 mnt4_double_miller_loop(const mnt4_G1_precomp &prec_P1,
                                            const mnt4_G2_precomp &prec_Q1,
                                            const mnt4_G1_precomp &prec_P2,
                                            const mnt4_G2_precomp &prec_Q2);

__device__ mnt4_Fq4 mnt4_pairing(const mnt4_G1& P, const mnt4_G2 &Q);

__device__ mnt4_GT mnt4_reduced_pairing(const mnt4_G1 &P, const mnt4_G2 &Q);

__device__ mnt4_GT mnt4_affine_reduced_pairing(const mnt4_G1 &P, const mnt4_G2 &Q);

}

#include "mnt4_pairing.cu"

#endif