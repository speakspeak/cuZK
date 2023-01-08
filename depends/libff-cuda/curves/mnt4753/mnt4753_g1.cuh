#ifndef __MNT4753_G1_CUH__
#define __MNT4753_G1_CUH__

#include "mnt4753_init.cuh"
#include "../curve_utils.cuh"
#include "../../depends/libstl-cuda/vector.cuh"
#include "../../fields/field_utils.cuh"

namespace libff {

struct mnt4753_G1_params
{
    Fp_params<mnt4753_r_limbs>* fr_params;
    Fp_params<mnt4753_q_limbs>* fq_params;

    size_t* wnaf_window_table;
    size_t wnaf_window_table_size;
    
    size_t* fixed_base_exp_window_table_length;
    size_t fixed_base_exp_window_table_length_size;

    mnt4753_Fq* G1_zero_X;
    mnt4753_Fq* G1_zero_Y;
    mnt4753_Fq* G1_zero_Z;
    mnt4753_Fq* G1_one_X;
    mnt4753_Fq* G1_one_Y;
    mnt4753_Fq* G1_one_Z;

    mnt4753_Fq* coeff_a;
    mnt4753_Fq* coeff_b;
};

class mnt4753_G1 {
public:
    typedef mnt4753_Fq base_field;
    typedef mnt4753_Fr scalar_field;

    mnt4753_G1_params* params;

    mnt4753_Fq X, Y, Z;

    __device__ mnt4753_G1() : params(nullptr) {}
    __device__ mnt4753_G1(mnt4753_G1_params* params);
    __device__ mnt4753_G1(mnt4753_G1_params* params, const mnt4753_Fq& X, const mnt4753_Fq& Y, const mnt4753_Fq& Z) : params(params), X(X), Y(Y), Z(Z) {};
    __device__ mnt4753_G1(const mnt4753_G1& other) = default;

    __device__ mnt4753_G1& operator=(const mnt4753_G1& other) = default;

    __device__ void to_affine_coordinates();
    __device__ void to_special();
    __device__ bool is_special() const;

    __device__ bool is_zero() const;

    __device__ bool operator==(const mnt4753_G1 &other) const;
    __device__ bool operator!=(const mnt4753_G1 &other) const;

    __device__ mnt4753_G1 operator+(const mnt4753_G1 &other) const;

    __device__ mnt4753_G1 operator-() const;
    __device__ mnt4753_G1 operator-(const mnt4753_G1 &other) const;
    
    __device__ mnt4753_G1 operator*(const unsigned long lhs) const;

    __device__ mnt4753_G1 dbl() const;
    __device__ mnt4753_G1 add(const mnt4753_G1 &other) const;
    __device__ mnt4753_G1 mixed_add(const mnt4753_G1 &other) const;

    __device__ bool is_well_formed() const;

    __device__ mnt4753_G1 zero() const;
    __host__ mnt4753_G1* zero_host();
    __device__ mnt4753_G1 one() const;
    __host__ mnt4753_G1* one_host();
    __device__ mnt4753_G1 random_element() const;

    __device__ size_t size_in_bits();
    __device__ bigint<mnt4753_q_limbs> base_field_char();
    __device__ bigint<mnt4753_r_limbs> order();

    __device__ void set_params(mnt4753_G1_params* params);

    __device__ void batch_to_special(libstl::vector<mnt4753_G1> &vec);
    __device__ void p_batch_to_special(libstl::vector<mnt4753_G1> &vec, size_t gridSize, size_t blockSize);
};

template<mp_size_t_ m>
__device__ inline mnt4753_G1 operator*(const bigint<m>& lhs, const mnt4753_G1& rhs)
{
    return scalar_mul<mnt4753_G1, m>(rhs, lhs);
}

template<mp_size_t_ m>
__device__ inline mnt4753_G1 operator*(const Fp_model<m>& lhs, const mnt4753_G1& rhs)
{
    return scalar_mul<mnt4753_G1, m>(rhs, lhs.as_bigint());
}

}

#include "mnt4753_g1.cu"

#endif
