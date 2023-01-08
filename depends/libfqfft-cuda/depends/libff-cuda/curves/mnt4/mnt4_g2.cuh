#ifndef __MNT4_G2_CUH__
#define __MNT4_G2_CUH__

#include "mnt4_init.cuh"
#include "../curve_utils.cuh"

namespace libff {

struct mnt4_G2_params
{
    Fp_params<mnt4_r_limbs>* fr_params;
    Fp_params<mnt4_q_limbs>* fq_params;
    Fp2_params<mnt4_q_limbs>* fq2_params;

    size_t* wnaf_window_table;
    size_t wnaf_window_table_size;
    
    size_t* fixed_base_exp_window_table_length;
    size_t fixed_base_exp_window_table_length_size;

    mnt4_Fq2* G2_zero_X;
    mnt4_Fq2* G2_zero_Y;
    mnt4_Fq2* G2_zero_Z;
    mnt4_Fq2* G2_one_X;
    mnt4_Fq2* G2_one_Y;
    mnt4_Fq2* G2_one_Z;

    mnt4_Fq2* twist;
    mnt4_Fq2* coeff_a;
    mnt4_Fq2* coeff_b;
};

class mnt4_G2 {
public:
    typedef mnt4_Fr scalar_field;
    typedef mnt4_Fq base_field;
    typedef mnt4_Fq2 twist_field;

    mnt4_G2_params* params;

    mnt4_Fq2 X, Y, Z;
 
    __device__ mnt4_G2() : params(nullptr) {}
    __device__ mnt4_G2(mnt4_G2_params* params);
    __device__ mnt4_G2(mnt4_G2_params* params, const mnt4_Fq2& X, const mnt4_Fq2& Y, const mnt4_Fq2& Z): params(params), X(X), Y(Y), Z(Z) {};
    __device__ mnt4_G2(const mnt4_G2& other) = default;
    
    __device__ mnt4_G2& operator=(const mnt4_G2& other) = default;

    __device__ mnt4_Fq2 mul_by_a(const mnt4_Fq2 &elt) const;
    __device__ mnt4_Fq2 mul_by_b(const mnt4_Fq2 &elt) const;

    __device__ void to_affine_coordinates();
    __device__ void to_special();
    __device__ bool is_special() const;

    __device__ bool is_zero() const;

    __device__ bool operator==(const mnt4_G2 &other) const;
    __device__ bool operator!=(const mnt4_G2 &other) const;

    __device__ mnt4_G2 operator+(const mnt4_G2 &other) const;
    __device__ mnt4_G2 operator-() const;
    __device__ mnt4_G2 operator-(const mnt4_G2 &other) const;

    __device__ mnt4_G2 operator*(const unsigned long lhs) const;
    
    __device__ mnt4_G2 dbl() const;
    __device__ mnt4_G2 add(const mnt4_G2 &other) const;
    __device__ mnt4_G2 mixed_add(const mnt4_G2 &other) const;
    __device__ mnt4_G2 mul_by_q() const;

    __device__ mnt4_G2 zero() const;
    __host__ mnt4_G2* zero_host();
    __device__ mnt4_G2 one() const;
    __host__ mnt4_G2* one_host();
    __device__ mnt4_G2 random_element() const;

    __device__ bool is_well_formed() const;

    __device__ size_t size_in_bits();
    __device__ bigint<mnt4_q_limbs> base_field_char();
    __device__ bigint<mnt4_r_limbs> order();

    __device__ void set_params(mnt4_G2_params* params);

    __device__ void batch_to_special(libstl::vector<mnt4_G2> &vec);
    __device__ void p_batch_to_special(libstl::vector<mnt4_G2> &vec, size_t gridSize, size_t blockSize);
};

template<mp_size_t_ m>
__device__ inline mnt4_G2 operator*(const bigint<m>& lhs, const mnt4_G2& rhs)
{
    return scalar_mul<mnt4_G2, m>(rhs, lhs);
}

template<mp_size_t_ m>
__device__ inline mnt4_G2 operator*(const Fp_model<m>& lhs, const mnt4_G2& rhs)
{
    return scalar_mul<mnt4_G2, m>(rhs, lhs.as_bigint());
}

} 

#include "mnt4_g2.cu"

#endif
