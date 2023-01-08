#ifndef __ALT_BN128_G2_CUH__
#define __ALT_BN128_G2_CUH__

#include "alt_bn128_init.cuh"
#include "../curve_utils.cuh"
#include "../../depends/libstl-cuda/vector.cuh"

namespace libff {

struct alt_bn128_G2_params
{
    Fp_params<alt_bn128_r_limbs>* fr_params;
    Fp_params<alt_bn128_q_limbs>* fq_params;
    Fp2_params<alt_bn128_q_limbs>* fq2_params;

    size_t* wnaf_window_table;
    size_t wnaf_window_table_size;
    
    size_t* fixed_base_exp_window_table_length;
    size_t fixed_base_exp_window_table_length_size;

    alt_bn128_Fq2* G2_zero_X;
    alt_bn128_Fq2* G2_zero_Y;
    alt_bn128_Fq2* G2_zero_Z;
    alt_bn128_Fq2* G2_one_X;
    alt_bn128_Fq2* G2_one_Y;
    alt_bn128_Fq2* G2_one_Z;
};

class alt_bn128_G2 {
public:
    typedef alt_bn128_Fr scalar_field;
    typedef alt_bn128_Fq base_field;
    typedef alt_bn128_Fq2 twist_field;

    alt_bn128_G2_params* params;

    alt_bn128_Fq2 X, Y, Z;
 
    __device__ alt_bn128_G2() : params(nullptr) {}
    __device__ alt_bn128_G2(alt_bn128_G2_params* params);
    __device__ alt_bn128_G2(alt_bn128_G2_params* params, const alt_bn128_Fq2& X, const alt_bn128_Fq2& Y, const alt_bn128_Fq2& Z): params(params), X(X), Y(Y), Z(Z) {};
    __device__ alt_bn128_G2(const alt_bn128_G2& other) = default;
    
    __device__ alt_bn128_G2& operator=(const alt_bn128_G2& other) = default;

    __device__ alt_bn128_Fq2 mul_by_b(const alt_bn128_Fq2 &elt);

    __device__ void to_affine_coordinates();
    __device__ void to_special();
    __device__ bool is_special() const;

    __device__ bool is_zero() const;

    __device__ bool operator==(const alt_bn128_G2 &other) const;
    __device__ bool operator!=(const alt_bn128_G2 &other) const;

    __device__ alt_bn128_G2 operator+(const alt_bn128_G2 &other) const;
    __device__ alt_bn128_G2 operator-() const;
    __device__ alt_bn128_G2 operator-(const alt_bn128_G2 &other) const;

    __device__ alt_bn128_G2 operator*(const unsigned long lhs) const;
    
    __device__ alt_bn128_G2 dbl() const;
    __device__ alt_bn128_G2 add(const alt_bn128_G2 &other) const;
    __device__ alt_bn128_G2 mixed_add(const alt_bn128_G2 &other) const;
    __device__ alt_bn128_G2 mul_by_q() const;

    __device__ alt_bn128_G2 zero() const;
    __host__ alt_bn128_G2* zero_host();
    __device__ alt_bn128_G2 one() const;
    __host__ alt_bn128_G2* one_host();
    __device__ alt_bn128_G2 random_element() const;

    __device__ bool is_well_formed() const;

    __device__ size_t size_in_bits();
    __device__ bigint<alt_bn128_q_limbs> base_field_char();
    __device__ bigint<alt_bn128_r_limbs> order();

    __device__ void set_params(alt_bn128_G2_params* params);

    __device__ void batch_to_special(libstl::vector<alt_bn128_G2> &vec);
    __device__ void p_batch_to_special(libstl::vector<alt_bn128_G2> &vec, size_t gridSize, size_t blockSize);
};

template<mp_size_t_ m>
__device__ inline alt_bn128_G2 operator*(const bigint<m>& lhs, const alt_bn128_G2& rhs)
{
    return scalar_mul<alt_bn128_G2, m>(rhs, lhs);
}

template<mp_size_t_ m>
__device__ inline alt_bn128_G2 operator*(const Fp_model<m>& lhs, const alt_bn128_G2& rhs)
{
    return scalar_mul<alt_bn128_G2, m>(rhs, lhs.as_bigint());
}

} 

#include "alt_bn128_g2.cu"

#endif
