#ifndef __ALT_BN128_G1_HOST_CUH__
#define __ALT_BN128_G1_HOST_CUH__

#include "alt_bn128_init_host.cuh"
#include "../curve_utils_host.cuh"

namespace libff {

struct alt_bn128_G1_params_host
{
    Fp_params_host<alt_bn128_r_limbs_host>* fr_params;
    Fp_params_host<alt_bn128_q_limbs_host>* fq_params;

    size_t* wnaf_window_table;
    size_t wnaf_window_table_size;
    
    size_t* fixed_base_exp_window_table_length;
    size_t fixed_base_exp_window_table_length_size;

    alt_bn128_Fq_host* G1_zero_X;
    alt_bn128_Fq_host* G1_zero_Y;
    alt_bn128_Fq_host* G1_zero_Z;
    alt_bn128_Fq_host* G1_one_X;
    alt_bn128_Fq_host* G1_one_Y;
    alt_bn128_Fq_host* G1_one_Z;
};


class alt_bn128_G1_host {
public:
    typedef alt_bn128_Fq_host base_field;
    typedef alt_bn128_Fr_host scalar_field;

    alt_bn128_G1_params_host* params;

    alt_bn128_Fq_host X, Y, Z;

     alt_bn128_G1_host() : params(nullptr) {}
     alt_bn128_G1_host(alt_bn128_G1_params_host* params);
     alt_bn128_G1_host(alt_bn128_G1_params_host* params, const alt_bn128_Fq_host& X, const alt_bn128_Fq_host& Y, const alt_bn128_Fq_host& Z) : params(params), X(X), Y(Y), Z(Z) {};
     alt_bn128_G1_host(const alt_bn128_G1_host& other) = default;

     alt_bn128_G1_host& operator=(const alt_bn128_G1_host& other) = default;

     void to_affine_coordinates();
     void to_special();
     bool is_special() const;

     bool is_zero() const;

     bool operator==(const alt_bn128_G1_host &other) const;
     bool operator!=(const alt_bn128_G1_host &other) const;

     alt_bn128_G1_host operator+(const alt_bn128_G1_host &other) const;

     alt_bn128_G1_host operator-() const;
     alt_bn128_G1_host operator-(const alt_bn128_G1_host &other) const;
    
     alt_bn128_G1_host operator*(const unsigned long lhs) const;

     alt_bn128_G1_host dbl() const;
     alt_bn128_G1_host add(const alt_bn128_G1_host &other) const;
     alt_bn128_G1_host mixed_add(const alt_bn128_G1_host &other) const;

     bool is_well_formed() const;

     alt_bn128_G1_host zero() const;
     alt_bn128_G1_host one() const;
     alt_bn128_G1_host random_element() const;

     size_t size_in_bits();
     bigint_host<alt_bn128_q_limbs_host> base_field_char();
     bigint_host<alt_bn128_r_limbs_host> order();

     void set_params(alt_bn128_G1_params_host* params);

    //  void batch_to_special(libstl::vector<alt_bn128_G1_host> &vec);
    //  void p_batch_to_special(libstl::vector<alt_bn128_G1_host> &vec, size_t gridSize, size_t blockSize);
};

template<mp_size_t m>
inline alt_bn128_G1_host operator*(const bigint_host<m>& lhs, const alt_bn128_G1_host& rhs)
{
    return scalar_mul_host<alt_bn128_G1_host, m>(rhs, lhs);
}

template<mp_size_t m>
inline alt_bn128_G1_host operator*(const Fp_model_host<m>& lhs, const alt_bn128_G1_host& rhs)
{
    return scalar_mul_host<alt_bn128_G1_host, m>(rhs, lhs.as_bigint());
}


}


#endif
