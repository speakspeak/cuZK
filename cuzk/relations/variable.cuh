#ifndef __VARIABLE_CUH__
#define __VARIABLE_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace cuzk {

/**
 * Mnemonic typedefs.
 */
typedef size_t var_index_t;
typedef long integer_coeff_t;

/**
 * Forward declaration.
 */
template<typename FieldT>
class linear_term;

/**
 * Forward declaration.
 */
template<typename FieldT>
class linear_combination;


template<typename FieldT>
class variable {
public:

    var_index_t index;

    __device__ variable(const var_index_t index = 0) : index(index) {};
    __device__ bool operator==(const variable<FieldT> &other) const;
};


template<typename FieldT>
class linear_term {
public:

    var_index_t index;
    FieldT coeff;

    __device__ linear_term(const FieldT& instance);
    __device__ linear_term(const variable<FieldT> &var, const FieldT& instance);
    __device__ linear_term(const variable<FieldT> &var, const integer_coeff_t int_coeff, const FieldT& instance);
    __device__ linear_term(const variable<FieldT> &var, const FieldT &field_coeff, const FieldT& instance);
    
    __device__ linear_term& operator=(const linear_term& other);

    __device__ linear_term<FieldT> operator*(const integer_coeff_t int_coeff) const;
    __device__ linear_term<FieldT> operator*(const FieldT &field_coeff) const;

    __device__ linear_combination<FieldT> operator+(const linear_combination<FieldT> &other) const;
    __device__ linear_combination<FieldT> operator-(const linear_combination<FieldT> &other) const;

    __device__ linear_term<FieldT> operator-() const;

    __device__ bool operator==(const linear_term<FieldT> &other) const;
};

template<typename FieldT>
__device__ linear_term<FieldT> operator*(const integer_coeff_t int_coeff, const linear_term<FieldT> &lt);

template<typename FieldT>
__device__ linear_term<FieldT> operator*(const FieldT &field_coeff, const linear_term<FieldT> &lt);

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const integer_coeff_t int_coeff, const linear_term<FieldT> &lt);

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const FieldT &field_coeff, const linear_term<FieldT> &lt);

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const integer_coeff_t int_coeff, const linear_term<FieldT> &lt);

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const FieldT &field_coeff, const linear_term<FieldT> &lt);


template<typename FieldT>
class linear_combination {
public:

    libstl::vector<linear_term<FieldT>> vterms;
    // FieldT instance;

    __device__ linear_combination(){};
    // __device__ linear_combination(const FieldT& instance);
    __device__ linear_combination(const integer_coeff_t int_coeff, const FieldT& instance);
    __device__ linear_combination(const FieldT &field_coeff, const FieldT& instance);
    __device__ linear_combination(const variable<FieldT> &var, const FieldT& instance);
    __device__ linear_combination(const linear_term<FieldT> &lt);
    __device__ linear_combination(const linear_combination<FieldT>& other);


    __device__ linear_combination& operator=(const linear_combination& other);

    __device__ void reserve_term(const size_t size, const FieldT& instance);

    __device__ void add_term(const variable<FieldT> &var, const FieldT& instance);
    __device__ void add_term(const variable<FieldT> &var, const integer_coeff_t int_coeff, const FieldT& instance);
    __device__ void add_term(const variable<FieldT> &var, const FieldT &field_coeff, const FieldT& instance);
    __device__ void add_term(const linear_term<FieldT> &lt);

    __device__ void set_term(const size_t idx, const variable<FieldT> &var, const FieldT& instance);
    __device__ void set_term(const size_t idx, const variable<FieldT> &var, const integer_coeff_t int_coeff, const FieldT& instance);
    __device__ void set_term(const size_t idx, const variable<FieldT> &var, const FieldT &field_coeff, const FieldT& instance);
    __device__ void set_term(const size_t idx, const linear_term<FieldT> &lt);

    __device__ FieldT evaluate(const libstl::vector<FieldT>& assignment, const FieldT& instance) const;

    __device__ linear_combination<FieldT> operator*(const integer_coeff_t int_coeff) const;
    __device__ linear_combination<FieldT> operator*(const FieldT &field_coeff) const;

    __device__ linear_combination<FieldT> operator+(const linear_combination<FieldT> &other) const;

    __device__ linear_combination<FieldT> operator-(const linear_combination<FieldT> &other) const;
    __device__ linear_combination<FieldT> operator-() const;

    __device__ bool operator==(const linear_combination<FieldT> &other) const;

    __device__ bool is_valid(const size_t num_variables) const;
};

template<typename FieldT>
__device__ linear_combination<FieldT> operator*(const integer_coeff_t int_coeff, const linear_combination<FieldT> &lc);

template<typename FieldT>
__device__ linear_combination<FieldT> operator*(const FieldT &field_coeff, const linear_combination<FieldT> &lc);

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const integer_coeff_t int_coeff, const linear_combination<FieldT> &lc);

template<typename FieldT>
__device__ linear_combination<FieldT> operator+(const FieldT &field_coeff, const linear_combination<FieldT> &lc);

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const integer_coeff_t int_coeff, const linear_combination<FieldT> &lc);

template<typename FieldT>
__device__ linear_combination<FieldT> operator-(const FieldT &field_coeff, const linear_combination<FieldT> &lc);


}


#include "variable.cu"


#endif