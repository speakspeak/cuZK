#ifndef __R1CS_CUH__
#define __R1CS_CUH__

#include "../../../../depends/libstl-cuda/vector.cuh"
#include "../../../../depends/libmatrix-cuda/sparse-matrix/csr.cuh"
#include <vector>

namespace cuzk{

template<typename FieldT>
class r1cs_constraint {
public:

    linear_combination<FieldT> a, b, c;

    __device__ r1cs_constraint(){};
    __device__ r1cs_constraint(const FieldT& instance);

    __device__ r1cs_constraint(const linear_combination<FieldT> &a,
                    const linear_combination<FieldT> &b,
                    const linear_combination<FieldT> &c);

    __device__ r1cs_constraint(const r1cs_constraint& other);

    __device__ r1cs_constraint& operator=(const r1cs_constraint& other);
    

    __device__ bool operator==(const r1cs_constraint<FieldT> &other) const;
};


template<typename FieldT>
class r1cs_constraint_system {
public:
    size_t primary_input_size;
    size_t auxiliary_input_size;

    libstl::vector<r1cs_constraint<FieldT>> vconstraints;
    libmatrix::CSR_matrix<FieldT> ma;
    libmatrix::CSR_matrix<FieldT> mb;
    libmatrix::CSR_matrix<FieldT> mc;

    // __device__ r1cs_constraint_system(): primary_input_size(0), auxiliary_input_size(0), constraints(NULL), constraints_size(0), constraints_alloc(0) {}
    __device__ r1cs_constraint_system(): primary_input_size(0), auxiliary_input_size(0) {}
    __device__ r1cs_constraint_system(libstl::vector<r1cs_constraint<FieldT>>&& vconstraints, size_t primary_input_size, size_t auxiliary_input_size); 
    __device__ r1cs_constraint_system(const r1cs_constraint_system& other);
    __device__ r1cs_constraint_system(r1cs_constraint_system<FieldT>&& other);
    
    __device__ r1cs_constraint_system<FieldT>& operator=(const r1cs_constraint_system<FieldT> &other);
    __device__ r1cs_constraint_system<FieldT>& operator=(r1cs_constraint_system<FieldT> &&other);

    __device__ size_t num_inputs() const;
    __device__ size_t num_variables() const;
    __device__ size_t num_constraints() const;

    __host__ size_t num_inputs_host();
    __host__ size_t num_variables_host();
    __host__ size_t num_constraints_host();

    __device__ bool is_valid() const;
    // __device__ bool is_satisfied(FieldT* primary_input, size_t primary_input_size,
    //                   FieldT* auxiliary_input, size_t auxiliary_input_size) const;

    __device__ bool is_satisfied(const libstl::vector<FieldT>& primary_input, const libstl::vector<FieldT>& auxiliary_input, const FieldT& instance) const;

    __device__ void add_constraint(const r1cs_constraint<FieldT> &c, const FieldT& instance);

    __device__ void add_constraint(libstl::vector<r1cs_constraint<FieldT>> cs, const FieldT& instance);

    __device__ void swap_AB_if_beneficial();

    __device__ bool operator==(const r1cs_constraint_system<FieldT> &other) const;
};

template<typename FieldT>
struct r1cs_constraint_system_host {

    size_t primary_input_size;
    size_t auxiliary_input_size;

    libmatrix::CSR_matrix_host<FieldT> ma;
    libmatrix::CSR_matrix_host<FieldT> mb;
    libmatrix::CSR_matrix_host<FieldT> mc;
};


template<typename FieldT>
using r1cs_primary_input = libstl::vector<FieldT>;

template<typename FieldT>
using r1cs_auxiliary_input = libstl::vector<FieldT>;

template<typename FieldT>
using r1cs_primary_input_host = libstl::vector<FieldT>;

template<typename FieldT>
using r1cs_auxiliary_input_host = libstl::vector<FieldT>;


template<typename FieldT_host, typename FieldT_device>
void r1cs_constraint_system_device2host(r1cs_constraint_system_host<FieldT_host>* hcs, r1cs_constraint_system<FieldT_device>* dcs);

template<typename FieldT_host, typename FieldT_device>
void r1cs_constraint_system_host2device(r1cs_constraint_system<FieldT_device>* dcs, r1cs_constraint_system_host<FieldT_host>* hcs, FieldT_device* instance);

template<typename FieldT_host, typename FieldT_device>
void r1cs_primary_input_device2host(r1cs_primary_input_host<FieldT_host>* hpi, r1cs_primary_input<FieldT_device>* dpi);

template<typename FieldT_host, typename FieldT_device>
void r1cs_primary_input_host2device(r1cs_primary_input<FieldT_device>* dpi, r1cs_primary_input_host<FieldT_host>* hpi, FieldT_device* instance);

template<typename FieldT_host, typename FieldT_device>
void r1cs_auxiliary_input_device2host(r1cs_auxiliary_input_host<FieldT_host>* hai, r1cs_auxiliary_input<FieldT_device>* dai);

template<typename FieldT_host, typename FieldT_device>
void r1cs_auxiliary_input_host2device(r1cs_auxiliary_input<FieldT_device>* dai, r1cs_auxiliary_input_host<FieldT_host>* hai);

}

#include "r1cs.cu"

#endif