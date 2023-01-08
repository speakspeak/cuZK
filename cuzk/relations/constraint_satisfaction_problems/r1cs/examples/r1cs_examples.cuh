#ifndef __R1CS_EXAMPLES_CUH__
#define __R1CS_EXAMPLES_CUH__


#include "../../../../../depends/libstl-cuda/vector.cuh"
#include "../r1cs.cuh"

namespace cuzk {
template<typename FieldT>
struct r1cs_example {
    r1cs_constraint_system<FieldT> constraint_system;
    r1cs_primary_input<FieldT> r1cs_primary_input;
    r1cs_auxiliary_input<FieldT> r1cs_auxiliary_input;

    // __device__ r1cs_example<FieldT>(){};

    __device__ r1cs_example<FieldT>(const r1cs_constraint_system<FieldT> &constraint_system,
                         const libstl::vector<FieldT>& primary_input,
                         const libstl::vector<FieldT>& auxiliary_input):
                    constraint_system(constraint_system),
                    r1cs_primary_input(primary_input),
                    r1cs_auxiliary_input(auxiliary_input)
                    {};

    __device__ r1cs_example<FieldT>(r1cs_constraint_system<FieldT>&& constraint_system,
                        libstl::vector<FieldT>&& primary_input,
                        libstl::vector<FieldT>&& auxiliary_input):
                    constraint_system(libstl::move(constraint_system)),
                    r1cs_primary_input(libstl::move(primary_input)),
                    r1cs_auxiliary_input(libstl::move(auxiliary_input))
                    {};


    __device__ r1cs_example<FieldT>(const r1cs_example<FieldT>& other):
    constraint_system(other.constraint_system),
    r1cs_primary_input(other.r1cs_primary_input),
    r1cs_auxiliary_input(other.r1cs_primary_input)
    {};
    
    __device__ r1cs_example<FieldT>(r1cs_example<FieldT>&& other):
    constraint_system(libstl::move(other.constraint_system)),
    r1cs_primary_input(libstl::move(other.r1cs_primary_input)),
    r1cs_auxiliary_input(libstl::move(other.r1cs_primary_input))
    {};

    

    __device__ r1cs_example<FieldT>& operator=(const r1cs_example<FieldT> &other)
    {
        constraint_system = other.constraint_system;
        r1cs_primary_input = other.r1cs_primary_input;
        r1cs_auxiliary_input = other.r1cs_auxiliary_input;

        return *this;
    };

    __device__ r1cs_example<FieldT>& operator=(r1cs_example<FieldT> &&other)
    {
        constraint_system = libstl::move(other.constraint_system);
        r1cs_primary_input = libstl::move(other.r1cs_primary_input);
        r1cs_auxiliary_input = libstl::move(other.r1cs_auxiliary_input);

        return *this;
    }

    // __device__ r1cs_example<FieldT>(const r1cs_example<FieldT>& other):
    //                 constraint_system(other.constraint_system),
    //                 r1cs_primary_input(other.primary_input),
    //                 r1cs_auxiliary_input(other.auxiliary_input)
    //                 {};



};


template<typename FieldT>
__device__ r1cs_example<FieldT> generate_r1cs_example_with_field_input(const size_t num_constraints,
                                                            const size_t num_inputs, const FieldT& instance);

template<typename FieldT>
__device__ r1cs_example<FieldT> generate_r1cs_example_like_bellperson(const size_t num_constraints, const FieldT& instance);


template<typename FieldT>
__device__ r1cs_example<FieldT> generate_r1cs_example_from_file(char* r1cs_file, size_t length, const FieldT& instance);

// template<typename FieldT>
// __device__ r1cs_example<FieldT> generate_r1cs_example_with_binary_input(const size_t num_constraints,
//                                                              const size_t num_inputs, const FieldT& instance);

}


#include "r1cs_examples.cu"

#endif