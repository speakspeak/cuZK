
#ifndef __R1CS_TO_QAP_CUH__
#define __R1CS_TO_QAP_CUH__

#include "../../relations/arithmetic_programs/qap/qap.cuh"
#include "../../relations/constraint_satisfaction_problems/r1cs/r1cs.cuh"
#include "../../../depends/libff-cuda/curves/public_params.cuh"
#include "../../../depends/libmatrix-cuda/sparse-matrix/csr.cuh"
#include "../../../depends/libmatrix-cuda/transpose/transpose_csr2csr.cuh"
#include "../../../depends/libmatrix-cuda/spmv/csr-scalar.cuh"
#include "../../../depends/libmatrix-cuda/spmv/csr-balanced.cuh"

namespace cuzk {
    
/**
 * Instance map for the R1CS-to-QAP reduction.
 */
template<typename FieldT>
__device__ qap_instance<FieldT> r1cs_to_qap_instance_map(const r1cs_constraint_system<FieldT> &cs, const FieldT &instance);

/**
 * Instance map for the R1CS-to-QAP reduction followed by evaluation of the resulting QAP instance.
 */

template<typename ppT>
void r1cs_to_qap_instance_map_with_evaluation(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip);


// template<typename ppT>
// void r1cs_to_qap_witness_map_host(prover_params<ppT>* pp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip);


template<typename FieldT>
qap_witness<FieldT>* r1cs_to_qap_witness_map(r1cs_constraint_system<FieldT>* cs,
                                            r1cs_primary_input<FieldT>* primary_input,
                                            r1cs_auxiliary_input<FieldT>* auxiliary_input,
                                            FieldT* instance);


                                            
}

#include "r1cs_to_qap.cu"

#endif