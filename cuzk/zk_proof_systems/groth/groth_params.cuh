#ifndef __GROTH_PARAMS_CUH__
#define __GROTH_PARAMS_CUH__


#include "../../../depends/libff-cuda/curves/public_params.cuh"
#include "../../relations/constraint_satisfaction_problems/r1cs/r1cs.cuh"

namespace cuzk{

template<typename ppT>
using groth_constraint_system = r1cs_constraint_system<libff::Fr<ppT> >;

}

#endif