#ifndef __GET_EVALUATION_DOMAIN_CU__
#define __GET_EVALUATION_DOMAIN_CU__

#include "evaluation_domain.cuh"
#include "domains/basic_radix2_domain.cuh"
#include "domains/extended_radix2_domain.cuh"
#include "domains/step_radix2_domain.cuh"
//#include "domains/arithmetic_sequence_domain.cuh"
//#include "domains/geometric_sequence_domain.cuh"

namespace libfqfft {

// template<typename FieldT>
// __device__ evaluation_domain<FieldT>* get_evaluation_domain(const size_t min_size, const FieldT& instance)
// {
//     evaluation_domain<FieldT>* result;
    
//     result = new basic_radix2_domain<FieldT>(min_size, instance);

//     return result;
// }

}

#endif