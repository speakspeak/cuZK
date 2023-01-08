#ifndef __GET_EVALUATION_DOMAIN_CUH__
#define __GET_EVALUATION_DOMAIN_CUH__

#include "domains/basic_radix2_domain.cuh"
#include "domains/extended_radix2_domain.cuh"
#include "domains/step_radix2_domain.cuh"
//#include "domains/arithmetic_sequence_domain.cuh"
//#include "domains/geometric_sequence_domain.cuh"

namespace libfqfft {

template<typename FieldT, template<typename T> class Domain>
__device__ Domain<FieldT>* get_evaluation_domain(const size_t min_size, FieldT* instance) { return libstl::create<Domain<FieldT>>(Domain<FieldT>(min_size, instance));}

};

#endif