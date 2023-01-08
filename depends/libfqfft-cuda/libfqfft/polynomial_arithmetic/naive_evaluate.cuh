#ifndef __NAIVE_EVALUATE_CUH__
#define __NAIVE_EVALUATE_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace libfqfft {

/**
 * Naive evaluation of a *single* polynomial, used for testing purposes.
 *
 * The inputs are:
 * - an integer m
 * - a vector coeff representing monomial P of size m
 * - a field element element t
 * The output is the polynomial P(x) evaluated at x = t.
 */
//template<typename FieldT>
//__device__ FieldT evaluate_polynomial(const size_t& m, const libstl::vector<FieldT>& coeff, const FieldT& t);

/**
 * Naive evaluation of a *single* Lagrange polynomial, used for testing purposes.
 *
 * The inputs are:
 * - an integer m
 * - a domain S = (a_{0},...,a_{m-1}) of size m
 * - a field element element t
 * - an index idx in {0,...,m-1}
 * The output is the polynomial L_{idx,S}(z) evaluated at z = t.
 */
//template<typename FieldT>
//__device__ FieldT evaluate_lagrange_polynomial(const size_t& m, const libstl::vector<FieldT> &domain, const FieldT& t, const size_t& idx);

} // libfqfft

#include "naive_evaluate.cu"

#endif