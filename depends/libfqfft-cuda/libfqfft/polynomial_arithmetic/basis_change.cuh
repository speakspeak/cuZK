#ifndef __BASIS_CHANGE_CUH__
#define __BASIS_CHANGE_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace libfqfft {

/**
 * Compute the Subproduct Tree of degree 2^M and store it in Tree T.
 * Below we make use of the Subproduct Tree description from
 * [Bostan and Schost 2005. Polynomial Evaluation and Interpolation on Special Sets of Points], on page 7.
 */
template<typename FieldT>
__device__ void compute_subproduct_tree(const size_t& m, const FieldT& instance, libstl::vector<libstl::vector<libstl::vector<FieldT>>>& T);

/**
 * Perform the general change of basis from Monomial to Newton Basis with Subproduct Tree T.
 * Below we make use of the MonomialToNewton and TNewtonToMonomial pseudocode from
 * [Bostan and Schost 2005. Polynomial Evaluation and Interpolation on Special Sets of Points], on page 12 and 14.
 */
template<typename FieldT>
__device__ void monomial_to_newton_basis(libstl::vector<FieldT>& a, const libstl::vector<libstl::vector<libstl::vector<FieldT>>>& T, const size_t& n);

/**
 * Perform the general change of basis from Newton to Monomial Basis with Subproduct Tree T.
 * Below we make use of the NewtonToMonomial pseudocode from
 * [Bostan and Schost 2005. Polynomial Evaluation and Interpolation on Special Sets of Points], on page 11.
 */
template<typename FieldT>
__device__ void newton_to_monomial_basis(libstl::vector<FieldT>& a, const libstl::vector<libstl::vector<libstl::vector<FieldT>>>& T, const size_t& n);

/**
 * Perform the change of basis from Monomial to Newton Basis for geometric sequence.
 * Below we make use of the psuedocode from
 * [Bostan& Schost 2005. Polynomial Evaluation and Interpolation on Special Sets of Points] on page 26.
 */
template<typename FieldT>
__device__ void monomial_to_newton_basis_geometric(libstl::vector<FieldT>& a,
                                             const libstl::vector<FieldT>& geometric_sequence,
                                             const libstl::vector<FieldT>& geometric_triangular_sequence,
                                             const size_t& n);

/**
 * Perform the change of basis from Newton to Monomial Basis for geometric sequence
 * Below we make use of the psuedocode from
 * [Bostan& Schost 2005. Polynomial Evaluation and Interpolation on Special Sets of Points] on page 26.
 */
template<typename FieldT>
__device__ void newton_to_monomial_basis_geometric(libstl::vector<FieldT>& a,
                                             const libstl::vector<FieldT>& geometric_sequence,
                                             const libstl::vector<FieldT>& geometric_triangular_sequence,
                                             const size_t& n);
 
} // libfqfft

#include "basis_change.cu"

#endif