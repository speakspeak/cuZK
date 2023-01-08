#ifndef __BASIC_OPERATIONS_CUH__
#define __BASIC_OPERATIONS_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace libfqfft {
    
/**
 * Returns true if polynomial A is a zero polynomial.
 */
template<typename FieldT>
__device__ bool _is_zero(const libstl::vector<FieldT>& a);

/**
 * Removes extraneous zero entries from in vector representation of polynomial.
 * Example - Degree-4 Polynomial: [0, 1, 2, 3, 4, 0, 0, 0, 0] -> [0, 1, 2, 3, 4]
 * Note: Simplest condensed form is a zero polynomial of vector form: [0]
 */
template<typename FieldT>
__device__ void _condense(libstl::vector<FieldT>& a);

/**
 * Compute the reverse polynomial up to vector size n (degree n-1).
 * Below we make use of the reversal endomorphism definition from
 * [Bostan, Lecerf,&  Schost, 2003. Tellegen's Principle in Practice, on page 38].
 */
template<typename FieldT>
__device__ void _reverse(libstl::vector<FieldT>& a, const size_t n);

/**
 * Computes the standard polynomial addition, polynomial A + polynomial B, and stores result in polynomial C.
 */
template<typename FieldT>
__device__ void _polynomial_addition(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b);

/**
 * Computes the standard polynomial subtraction, polynomial A - polynomial B, and stores result in polynomial C.
 */
template<typename FieldT>
__device__ void _polynomial_subtraction(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b);

/**
 * Perform the multiplication of two polynomials, polynomial A * polynomial B, and stores result in polynomial C.
 */
template<typename FieldT>
__device__ void _polynomial_multiplication(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b);

/**
 * Perform the multiplication of two polynomials, polynomial A * polynomial B, using FFT, and stores result in polynomial C.
 */
template<typename FieldT>
__device__ void _polynomial_multiplication_on_fft(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b);

/**
 * Perform the multiplication of two polynomials, polynomial A * polynomial B, using Kronecker Substitution, and stores result in polynomial C.
 */
template<typename FieldT>
__device__ void _polynomial_multiplication_on_kronecker(libstl::vector<FieldT>& c, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b);

/**
 * Compute the transposed, polynomial multiplication of vector a and vector b.
 * Below we make use of the transposed multiplication definition from
 * [Bostan, Lecerf,&  Schost, 2003. Tellegen's Principle in Practice, on page 39].
 */
template<typename FieldT>
__device__ libstl::vector<FieldT> _polynomial_multiplication_transpose(const size_t& n, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& c);

/**
 * Perform the standard Euclidean Division algorithm.
 * Input: Polynomial A, Polynomial B, where A / B
 * Output: Polynomial Q, Polynomial R, such that A = (Q * B) + R.
 */
template<typename FieldT>
__device__ void _polynomial_division(libstl::vector<FieldT>& q, libstl::vector<FieldT>& r, const libstl::vector<FieldT>& a, const libstl::vector<FieldT>& b);

}

#include "basic_operations.cu"

#endif