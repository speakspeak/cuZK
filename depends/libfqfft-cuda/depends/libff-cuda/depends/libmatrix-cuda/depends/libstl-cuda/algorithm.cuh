#ifndef __STL_CUDA_ALGORITHM_CUH__
#define __STL_CUDA_ALGORITHM_CUH__

namespace libstl {

template<typename InputIterator, typename UnaryPredicate>
__device__ bool all_of(InputIterator first, InputIterator last, UnaryPredicate pred);

template<typename InputIterator, typename UnaryPredicate>
__device__ bool any_of(InputIterator first, InputIterator last, UnaryPredicate pred);

template<typename InputIterator, typename UnaryPredicate>
__device__ bool none_of(InputIterator first, InputIterator last, UnaryPredicate pred);

template <typename InputIterator, typename OutputIterator>
__device__ OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result);

template <class ForwardIterator1, class ForwardIterator2>
__device__ void iter_swap(ForwardIterator1 a, ForwardIterator2 b);

template <typename InputIterator, typename OutputIterator, typename UnaryOperation>
__device__ OutputIterator transform(InputIterator first1, InputIterator last1,
                                    OutputIterator result, UnaryOperation op);

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryOperation>
__device__ OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                                    InputIterator2 first2, OutputIterator result,
                                    BinaryOperation binary_op);

template<typename InputIterator, typename UnaryOperation>
__device__ void for_each(InputIterator first, InputIterator last, UnaryOperation op);

template <typename BidirectionalIterator>
__device__ void reverse(BidirectionalIterator first, BidirectionalIterator last);

template <typename ForwardIterator>
__device__ ForwardIterator min_element(ForwardIterator first, ForwardIterator last);

template <typename ForwardIterator>
__device__ ForwardIterator max_element(ForwardIterator first, ForwardIterator last);

}

#include "algorithm.cu"

#endif

