#ifndef __STL_CUDA_ALGORITHM_CU__
#define __STL_CUDA_ALGORITHM_CU__

#include "utility.cuh"

namespace libstl {

template<typename InputIterator, typename UnaryPredicate>
__device__ bool all_of(InputIterator first, InputIterator last, UnaryPredicate pred)
{
    for (; first!= last; ++first)
        if (!pred(*first))
            return false;

    return true;
}

template<typename InputIterator, typename UnaryPredicate>
__device__ bool any_of(InputIterator first, InputIterator last, UnaryPredicate pred)
{
    for (; first!= last; ++first)
        if (pred(*first))
            return true;

    return false;
}

template<typename InputIterator, typename UnaryPredicate>
__device__ bool none_of(InputIterator first, InputIterator last, UnaryPredicate pred)
{
    for (; first!= last; ++first)
        if (pred(*first))
            return false;

    return true;
}

template <typename InputIterator, typename OutputIterator>
__device__ OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result)
{
    for (; first!= last; ++first, ++result)
        *result = *first;

    return result;
}

template <class ForwardIterator1, class ForwardIterator2>
__device__ void iter_swap(ForwardIterator1 a, ForwardIterator2 b)
{
    swap(*a, *b);
}

template <typename InputIterator, typename OutputIterator, typename UnaryOperation>
__device__ OutputIterator transform(InputIterator first1, InputIterator last1,
                                    OutputIterator result, UnaryOperation op)
{
    for(; first1 != last1; ++first1, ++result)
        *result = op(*first1);
       
    return result;
}

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryOperation>
__device__ OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                                    InputIterator2 first2, OutputIterator result,
                                    BinaryOperation binary_op)
{
    for(; first1 != last1; ++first1, ++first2, ++result)
        *result = binary_op(*first1, *first2);

    return result;
}

template<typename InputIterator, typename UnaryOperation>
__device__ void for_each(InputIterator first, InputIterator last, UnaryOperation op)
{
    for(; first != last; ++first)
        op(*first);
}

template <typename BidirectionalIterator>
__device__ void reverse(BidirectionalIterator first, BidirectionalIterator last)
{
    for(; (first != last) && (first != --last); ++first)
        iter_swap (first, last);
}

template <typename ForwardIterator>
__device__ ForwardIterator min_element(ForwardIterator first, ForwardIterator last)
{
    if (first == last) 
        return last;
  
    ForwardIterator smallest = first;

    while (++first != last)
        if (*first < *smallest)
            smallest = first;

    return smallest;
}

template <typename ForwardIterator>
__device__ ForwardIterator max_element(ForwardIterator first, ForwardIterator last)
{
    if (first == last) 
        return last;

    ForwardIterator largest = first;

    while (++first != last)
        if (*largest < *first) 
            largest = first;

    return largest;
}

}

#endif