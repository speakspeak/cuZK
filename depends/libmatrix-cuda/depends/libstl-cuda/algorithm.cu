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

template<typename T, typename S>
__host__ void sort_pair_host(vector<T>* key, vector<S>* input, vector<T>* key_out, vector<S>* output, size_t n, size_t start_bit, size_t end_bit)
{
    T* key_addr, key_out_addr;
    S* input_addr, output_addr;
    libstl::get_host(&key_addr, &key->_data);
    libstl::get_host(&input_addr, &input->_data);
    libstl::get_host(&key_out_addr, &key_out->_data);
    libstl::get_host(&output_addr, &output->_data);

    void *d_temp = NULL;
    size_t sort_size = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, sort_size, key_addr, input_addr, key_out_addr, output_addr, n, start_bit, end_bit);  // Determine temporary device storage requirements
    cudaMalloc(&d_temp, sort_size);
    cub::DeviceRadixSort::SortPairs(d_temp, sort_size, key_addr, input_addr, key_out_addr, output_addr, n, start_bit , end_bit);
    cudaFree(d_temp);
}

template<typename T>
__host__ void run_length_host(vector<T>* input, vector<T>* unique, vector<T>* count, vector<T>* unique_count, size_t n)
{
    T* input_addr, unique_addr, count_addr, unique_count_addr;
    libstl::get_host(&input_addr, &input->_data);
    libstl::get_host(&unique_addr, &unique->_data);
    libstl::get_host(&count_addr, &count->_data);
    libstl::get_host(&unique_count_addr, &unique_count->_data);

    void *d_temp = NULL;
    size_t encode_size = 0;
    cub::DeviceRunLengthEncode::Encode(d_temp, encode_size, input_addr, unique_addr, count_addr, unique_count_addr, n);
    cudaMalloc(&d_temp, encode_size);
    cub::DeviceRunLengthEncode::Encode(d_temp, encode_size, input_addr, unique_addr, count_addr, unique_count_addr, n);
    cudaFree(d_temp);
}

template<typename T>
__host__ void inclusive_sum(vector<T>* input,  vector<T>* output, size_t n)
{
    T* input_addr, output_addr;
    libstl::get_host(&input_addr, &input->_data);
    libstl::get_host(&output_addr, &output->_data);

    void *d_temp = NULL;
    size_t sum_size = 0;
    cub::DeviceScan::InclusiveSum(d_temp, sum_size, input_addr, output_addr, n);
    cudaMalloc(&d_temp, sum_size);
    cub::DeviceScan::InclusiveSum(d_temp, sum_size, input_addr, output_addr, n);
    cudaFree(d_temp);
}


}

#endif