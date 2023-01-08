#ifndef __STL_CUDA_UTILITY_CUH__
#define __STL_CUDA_UTILITY_CUH__

#include "type_traits.cuh"

namespace libstl {

template<typename T>
__device__ typename remove_reference<T>::type&& move(T&& arg) { return static_cast<typename remove_reference<T>::type&&>(arg); }

template<typename T> 
__device__ void swap (T& a, T& b) { T c(move(a)); a = move(b); b = move(c); }

template<typename T, size_t N> 
__device__ void swap(T (&a)[N], T (&b)[N]) { for (size_t i = 0; i < N; i++) swap(a[i], b[i]);}

template<typename T>
__device__ T&& forward(typename remove_reference<T>::type& arg) { return static_cast<T&&>(arg); }

template<typename T>
__device__ T&& forward(typename remove_reference<T>::type&& arg) { return static_cast<T&&>(arg); }

template<typename Func, typename ...Args>
__global__ void launch(const Func func, Args&&... args) { func(forward<Args>(args)...); }

template<typename Func, typename ...Args>
__global__ void launch_with_shared(const Func func, Args&&... args)
{ 
    extern __shared__ unsigned char s[];
    func(forward<Args>(args)..., s);
}

template<typename T1, typename T2>
struct pair {
    
    typedef T1           first_type;
    typedef T2           second_type;
    typedef pair<T1, T2> self_type;

    first_type first;
    second_type second;

    __device__ pair(const first_type& first, const second_type& second) : first(first), second(second) {}

    __device__ pair(const self_type& other) : first(other.first), second(other.second) {}

    __device__ pair(self_type&& other) : first(move(other.first)), second(move(other.second)) {} 
};

template<typename T1, typename T2>
__device__ pair<T1, T2> make_pair(const T1& t, const T2& u) { return pair<T1, T2>(t, u); } 

}


#endif