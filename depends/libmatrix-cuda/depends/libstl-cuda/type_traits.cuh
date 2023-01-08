#ifndef __STL_CUDA_TYPE_TRAITS_CUH__
#define __STL_CUDA_TYPE_TRAITS_CUH__

namespace libstl {

template<typename T> 
struct remove_reference {
    typedef T type;
};

template<typename T> 
struct remove_reference<T&> { 
    typedef T type; 
};
 
template<typename T> 
struct remove_reference<T&&> { 
    typedef T type; 
}; 

}

#endif