#ifndef __STL_CUDA_FUNCTIONAL_CUH__
#define __STL_CUDA_FUNCTIONAL_CUH__

namespace libstl {

template<typename Arg, typename Result>
struct unary_function {
    typedef Arg argument_type;
    typedef Result result_type;
};

template<typename Arg1, typename Arg2, typename Result>
struct binary_function {
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};

template<typename T> 
struct plus : public binary_function<T, T, T> {  
    __device__ T operator() (const T& x, const T& y) const { return x + y; }
};

template<typename T> 
struct minus : public binary_function<T, T, T> {  
    __device__ T operator() (const T& x, const T& y) const { return x - y; }
};

template<typename T> 
struct multiplies : public binary_function<T, T, T> {
    __device__ T operator() (const T& x, const T& y) const { return x * y; }
};

template<typename T> 
struct divides : public binary_function<T, T, T> {
    __device__ T operator() (const T& x, const T& y) const { return x / y; }
};

template<typename T> 
struct modulus : public binary_function<T, T, T> {
    __device__ T operator() (const T& x, const T& y) const { return x % y; }
};

template<typename T> 
struct negate : public unary_function<T, T> {
    __device__ T operator() (const T& x) const { return -x; }
};

template<typename Operation> 
class binder1st: public unary_function<typename Operation::second_argument_type,
                                       typename Operation::result_type> {
protected:
    Operation op;
    typename Operation::first_argument_type value;

public:
    __device__ binder1st(const Operation& x, 
                         const typename Operation::first_argument_type& y) : op (x), value(y) {}

    __device__ typename Operation::result_type operator()
                    (const typename Operation::second_argument_type& x) const 
                        { return op(value, x); }
};

template<typename Operation> 
class binder2nd : public unary_function<typename Operation::first_argument_type,
                                        typename Operation::result_type>
{
protected:
    Operation op;
    typename Operation::second_argument_type value;

public:
    __device__ binder2nd(const Operation& x,
                         const typename Operation::second_argument_type& y) : op (x), value(y) {}

    __device__ typename Operation::result_type operator()
                    (const typename Operation::first_argument_type& x) const 
                        { return op(x, value); }
};

template<typename Operation, typename T>
__device__ binder1st<Operation> bind1st (const Operation& op, const T& x)
{
    return binder1st<Operation>(op, typename Operation::first_argument_type(x));
}

template<typename Operation, typename T>
__device__ binder2nd<Operation> bind2nd (const Operation& op, const T& x)
{
    return binder2nd<Operation>(op, typename Operation::second_argument_type(x));
}


template<typename Key>
struct hash {

};

template<>
struct hash<bool> {
    __device__ size_t operator()(bool val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<char> {
    __device__ size_t operator()(char val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<signed char> {
    __device__ size_t operator()(signed char val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<unsigned char> {
    __device__ size_t operator()(unsigned char val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<wchar_t> {
    __device__ size_t operator()(wchar_t val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<short> {
    __device__ size_t operator()(short val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<unsigned short> {
    __device__ size_t operator()(unsigned short val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<int> {
    __device__ size_t operator()(int val) const { return static_cast<size_t>(val); } 
};

template<>
struct hash<unsigned int> {
    __device__ size_t operator()(unsigned int val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<long> {
    __device__ size_t operator()(long val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<long long> {
    __device__ size_t operator()(long long val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<unsigned long> {
    __device__ size_t operator()(unsigned long val) const { return static_cast<size_t>(val); }
};

template<>
struct hash<unsigned long long> {
    __device__ size_t operator()(unsigned long long val) const { return static_cast<size_t>(val); }
};

template<typename T>
struct hash<T*> {
    __device__ size_t operator()(T* ptr) const { return reinterpret_cast<size_t>(ptr); }
};

}

#endif