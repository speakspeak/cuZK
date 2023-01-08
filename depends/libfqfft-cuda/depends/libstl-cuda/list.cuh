#ifndef __STL_CUDA_LIST_CUH__
#define __STL_CUDA_LIST_CUH__

namespace libstl {

template <class T>
struct list_node {
    T value;
    list_node* next;

    __device__ list_node() : next(nullptr) {}

    __device__ list_node(T value, list_node* next = nullptr) : value(value), next(next) {}
};

template <class T>
struct list_iterator : public iterator<forward_iterator_tag, T> {

    typedef T                value_type;
    typedef T*               pointer;
    typedef T&               reference;
    typedef list_node<T>     node_type;
    typedef list_iterator<T> self_type;

    node_type* cur;

    __device__ list_iterator(node_type* x) :cur(x) {}

    __device__ list_iterator(const self_type& other) :cur(other.cur) {}

    __device__ self_type& operator=(const self_type& other) { cur = other.cur; return *this; }

    __device__ reference operator*()  const { return cur->value; }

    __device__ pointer   operator->() const { return &(operator*()); }

    __device__ self_type& operator++() { cur = cur->next; return *this;}

    __device__ self_type operator++(int) { auto temp = *this; ++*this; return temp;}

    __device__ bool operator==(const self_type& other) const { return cur == other.cur; }

    __device__ bool operator!=(const self_type& other) const { return cur != other.cur; }
};

template <class T>
struct list_const_iterator : public iterator<forward_iterator_tag, T> {

    typedef T                       value_type;
    typedef const T*                pointer;
    typedef const T&                reference;
    typedef list_node<T>            node_type;
    typedef list_const_iterator<T>  self_type;

    node_type* cur;

    __device__ list_const_iterator(node_type* x) :cur(x) {}

    __device__ list_const_iterator(const self_type& other) :cur(other.cur) {}

    __device__ self_type& operator=(const self_type& other) { cur = other.cur; return *this; }

    __device__ reference operator*()  const { return cur->value; }

    __device__ pointer   operator->() const { return &(operator*()); }

    __device__ self_type& operator++() { cur = cur->next; return *this;}

    __device__ self_type operator++(int) { auto temp = *this; ++*this; return temp;}

    __device__ bool operator==(const self_type& other) const { return cur == other.cur; }

    __device__ bool operator!=(const self_type& other) const { return cur != other.cur; }
};


template<typename T>
class list {
public:

    typedef list_iterator<T>        iterator;
    typedef list_const_iterator<T>  const_iterator;
    typedef list_node<T>            node_type;
    typedef list<T>                 self_type;

    __device__ list();

    __device__ list(const self_type& x);

    __device__ list(self_type&& x);

    __device__ list(const_iterator first, const_iterator last);

    __device__ ~list();

    __device__ self_type& operator=(const self_type& x);

    __device__ self_type& operator=(self_type&& x);

    __device__ iterator begin();

    __device__ const_iterator begin() const;

    __device__ const_iterator cbegin() const;

    __device__ iterator end();

    __device__ const_iterator end() const;

    __device__ const_iterator cend() const;

    __device__ void emplace_front(const T& val);

    __device__ void emplace_back(const T& val);

    __device__ inline void set_serial_allocator(SerialAllocator* allocator = mainAllocator) { _allocManager.setSerialAllocator(allocator); }

    __device__ inline void set_parallel_allocator(ParallalAllocator* allocator) { _allocManager.setParallelAllocator(allocator); }

    __device__ static size_t memory_need(size_t n);

private:
    node_type* _head;
    node_type* _tail;

    AllocatorManager _allocManager;
};

}

#include "list.cu"

#endif