#ifndef __STL_CUDA_LIST_CU__
#define __STL_CUDA_LIST_CU__

namespace libstl {

template<typename T>
__device__ list<T>::list() : _head(nullptr), _tail(nullptr)
{

}

template<typename T>
__device__ list<T>::list(const self_type& x) : _head(nullptr), _tail(nullptr)
{
    if (x.begin() == x.end())
        return;

    auto it = x.begin();

    _head = _tail = (node_type*)_allocManager.allocate(sizeof(node_type));
    construct(_head, *it);

    ++it;

    for (; it != x.end(); ++it)
    {
        node_type* node = (node_type*)_allocManager.allocate(sizeof(node_type));
        construct(node, *it);

        _tail->next = node;
        _tail = node;
    }
}

template<typename T>
__device__ list<T>::list(self_type&& x) : _head(x._head), _tail(x._tail)
{
    x._head = x._tail = nullptr;
}

template<typename T>
__device__ list<T>::list(const_iterator first, const_iterator last) : _head(nullptr), _tail(nullptr)
{
    if (first == last)
        return;

    auto it = first;

    _head = _tail = (node_type*)_allocManager.allocate(sizeof(node_type));
    construct(_head, *it);

    ++it;

    for (; it != last; ++it)
    {
        node_type* node = (node_type*)_allocManager.allocate(sizeof(node_type));
        construct(node, *it);

        _tail->next = node;
        _tail = node;
    }
}

template<typename T>
__device__ list<T>::~list()
{

}

template<typename T>
__device__ list<T>& list<T>::operator=(const self_type& x)
{
    _head = _tail = nullptr;

    if (x.begin() == x.end())
        return *this;

    auto it = x.begin();

    _head = _tail = (node_type*)_allocManager.allocate(sizeof(node_type));
    construct(_head, *it);

    ++it;

    for (; it != x.end(); ++it)
    {
        node_type* node = (node_type*)_allocManager.allocate(sizeof(node_type));
        construct(node, *it);

        _tail->next = node;
        _tail = node;
    }

    return *this;
}

template<typename T>
__device__ list<T>& list<T>::operator=(self_type&& x)
{
    _head = x._head;
    _tail = x._tail;
    x._head = x._tail = nullptr;
    
    return *this;
}

template<typename T>
__device__ list<T>::iterator list<T>::begin()
{
    return iterator(_head);
}

template<typename T>
__device__ list<T>::const_iterator list<T>::begin() const
{
    return const_iterator(_head);
}

template<typename T>
__device__ list<T>::const_iterator list<T>::cbegin() const
{
    return const_iterator(_head);
}

template<typename T>
__device__ list<T>::iterator list<T>::end()
{
    return iterator(nullptr);
}

template<typename T>
__device__ list<T>::const_iterator list<T>::end() const
{
    return const_iterator(nullptr);
}

template<typename T>
__device__ list<T>::const_iterator list<T>::cend() const
{
    return const_iterator(nullptr);
}

template<typename T>
__device__ void list<T>::emplace_front(const T& val)
{
    node_type* node = (node_type*)_allocManager.allocate(sizeof(node_type));
    construct(node, val);

    if (_head == nullptr)
        _head = _tail = node;
    else
    {
        node->next = _head;
        _head = node;
    }
}

template<typename T>
__device__ void list<T>::emplace_back(const T& val)
{
    node_type* node = (node_type*)_allocManager.allocate(sizeof(node_type));
    construct(node, val);

    if (_head == nullptr)
        _head = _tail = node;
    else
    {
        _tail->next = node;
        _tail = node;
    }
}

template<typename T>
__device__ size_t list<T>::memory_need(size_t n)
{
    return n * sizeof(node_type);
}

}

#endif