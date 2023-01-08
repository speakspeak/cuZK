#ifndef __STL_CUDA_UNORDERED_MAP_CUH__
#define __STL_CUDA_UNORDERED_MAP_CUH__

#include "memory.cuh"
#include "iterator.cuh"
#include "utility.cuh"
#include "functional.cuh"

namespace libstl {

template<typename Key, typename T, typename Hash>
class unordered_map;

template<typename Key, typename T, typename Hash>
struct unordered_map_iterator : public iterator<forward_iterator_tag, T> {
 
    typedef pair<Key, T>                             value_type;
    typedef pair<Key, T>*                            pointer;
    typedef pair<Key, T>&                            reference;
    typedef list_iterator<pair<Key, T>>              node_type;
    typedef unordered_map<Key, T, Hash>              container_type;
    typedef unordered_map_iterator<Key, T, Hash>     self_type;

    node_type node;
    container_type* container;

    size_t index;

    __device__ unordered_map_iterator() {}

    __device__ unordered_map_iterator(const node_type& node, container_type* container, size_t index) : node(node), container(container), index(index) {}

    __device__ unordered_map_iterator(const self_type& other) : node(other.node), container(other.container), index(other.index) {}

    __device__ self_type& operator=(const self_type& other) { node = other.node; container = other.container; index = other.index; return *this; }

    __device__ reference operator*() const { return *node; }

    __device__ pointer   operator->() const { return &(operator*()); }

    __device__ self_type& operator++() 
    { 
        node_type nil = node_type(nullptr);

        if (++node == nil)
            while (node == nil && ++index < container->_buckets.size())
                node = container->_buckets[index].begin();

        return *this;
    }

    __device__ self_type operator++(int) 
    { 
        self_type temp = *this;
        ++*this;
        return temp;
    }

    __device__ bool operator==(const self_type& other) const { return node == other.node; }

    __device__ bool operator!=(const self_type& other) const { return node != other.node; }

};

template<typename Key, typename T, typename Hash>
struct unordered_map_const_iterator : public iterator<forward_iterator_tag, T> {
 
    typedef T                                              value_type;
    typedef const T*                                       pointer;
    typedef const T&                                       reference;
    typedef list_const_iterator<pair<Key, T>>              node_type;
    typedef unordered_map<Key, T, Hash>                    container_type;
    typedef unordered_map_const_iterator<Key, T, Hash>     self_type;

    node_type node;
    container_type* container;

    size_t index;

    __device__ unordered_map_const_iterator() {}

    __device__ unordered_map_const_iterator(const node_type& node, container_type* container, size_t index) : node(node), container(container), index(index) {}

    __device__ unordered_map_const_iterator(const self_type& other) : node(other.node), container(other.container), index(other.index) {}

    __device__ self_type& operator=(const self_type& other) { node = other.node; container = other.container; index = other.index; return *this; }

    __device__ reference operator*() const { return *node; }

    __device__ pointer   operator->() const { return &(operator*()); }

    __device__ self_type& operator++() 
    { 
        node_type nil = node_type(nullptr);

        if (++node == nil)
            while (node == nil && ++index < container->_buckets.size())
                node = container->_buckets[index].begin();

        return *this;
    }

    __device__ self_type operator++(int) 
    { 
        self_type temp = *this;
        ++*this;
        return temp;
    }

    __device__ bool operator==(const self_type& other) const { return node == other.node; }

    __device__ bool operator!=(const self_type& other) const { return node != other.node; }

};

template<typename Key, typename T, typename Hash = hash<Key>>
class unordered_map {
public:

    friend struct unordered_map_iterator<Key, T, Hash>;
    friend struct unordered_map_const_iterator<Key, T, Hash>;

    typedef unordered_map_iterator<Key, T, Hash>        iterator;
    typedef unordered_map_const_iterator<Key, T, Hash>  const_iterator;
    typedef unordered_map<Key, T, Hash>                 self_type;

    __device__ unordered_map(const Hash& hash = Hash());

    __device__ unordered_map(size_t bucket_count, const Hash& hash = Hash());

    __device__ unordered_map(size_t bucket_count, ParallalAllocator* allocator, const Hash& hash = Hash());

    __device__ unordered_map(const self_type& other);

    __device__ unordered_map(self_type&& other);

    __device__ iterator find(const Key& key);

    __device__ const_iterator find(const Key& key) const;

    __device__ pair<iterator, bool> insert(const Key& key, const T& value);

    __device__ T& at(const Key& key);

    __device__ T& operator[](const Key& key);

    __device__ iterator begin();

    __device__ const_iterator begin() const;

    __device__ const_iterator cbegin() const;

    __device__ iterator end();

    __device__ const_iterator end() const;

    __device__ const_iterator cend() const;

    __device__ inline size_t hash(const Key& key) const { return _hash(key) % _buckets.size(); }

    __device__ static size_t memory_need(size_t bucket_count, size_t size);


private:
    vector<list<pair<Key, T>>> _buckets;
    size_t _size;

    Hash _hash;
};

}

#include "unordered_map.cu"

#endif