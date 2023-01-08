#ifndef __STL_CUDA_UNORDERED_MAP_CU__
#define __STL_CUDA_UNORDERED_MAP_CU__

namespace libstl {

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::unordered_map(const Hash& hash) : _size(0), _hash(hash)
{

}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::unordered_map(size_t bucket_count, const Hash& hash) : _buckets(bucket_count), _size(0), _hash(hash) 
{

}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::unordered_map(size_t bucket_count, ParallalAllocator* allocator, const Hash& hash) : _size(0), _hash(hash)
{
    _buckets.set_parallel_allocator(allocator);
    _buckets.resize(bucket_count);

    for (size_t index = 0; index < _buckets.size(); ++index)
        _buckets[index].set_parallel_allocator(allocator);
}


template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::unordered_map(const self_type& other) : _buckets(other._buckets), _size(other._size), _hash(other._hash)
{

}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::unordered_map(self_type&& other) : _buckets(move(other._buckets)), _size(other._size), _hash(other._hash)
{

}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::iterator unordered_map<Key, T, Hash>::find(const Key& key)
{
    size_t index = hash(key);

    for (auto it = _buckets[index].begin(); it != _buckets[index].end(); ++it)
        if (it->first == key)
            return iterator(it, this, index);

    return end();
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::const_iterator unordered_map<Key, T, Hash>::find(const Key& key) const
{
    size_t index = hash(key);

    for (auto it = _buckets[index].cbegin(); it != _buckets[index].cend(); ++it)
        if (it->first == key)
            return const_iterator(it, this, index);

    return cend();   
}

template<typename Key, typename T, typename Hash>
__device__ pair<typename unordered_map<Key, T, Hash>::iterator, bool> unordered_map<Key, T, Hash>::insert(const Key& key, const T& value)
{
    size_t index = hash(key);

    for (auto it = _buckets[index].begin(); it != _buckets[index].end(); ++it)
        if (it->first == key)
            return make_pair<iterator, bool>(iterator(it, this, index), false);
    
    _buckets[index].emplace_front(make_pair<Key, T>(key, value));
    ++_size;

    return make_pair<iterator, bool>(iterator(_buckets[index].begin(), this, index), true);
}

template<typename Key, typename T, typename Hash>
__device__ T& unordered_map<Key, T, Hash>::at(const Key& key)
{
    iterator it = find(key);
    assert(it != end());

    return it->second;
}

template<typename Key, typename T, typename Hash>
__device__ T& unordered_map<Key, T, Hash>::operator[](const Key& key)
{
    iterator it = find(key);
    return it->second;
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::iterator unordered_map<Key, T, Hash>::begin()
{
    for (size_t index = 0; index < _buckets.size(); ++index)
        if (_buckets[index].begin() != _buckets[index].end())
            return iterator(_buckets[index].begin(), this, index);

    return end();
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::const_iterator unordered_map<Key, T, Hash>::begin() const
{
    for (size_t index = 0; index < _buckets.size(); ++index)
        if (_buckets[index].cbegin() != _buckets[index].cend())
            return const_iterator(_buckets[index].cbegin(), this, index);

    return cend();
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::const_iterator unordered_map<Key, T, Hash>::cbegin() const
{
    for (size_t index = 0; index < _buckets.size(); ++index)
        if (_buckets[index].cbegin() != _buckets[index].cend())
            return const_iterator(_buckets[index].cbegin(), this, index);

    return cend();
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::iterator unordered_map<Key, T, Hash>::end()
{
    return iterator(_buckets[0].end(), this, 0);
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::const_iterator unordered_map<Key, T, Hash>::end() const
{
    return const_iterator(_buckets[0].cend(), this, 0);
}

template<typename Key, typename T, typename Hash>
__device__ unordered_map<Key, T, Hash>::const_iterator unordered_map<Key, T, Hash>::cend() const
{
    return const_iterator(_buckets[0].cend(), this, 0);
}

template<typename Key, typename T, typename Hash>
__device__ size_t unordered_map<Key, T, Hash>::memory_need(size_t bucket_count, size_t size)
{
    return vector<list<pair<Key, T>>>::memory_need(bucket_count) + list<pair<Key, T>>::memory_need(size);
}

}

#endif