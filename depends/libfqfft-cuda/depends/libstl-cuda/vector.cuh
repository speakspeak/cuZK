#ifndef __STL_CUDA_VECTOR_CUH__
#define __STL_CUDA_VECTOR_CUH__

#include "memory.cuh"
#include <vector>

namespace libstl {

template<typename T>
class vector {
public:

    typedef T*         iterator;
    typedef const T*   const_iterator;
    typedef vector<T>  self_type;

    __host__ __device__ vector();

    __device__ vector(size_t n);

    __device__ vector(size_t n, size_t gridSize, size_t blockSize);

    __device__ vector(size_t n, const T& val);

    __device__ vector(size_t n, const T& val, size_t gridSize, size_t blockSize);

    __device__ vector(const self_type& x);

    __device__ vector(const self_type& x, size_t gridSize, size_t blockSize);

    __device__ vector(self_type&& x);

    __device__ vector(const_iterator first, const_iterator last);

    __device__ vector(const_iterator first, const_iterator last, size_t gridSize, size_t blockSize);

    __host__ __device__ ~vector();

    __device__ self_type& operator=(const self_type& x);

    __device__ self_type& operator=(self_type&& x);

    __device__ void pcopy(const self_type& x, size_t gridSize, size_t blockSize);

    __host__ void pcopy_host(self_type& x, size_t gridSize, size_t blockSize);

    __host__ __device__ T& operator[](size_t n);

    __host__ __device__ const T& operator[](size_t n) const;

    __device__ T& front();

    __device__ const T& front() const;

    __device__ T& back() ;

    __device__ const T& back() const;

    __device__ iterator begin();

    __device__ const_iterator begin() const;

    __device__ const_iterator cbegin() const;

    __device__ iterator end();

    __device__ const_iterator end() const;

    __device__ const_iterator cend() const;

    __device__ size_t size() const;

    __host__ size_t size_host();

    __device__ void resize(size_t n);

    __host__ void resize_host(size_t n);

    __device__ void presize(size_t n, size_t gridSize, size_t blockSize);

    __host__ void presize_host(size_t n, size_t gridSize, size_t blockSize);

    __device__ void resize(size_t n, const T& val);

    __device__ void presize(size_t n, const T& val, size_t gridSize, size_t blockSize);

    __host__ void presize_host(size_t n, const T* val, size_t gridSize, size_t blockSize);

    __device__ inline void set_serial_allocator(SerialAllocator* allocator = mainAllocator) { _allocManager.setSerialAllocator(allocator); }

    __device__ inline void set_parallel_allocator(ParallalAllocator* allocator) { _allocManager.setParallelAllocator(allocator); }

    __device__ static size_t memory_need(size_t n);


public:
    T* _data;
    size_t _size;

    AllocatorManager _allocManager;
};

template<typename T, typename H>
__host__ void vector_device2host(libstl::vector<H>* hv, const libstl::vector<T>* dv, cudaStream_t stream = 0);

template<typename T, typename H>
__host__ void vector_host2device(libstl::vector<T>* dv, const libstl::vector<H>* hv, cudaStream_t stream = 0);

template<typename T>
__host__ void vector_device2host(libstl::vector<T>* hv, const libstl::vector<T>* dv, cudaStream_t stream = 0);

template<typename T>
__host__ void vector_host2device(libstl::vector<T>* dv, const libstl::vector<T>* hv, cudaStream_t stream = 0);

}

#include "vector.cu"

#endif