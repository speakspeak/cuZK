#ifndef __STL_CUDA_MEMORY_CUH__
#define __STL_CUDA_MEMORY_CUH__

#include "utility.cuh"
#include "iterator.cuh"
#include <new>

namespace libstl {

class ParallalAllocator;

class SerialAllocator  {
public:
    __device__ SerialAllocator(size_t size);

    __device__ SerialAllocator(void* mem, size_t size);
    
    __device__ void* allocate(size_t size);

    __host__ void* allocate_host(size_t size);

    __device__ ParallalAllocator* allocate(size_t gridSize, size_t blockSize, size_t size);

    __device__ void lock(size_t& lockMem);

    __host__ void lock_host(size_t& lockMem);

    __device__ void reset(size_t size);

    __host__ void reset_host(size_t size);

    __device__ void reset();

    __host__ void reset_host();

    __device__ void resetlock(size_t lockMem);

    __host__ void resetlock_host(size_t lockMem);

    __device__ void presetlock(size_t lockMem, size_t gridSize, size_t blockSize);

    __host__ void presetlock_host(size_t lockMem, size_t gridSize, size_t blockSize);

    __device__ void print();

    __host__ void print_host();

private:
    unsigned char* _begMem;
    unsigned char* _curMem;
    unsigned char* _endMem;
};

class ParallalAllocator {
public:
    __device__ ParallalAllocator(SerialAllocator* allocators);
    
    __device__ void* allocate(size_t size);

    __device__ void reset(size_t size);

    __device__ void reset();

private:
    SerialAllocator* _allocators;
};

__device__ extern SerialAllocator* mainAllocator;

// __device__ inline void initAllocator(void* mem, size_t size) { mainAllocator = new SerialAllocator(mem, size); }

__host__ void initAllocator(void* mem, size_t size);

__device__ inline void printAllocator() {mainAllocator->print();}

__host__ void printAllocator_host();

__device__ inline void* allocate(size_t size) { return mainAllocator->allocate(size); }

__host__ void* allocate_host(size_t size);

__device__ inline ParallalAllocator* allocate(size_t gridSize, size_t blockSize, size_t size) { return mainAllocator->allocate(gridSize, blockSize, size); }

__device__ inline void lock(size_t& lockMem) { return mainAllocator->lock(lockMem); }

__host__  void lock_host(size_t& lockMem);

__device__ inline void reset(size_t size) { return mainAllocator->reset(size); }

__host__ void reset_host(size_t size);

__device__ inline void reset() { return mainAllocator->reset(); }

__host__ void reset_host();

__device__ inline void resetlock(size_t lockMem) { return mainAllocator->resetlock(lockMem); }

__host__  void resetlock_host(size_t lockMem);

__device__ inline void presetlock(size_t lockMem, size_t gridSize, size_t blockSize) { return mainAllocator->presetlock(lockMem, gridSize, blockSize); }

__host__  void presetlock_host(size_t lockMem, size_t gridSize, size_t blockSize);

class AllocatorManager {
public:
    const int SERIAL = 0;
    const int PARALLEL = 1;

    __device__ AllocatorManager() : _select(SERIAL), _sAllocator(mainAllocator), _pAllocator(nullptr) {}

    __device__ inline void setSerialAllocator(SerialAllocator* allocator = mainAllocator) { _select = SERIAL; _sAllocator = allocator; }

    __device__ inline void setParallelAllocator(ParallalAllocator* allocator) { _select = PARALLEL; _pAllocator = allocator; }

    __device__ inline void* allocate(size_t size) { return _select == SERIAL ? _sAllocator->allocate(size) : _pAllocator->allocate(size); }

    __host__ void* allocate_host(size_t size);


private:
    int _select;

    SerialAllocator* _sAllocator;
    ParallalAllocator* _pAllocator;
};


template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator p_uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result, size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t n = last - first;
            while(idx < n)
            {
                new ((void*)&*(result+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(*(first+idx));
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
    return result;
}


template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result)
{
    for (; first != last; ++result, ++first)
        new ((void*)&*result) 
            typename iterator_traits<ForwardIterator>::value_type(*first);

    return result;
}

template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator p_uninitialized_copy_n(InputIterator first, size_t n, ForwardIterator result, size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < n)
            {
                new ((void*)&*(result+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(*(first+idx));
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return result;
}



template<typename InputIterator, typename ForwardIterator>
__host__ ForwardIterator p_uninitialized_copy_n_host(InputIterator first, size_t n, ForwardIterator result, size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < n)
            {
                new ((void*)&*(result+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(*(first+idx));
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);

    return result;
}



template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator uninitialized_copy_n(InputIterator first, size_t n, ForwardIterator result)
{
    for (; n > 0; ++result, ++first, --n)
        new ((void*)&*result) 
            typename iterator_traits<ForwardIterator>::value_type(*first);

    return result;
}


template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator p_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t n = last - first;
            while(idx < n)
            {
                new ((void*)&*(result+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(move(*(first+idx)));
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return result;
}



template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result)
{
    for (; first != last; ++result, ++first)
        new ((void*)&*result) 
            typename iterator_traits<ForwardIterator>::value_type(move(*first));

    return result;
}


template<typename InputIterator, typename ForwardIterator>
__device__ ForwardIterator p_uninitialized_move_n(InputIterator first, size_t n, ForwardIterator result, size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < n)
            {
                new ((void*)&*(result+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(move(*(first+idx)));
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    return result;
}


template<typename InputIterator, typename ForwardIterator>
__host__ ForwardIterator p_uninitialized_move_n_host(InputIterator first, size_t n, ForwardIterator result, size_t gridSize, size_t blockSize)
{
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < n)
            {
                new ((void*)&*(result+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(move(*(first+idx)));
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);

    return result;
}


template<typename InputIterator, typename ForwardIterator>
__host__ __device__ ForwardIterator uninitialized_move_n(InputIterator first, size_t n, ForwardIterator result)
{
    for (; n > 0; ++result, ++first, --n)
        new ((void*)&*result)
            typename iterator_traits<ForwardIterator>::value_type(move(*first));

    return result;
}


template <typename ForwardIterator, typename T>
__device__ void p_uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x, size_t gridSize, size_t blockSize)
{

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &x]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t n = last - first;
            while(idx < n)
            {
                new ((void*)&*(first+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(x);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
}



template <typename ForwardIterator, typename T>
__device__ void uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x)
{
    for (; first != last; ++first)
        new ((void*)&*first) 
            typename iterator_traits<ForwardIterator>::value_type(x);
}


template <typename ForwardIterator, typename T>
__device__ void p_uninitialized_fill_n(ForwardIterator first, size_t n, const T& x, size_t gridSize, size_t blockSize)
{

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &x]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < n)
            {
                new ((void*)&*(first+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(x);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
}

template <typename ForwardIterator, typename T>
__host__ void p_uninitialized_fill_n_host(ForwardIterator first, size_t n, T* x, size_t gridSize, size_t blockSize)
{ 
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < n)
            {
                new ((void*)&*(first+idx)) 
                    typename iterator_traits<ForwardIterator>::value_type(*x);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);
}


template <typename ForwardIterator, typename T>
__host__ __device__ void uninitialized_fill_n(ForwardIterator first, size_t n, const T& x)
{
    for (; n--; ++first)
        new ((void*)&*first) 
            typename iterator_traits<ForwardIterator>::value_type(x);
}

template <typename T, typename... Args>
__device__ inline void construct(T* ptr, Args&&... args)
{
    new ((void*)ptr) T(forward<Args>(args)...);    
}


template <typename T, typename... Args>
__host__ void construct_host(T* ptr, Args&&... args)
{
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (Args&&... args)
        {
            new ((void*)ptr) T(forward<Args>(args)...);
        }
        , args...
    );
    cudaStreamSynchronize(0);
}


template<typename T>
__host__ void get_host(T* dst, T* src)
{
    cudaMemcpy(dst, src, sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
__host__ void send_host(T* dst, T* src)
{
    cudaMemcpy(dst, src, sizeof(T), cudaMemcpyHostToDevice);
}


template<typename T, typename... Args>
__device__ inline T* create(Args&&... args)
{
    T* obj = (T*)allocate(sizeof(T));
    construct(obj, forward<Args>(args)...);
    return obj;
}

template<typename T, typename... Args>
__host__ T* create_host(Args&&... args)
{
    T* obj = (T*)allocate_host(sizeof(T));
    construct_host(obj, args...);
    return obj;
}





}

#endif