#ifndef __STL_CUDA_MEMORY_CU__
#define __STL_CUDA_MEMORY_CU__

#include <stdio.h>
#include "memory.cuh"

#define ALIGNED sizeof(double)

namespace libstl {



__device__ SerialAllocator::SerialAllocator(size_t size) : _begMem((unsigned char*)operator new(size)), _curMem(_begMem), _endMem(_begMem + size)
{
    _endMem = (unsigned char*)((size_t)_endMem & (size_t)~(ALIGNED - 1));
}

__device__ SerialAllocator::SerialAllocator(void* mem, size_t size) : _begMem((unsigned char*)mem), _curMem(_begMem), _endMem(_begMem + size)
{
    _begMem = _curMem = (unsigned char*)(((size_t)_begMem + ALIGNED - 1) & (size_t)~(ALIGNED - 1));
    _endMem = (unsigned char*)((size_t)_endMem & (size_t)~(ALIGNED - 1));
}


__device__ void* SerialAllocator::allocate(size_t size)
{ 
    void* mem = _curMem;

    size = (size + ALIGNED - 1) & ~(ALIGNED - 1);
    _curMem += size;

    if (_curMem > _endMem)
        printf("Memory allocation error\n");

    return mem;
}

__host__ void* SerialAllocator::allocate_host(size_t size)
{
    void* mem;
    cudaMemcpy(&mem, &this->_curMem, sizeof(void*), cudaMemcpyDeviceToHost);
    size = (size + ALIGNED - 1) & ~(ALIGNED - 1);

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            _curMem += size;
            if (_curMem > _endMem)
                printf("Memory allocation error\n");
        }
    );
    cudaStreamSynchronize(0);

    return mem;
}

__device__ ParallalAllocator* SerialAllocator::allocate(size_t gridSize, size_t blockSize, size_t size)
{
    size_t num = gridSize * blockSize;

    SerialAllocator* allocators = (SerialAllocator*)allocate(num * sizeof(SerialAllocator));
    unsigned char* mem = (unsigned char*)allocate(num * size);

    for (size_t i = 0; i < num; ++i)
        construct(&allocators[i], (void*)(mem + i * size), size);

    ParallalAllocator* pAllocator = (ParallalAllocator*)allocate(sizeof(ParallalAllocator));
    construct(pAllocator, allocators);

    return pAllocator;
}

__device__ void SerialAllocator::lock(size_t& lockMem)
{
    lockMem = (size_t)_curMem;
}

__host__ void SerialAllocator::lock_host(size_t& lockMem)
{
    cudaMemcpy(&lockMem, &_curMem, sizeof(size_t), cudaMemcpyDeviceToHost);
}

__device__ void SerialAllocator::reset(size_t size)
{
    size = (size + ALIGNED - 1) & ~(ALIGNED - 1);
    _curMem -= size;
}

__host__ void SerialAllocator::reset_host(size_t size)
{
    size = (size + ALIGNED - 1) & ~(ALIGNED - 1);
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            _curMem -= size;
        }
    );
    cudaStreamSynchronize(0);
}

__device__ void SerialAllocator::reset()
{
    _curMem = _begMem;
}

__host__ void SerialAllocator::reset_host()
{
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            _curMem = _begMem;
        }
    );
    cudaStreamSynchronize(0);
}


__device__ void SerialAllocator::resetlock(size_t lockMem)
{
    _curMem = (unsigned char*)lockMem;
}

__host__ void SerialAllocator::resetlock_host(size_t lockMem)
{
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            _curMem = (unsigned char*)lockMem;
        }
    );
    cudaStreamSynchronize(0);
}

__device__ void SerialAllocator::presetlock(size_t lockMem, size_t gridSize, size_t blockSize)
{
    unsigned char* _lock = (unsigned char*)lockMem;
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < this->_curMem - _lock)
            {
                _lock[idx] = 0;
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    _curMem = _lock;
}

__host__ void SerialAllocator::presetlock_host(size_t lockMem, size_t gridSize, size_t blockSize)
{

    unsigned char* lock = (unsigned char*)lockMem;
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx = tid;
            while(idx < this->_curMem - lock)
            {
                lock[idx] = 0;
                idx += gridDim.x * blockDim.x;
            }

        }
    );
    cudaStreamSynchronize(0);

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            this->_curMem = lock;
        }
    );
    cudaStreamSynchronize(0);
}


__device__ void SerialAllocator::print()
{
    printf("total: %p  ", _endMem - _begMem);
    printf("used: %p  ", _curMem - _begMem);
    printf("leave:  %p\n", _endMem - _curMem);
}

__host__ void SerialAllocator::print_host()
{
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            printf("start: %p ", _begMem);
            printf("end: %p ", _endMem);
            printf("total: %p  ", _endMem - _begMem);
            printf("used: %p  ", _curMem - _begMem);
            printf("leave:  %p\n", _endMem - _curMem);
        }
    );
    cudaStreamSynchronize(0);
}

__device__ ParallalAllocator::ParallalAllocator(SerialAllocator* allocators) : _allocators(allocators)
{
}
    
__device__ void* ParallalAllocator::allocate(size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    return _allocators[idx].allocate(size);
 }

__device__ void ParallalAllocator::reset(size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    _allocators[idx].reset(size);
}

__device__ void ParallalAllocator::reset()
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    _allocators[idx].reset();
}

__host__ void* AllocatorManager::allocate_host(size_t size)
{
    int select;
    cudaMemcpy(&select, &_select, sizeof(int), cudaMemcpyDeviceToHost);
    SerialAllocator* sAllocator;
    cudaMemcpy(&sAllocator, &_sAllocator, sizeof(SerialAllocator*), cudaMemcpyDeviceToHost);
    return select == 0 ? sAllocator->allocate_host(size) : nullptr;
}

__device__ SerialAllocator* mainAllocator;


__host__ void initAllocator(void* mem, size_t size)
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMalloc((void**)&hmainAllocator, sizeof(SerialAllocator));
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            new ((void*)hmainAllocator) SerialAllocator(mem, size);
        }
    );
    cudaStreamSynchronize(0);
    cudaMemcpyToSymbol(mainAllocator, &hmainAllocator, sizeof(SerialAllocator*));
}

__host__ void printAllocator_host()
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    return hmainAllocator->print_host();

}

__host__ void* allocate_host(size_t size) 
{ 
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    return hmainAllocator->allocate_host(size);
}

__host__  void lock_host(size_t& lockMem)
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    hmainAllocator->lock_host(lockMem);
}

__host__ void reset_host(size_t size)
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    hmainAllocator->reset_host(size);
}

__host__ void reset_host()
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    hmainAllocator->reset_host();
}

__host__ void resetlock_host(size_t lockMem)
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    hmainAllocator->resetlock_host(lockMem);
}

__host__ void presetlock_host(size_t lockMem, size_t gridSize, size_t blockSize)
{
    SerialAllocator* hmainAllocator = nullptr;
    cudaMemcpyFromSymbol(&hmainAllocator, mainAllocator, sizeof(SerialAllocator*));
    hmainAllocator->presetlock_host(lockMem, gridSize, blockSize);
}

}

#endif
