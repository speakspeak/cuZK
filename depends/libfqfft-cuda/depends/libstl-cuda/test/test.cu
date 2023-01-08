#include "../memory.cuh"
#include "../vector.cuh"
#include "../list.cuh"
#include "../algorithm.cuh"
#include "../functional.cuh"
#include "../unordered_map.cuh"

#include <stdio.h>

using namespace libstl;

__device__ ParallalAllocator* pAllocator;

__global__ void init(void)
{
    initAllocator(100000);
    
    size_t bucket_count = 5;
    size_t size = 10;

    size_t mem = unordered_map<int, int>::memory_need(bucket_count, size);

    pAllocator = allocate(2, 2, mem);
}

__global__ void kernel(void)
{
    size_t bucket_count = 5;
    size_t size = 10;

    unordered_map<int, int> a(bucket_count, pAllocator);

    for (int i = 0; i < size; ++i) 
    {
        auto res = a.insert(0, i + 1);

        if (res.second == true)
            printf("Insert success, no replication\n");
        else
            printf("Replication\n");

        int key = res.first->first;
        int value = res.first->second;  // if replicated, older (k,v) can be accessed from this
    }



    for (auto it = a.begin(); it != a.end(); ++it)
        printf("bid %lu tid %lu index %lu key %d value %d\n", (size_t)blockIdx.x, (size_t)threadIdx.x, it.index, it->first, it->second);

}

int main()
{
    init<<<1, 1>>>();
    kernel<<<2, 2>>>();
    cudaDeviceReset();
}
