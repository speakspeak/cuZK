#ifndef __WNAF_CU__
#define __WNAF_CU__

#include "../mini-mp-cuda/mini-mp-cuda.cuh"

namespace libff{

template<mp_size_t_ n>
__device__ void find_wnaf(long** wnaf, size_t* wnaf_size, const size_t window_size, const bigint<n> &scalar)
{
    const size_t length = scalar.max_bits(); // upper bound
    *wnaf = new long[length+1];
    *wnaf_size = length + 1;
    // std::vector<long> res(length+1);
    bigint<n> c = scalar;
    long j = 0;
    while (!c.is_zero())
    {
        long u;
        if ((c.data[0] & 1) == 1)
        {
            u = c.data[0] % (1u << (window_size+1));
            if (u > (1 << window_size))
            {
                u = u - (1 << (window_size+1));
            }

            if (u > 0)
            {
                mpn_sub__1_(c.data, c.data, n, u);
            }
            else
            {
                mpn_add__1_(c.data, c.data, n, -u);
            }
        }
        else
        {
            u = 0;
        }
        (*wnaf)[j] = u;
        ++j;

        mpn_rshift_(c.data, c.data, n, 1); // c = c/2
    }
}



template<typename T, mp_size_t_ n>
__device__ T fixed_window_wnaf_exp(const size_t window_size, const T &base, const bigint<n> &scalar)
{
    const T& instance = base;
    long * naf;
    size_t naf_size;
    find_wnaf(&naf, &naf_size, window_size, scalar);

    size_t table_size = 1ul<<(window_size-1);
    void *raw_memory_table = operator new[](table_size * sizeof(T));
    T* table = (T*) raw_memory_table;
    for(int i=0; i<table_size; i++)
    {
        new(&table[i]) T(instance);
    }

    // std::vector<T> table(1ul<<(window_size-1));
    T tmp = base;
    T dbl = base.dbl();
    for (size_t i = 0; i < 1ul<<(window_size-1); ++i)
    {
        table[i] = tmp;
        tmp = tmp + dbl;
    }

    T res = instance.zero();
    bool found_nonzero = false;
    for (long i = naf_size-1; i >= 0; --i)
    {
        if (found_nonzero)
        {
            res = res.dbl();
        }

        if (naf[i] != 0)
        {
            found_nonzero = true;
            if (naf[i] > 0)
            {
                res = res + table[naf[i]/2];
            }
            else
            {
                res = res - table[(-naf[i])/2];
            }
        }
    }

    for (int i = table_size - 1; i >= 0; --i) {
        table[i].~T();
    }
    operator delete[]((void*)table);
    
    delete[] naf;

    return res;
}


template<typename T, mp_size_t_ n>
__device__ T opt_window_wnaf_exp(const T &base, const bigint<n> &scalar, const size_t scalar_bits)
{
    size_t best = 0;
    const T& instance = base;
    for (long i = instance.params->wnaf_window_table_size - 1; i >= 0; --i)
    {
        if (scalar_bits >= instance.params->wnaf_window_table[i])
        {
            best = i+1;
            break;
        }
    }

    if (best > 0)
    {
        return fixed_window_wnaf_exp(best, base, scalar);
    }
    else
    {
        return scalar * base;
    }
}

}

#endif