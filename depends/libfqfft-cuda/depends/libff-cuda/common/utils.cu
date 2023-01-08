#include "utils.cuh"

namespace libff {

__host__ __device__ size_t log2(size_t n)
/* returns ceil(log2(n)), so 1ul<<log2(n) is the smallest power of 2,
   that is not less than n. */
{
    size_t r = ((n & (n-1)) == 0 ? 0 : 1); // add 1 if n is not power of 2

    while (n > 1)
    {
        n >>= 1;
        r++;
    }

    return r;
}



__device__ size_t bitreverse(size_t n, const size_t l)
{
    size_t r = 0;
    for (size_t k = 0; k < l; ++k)
    {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}


}