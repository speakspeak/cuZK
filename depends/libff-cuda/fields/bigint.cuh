#ifndef __BIGINT_CUH__
#define __BIGINT_CUH__

#include "../mini-mp-cuda/mini-mp-cuda.cuh"

namespace libff {

template<mp_size_t_ n>
class bigint {
public:
    mp_limb_t_ data[n];
    
    __device__ bigint() = default;
    __device__ bigint(const unsigned long x); 
    __device__ bigint(const char* s); 
    __device__ bigint(const mpz_t_ r); 

    __device__ bigint(const bigint<n>& other) = default;
    __device__ bigint<n>& operator=(const bigint<n>& other) = default;

    __device__ bool operator==(const bigint<n>& other) const;
    __device__ bool operator!=(const bigint<n>& other) const;
    __device__ bool operator>=(const bigint<n>& other) const;

    __device__ bigint<n>& operator+=(const bigint<n>& other);
    __device__ bigint<n>& operator-=(const bigint<n>& other);

    __device__ void clear();
    __device__ bool is_zero() const;
    __device__ size_t max_bits() const { return n * 8 * sizeof(mp_limb_t_); } 
    __device__ size_t num_bits() const; 

    __device__ unsigned long as_ulong() const;
    __device__ void to_mpz(mpz_t_ r) const;
    __device__ bool test_bit(const std::size_t bitno) const;
    __device__ void set_bit(const std::size_t bitno);

    __device__ bigint<n>& randomize();

    __device__ void print() const;

};

}

#include "bigint.cu"

#endif
