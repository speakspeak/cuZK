#ifndef __BIGINT_HOST_CUH__
#define __BIGINT_HOST_CUH__

#include <gmp.h>

namespace libff {

template<mp_size_t n>
class bigint_host {
public:
    mp_limb_t data[n] = {0};
    
    bigint_host() = default;
    bigint_host(const unsigned long x); 
    bigint_host(const char* s); 
    bigint_host(const mpz_t r); 

    bigint_host(const bigint_host<n>& other) = default;
    bigint_host<n>& operator=(const bigint_host<n>& other) = default;

    bool operator==(const bigint_host<n>& other) const;
    bool operator!=(const bigint_host<n>& other) const;
    bool operator>=(const bigint_host<n>& other) const;

    bigint_host<n>& operator+=(const bigint_host<n>& other);
    bigint_host<n>& operator-=(const bigint_host<n>& other);

    void clear();
    bool is_zero() const;
    size_t max_bits() const { return n * 8 * sizeof(mp_limb_t); } 
    size_t num_bits() const; 

    unsigned long as_ulong() const;
    void to_mpz(mpz_t r) const;
    bool test_bit(const std::size_t bitno) const;
    void set_bit(const std::size_t bitno);

    bigint_host<n>& randomize();

    void print() const;

};

}

#include "bigint_host.cu"

#endif
