#ifndef __GEOMETRIC_SEQUENCE_DOMAIN_CUH__
#define __GEOMETRIC_SEQUENCE_DOMAIN_CUH__

namespace libfqfft {

template<typename FieldT>
class geometric_sequence_domain {
public:
    size_t m;
    const FieldT& instance;

    bool precomputation_sentinel;

    libstl::vector<FieldT> geometric_sequence;

    libstl::vector<FieldT> geometric_triangular_sequence;

    __device__ void do_precomputation();

    __device__ geometric_sequence_domain(const size_t m, const FieldT& instance);

    __device__ void FFT(libstl::vector<FieldT>& a);
    __device__ void pFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize);

    __device__ void iFFT(libstl::vector<FieldT>& a);
    __device__ void piFFT(libstl::vector<FieldT>& a, size_t gridSize, size_t blockSize);

    __device__ void cosetFFT(libstl::vector<FieldT>& a, const FieldT& g);
    __device__ void pcosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize);

    __device__ void icosetFFT(libstl::vector<FieldT>& a, const FieldT& g);
    __device__ void picosetFFT(libstl::vector<FieldT>& a, const FieldT& g, size_t gridSize, size_t blockSize);

    __device__ libstl::vector<FieldT> evaluate_all_lagrange_polynomials(const FieldT &t);

    __device__ FieldT get_domain_element(const size_t idx);

    __device__ FieldT compute_vanishing_polynomial(const FieldT &t);

    __device__ void add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H);

    __device__ void divide_by_Z_on_coset(libstl::vector<FieldT>& P);
    
};

} // libfqfft

#include "geometric_sequence_domain.cu"

#endif