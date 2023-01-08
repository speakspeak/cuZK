#ifndef __EVALUATION_DOMAIN_CUH__
#define __EVALUATION_DOMAIN_CUH__

#include "../../depends/libstl-cuda/vector.cuh"

namespace libfqfft {

template<typename FieldT>
class evaluation_domain {
public:
    const size_t m;

    FieldT instance;

    /**
     * Construct an evaluation domain S of size m, if possible.
     *
     * (See the function get_evaluation_domain below.)
     */
    __device__ evaluation_domain(const size_t m, const FieldT& instance) : m(m), instance(instance) {};

    /**
     * Get the idx-th element in S.
     */
    __device__ virtual FieldT get_domain_element(const size_t idx, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;

    /**
     * Compute the FFT, over the domain S, of the vector a.
     */
    __device__ virtual void FFT(libstl::vector<FieldT>& a, void* dummy = nullptr) = 0;

    /**
     * Compute the inverse FFT, over the domain S, of the vector a.
     */
    __device__ virtual void iFFT(libstl::vector<FieldT>& a, void* dummy = nullptr) = 0;

    /**
     * Compute the FFT, over the domain g*S, of the vector a.
     */
    __device__ virtual void cosetFFT(libstl::vector<FieldT>& a, const FieldT &g, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;

    /**
     * Compute the inverse FFT, over the domain g*S, of the vector a.
     */
    __device__ virtual void icosetFFT(libstl::vector<FieldT>& a, const FieldT &g, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;

     /**
     * Evaluate all Lagrange polynomials.
     *
     * The inputs are:
     * - an integer m
     * - an element t
     * The output is a vector (b_{0},...,b_{m-1})
     * where b_{i} is the evaluation of L_{i,S}(z) at z = t.
     */
    __device__ virtual libstl::vector<FieldT> evaluate_all_lagrange_polynomials(const FieldT &t, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;


    __device__ virtual void evaluate_all_lagrange_polynomials_multi(libstl::vector<FieldT>& u, const FieldT &t, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;

    /**
     * Evaluate the vanishing polynomial of S at the field element t.
     */
    __device__ virtual FieldT compute_vanishing_polynomial(const FieldT &t, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;

    /**
     * Add the coefficients of the vanishing polynomial of S to the coefficients of the polynomial H.
     */
    __device__ virtual void add_poly_Z(const FieldT &coeff, libstl::vector<FieldT>& H, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;

    /**
     * Multiply by the evaluation, on a coset of S, of the inverse of the vanishing polynomial of S.
     */
    __device__ virtual void divide_by_Z_on_coset(libstl::vector<FieldT>& P, void* dummy1 = nullptr, void* dummy2 = nullptr) = 0;
};

}
#endif