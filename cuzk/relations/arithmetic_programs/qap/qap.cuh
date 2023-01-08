#ifndef __QAP_CUH__
#define __QAP_CUH__


#include "../../../../depends/libfqfft-cuda/libfqfft/evaluation_domain/evaluation_domain.cuh"
#include "../../../../depends/libfqfft-cuda/libfqfft/evaluation_domain/get_evaluation_domain.cuh"
#include "../../../../depends/libstl-cuda/vector.cuh"

namespace cuzk
{


template<typename FieldT>
class qap_instance {
private:
    size_t num_variables_;
    size_t degree_;
    size_t num_inputs_;

public:
    // libfqfft::evaluation_domain<FieldT>* domain;
    libfqfft::basic_radix2_domain<FieldT>* domain;

    libstl::vector<libstl::vector<FieldT>> vA_in_Lagrange_basis;
    libstl::vector<libstl::vector<FieldT>> vB_in_Lagrange_basis;
    libstl::vector<libstl::vector<FieldT>> vC_in_Lagrange_basis;

    __device__ qap_instance(libfqfft::basic_radix2_domain<FieldT>* domain,
                 const size_t num_variables,
                 const size_t degree,
                 const size_t num_inputs,
                 libstl::vector<libstl::vector<FieldT>> vA_in_Lagrange_basis,
                 libstl::vector<libstl::vector<FieldT>> vB_in_Lagrange_basis,
                 libstl::vector<libstl::vector<FieldT>> vC_in_Lagrange_basis
                );

    __device__ qap_instance(const qap_instance<FieldT> &other) = default;
    __device__ qap_instance(qap_instance<FieldT> &&other) = default;
    __device__ qap_instance& operator=(const qap_instance<FieldT> &other) = default;
    __device__ qap_instance& operator=(qap_instance<FieldT> &&other) = default;

    __device__ size_t num_variables() const;
    __device__ size_t degree() const;
    __device__ size_t num_inputs() const;

    // bool is_satisfied(const qap_witness<FieldT> &witness) const;

};



template<typename FieldT>
class qap_instance_evaluation {
public:
    size_t num_variables_;
    size_t degree_;
    size_t num_inputs_;
public:
    // libfqfft::evaluation_domain<FieldT>* domain;
    // libfqfft::basic_radix2_domain<FieldT>* domain;

    FieldT t;

    libstl::vector<FieldT> vAt;
    libstl::vector<FieldT> vBt;
    libstl::vector<FieldT> vCt;
    libstl::vector<FieldT> vHt;

    FieldT Zt;

    __device__ qap_instance_evaluation(){};

    __device__ qap_instance_evaluation(const size_t num_variables,
                            const size_t degree,
                            const size_t num_inputs,
                            const FieldT &t,
                            libstl::vector<FieldT> vAt,
                            libstl::vector<FieldT> vBt,
                            libstl::vector<FieldT> vCt,
                            libstl::vector<FieldT> vHt,
                            const FieldT &Zt);

    __device__ qap_instance_evaluation(const qap_instance_evaluation<FieldT> &other) = default;
    __device__ qap_instance_evaluation(qap_instance_evaluation<FieldT> &&other) = default;
    __device__ qap_instance_evaluation& operator=(const qap_instance_evaluation<FieldT> &other) = default;
    __device__ qap_instance_evaluation& operator=(qap_instance_evaluation<FieldT> &&other) = default;

    __device__ size_t num_variables() const;
    __device__ size_t degree() const;
    __device__ size_t num_inputs() const;

    // bool is_satisfied(const qap_witness<FieldT> &witness) const;
};

 
template<typename FieldT>
class qap_witness {
public:
    size_t num_variables_;
    size_t degree_;
    size_t num_inputs_;

public:
    FieldT d1, d2, d3;

    libstl::vector<FieldT> vcoefficients_for_ABCs;
    libstl::vector<FieldT> vcoefficients_for_H;

    __device__ qap_witness(){};

    __device__ qap_witness(const size_t num_variables,
                const size_t degree,
                const size_t num_inputs,
                const FieldT &d1,
                const FieldT &d2,
                const FieldT &d3,
                libstl::vector<FieldT> vcoefficients_for_ABCs,
                libstl::vector<FieldT> vcoefficients_for_H);

    __device__ qap_witness(const qap_witness<FieldT> &other) = default;
    __device__ qap_witness(qap_witness<FieldT> &&other) = default;
    __device__ qap_witness& operator=(const qap_witness<FieldT> &other) = default;
    __device__ qap_witness& operator=(qap_witness<FieldT> &&other) = default;

    __device__ size_t num_variables() const;
    __host__ size_t num_variables_host();
    __device__ size_t degree() const;
    __host__ size_t degree_host();
    __device__ size_t num_inputs() const;
    __host__ size_t num_inputs_host();
};


}

#include "qap.cu"

#endif