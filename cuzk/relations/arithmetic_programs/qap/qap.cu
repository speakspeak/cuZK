#ifndef __QAP_CU__
#define __QAP_CU__



namespace cuzk
{

template<typename FieldT>
__device__ qap_instance<FieldT>::qap_instance(libfqfft::basic_radix2_domain<FieldT>* domain,
                 const size_t num_variables,
                 const size_t degree,
                 const size_t num_inputs,
                 libstl::vector<libstl::vector<FieldT>> vA_in_Lagrange_basis,
                 libstl::vector<libstl::vector<FieldT>> vB_in_Lagrange_basis,
                 libstl::vector<libstl::vector<FieldT>> vC_in_Lagrange_basis
                 ):
num_variables_(num_variables),
degree_(degree),
num_inputs_(num_inputs),
domain(domain),
vA_in_Lagrange_basis(vA_in_Lagrange_basis),
vB_in_Lagrange_basis(vB_in_Lagrange_basis),
vC_in_Lagrange_basis(vC_in_Lagrange_basis)
{
}

template<typename FieldT>
__device__ size_t qap_instance<FieldT>::num_variables() const
{
    return num_variables_;
}

template<typename FieldT>
__device__ size_t qap_instance<FieldT>::degree() const
{
    return degree_;
}

template<typename FieldT>
__device__ size_t qap_instance<FieldT>::num_inputs() const
{
    return num_inputs_;
}


template<typename FieldT>
__device__ qap_instance_evaluation<FieldT>::qap_instance_evaluation(const size_t num_variables,
                            const size_t degree,
                            const size_t num_inputs,
                            const FieldT &t,
                            libstl::vector<FieldT> vAt,
                            libstl::vector<FieldT> vBt,
                            libstl::vector<FieldT> vCt,
                            libstl::vector<FieldT> vHt,
                            const FieldT &Zt):
    num_variables_(num_variables),
    degree_(degree),
    num_inputs_(num_inputs),
    t(t),
    vAt(vAt),
    vBt(vBt),
    vCt(vCt),
    vHt(vHt),
    Zt(Zt)
{
}


template<typename FieldT>
__device__ size_t qap_instance_evaluation<FieldT>::num_variables() const
{
    return num_variables_;
}

template<typename FieldT>
__device__ size_t qap_instance_evaluation<FieldT>::degree() const
{
    return degree_;
}

template<typename FieldT>
__device__ size_t qap_instance_evaluation<FieldT>::num_inputs() const
{
    return num_inputs_;
}



template<typename FieldT>
__device__ qap_witness<FieldT>::qap_witness(const size_t num_variables,
                const size_t degree,
                const size_t num_inputs,
                const FieldT &d1,
                const FieldT &d2,
                const FieldT &d3,
                libstl::vector<FieldT> vcoefficients_for_ABCs,
                libstl::vector<FieldT> vcoefficients_for_H):
    num_variables_(num_variables),
    degree_(degree),
    num_inputs_(num_inputs),
    d1(d1),
    d2(d2),
    d3(d3),
    vcoefficients_for_ABCs(vcoefficients_for_ABCs),
    vcoefficients_for_H(vcoefficients_for_H)
{
}



template<typename FieldT>
__device__ size_t qap_witness<FieldT>::num_variables() const
{
    return num_variables_;
}

template<typename FieldT>
__host__ size_t qap_witness<FieldT>::num_variables_host()
{
    size_t num_variables;
    cudaMemcpy(&num_variables, &this->num_variables_, sizeof(size_t), cudaMemcpyDeviceToHost);
    return num_variables;
}


template<typename FieldT>
__device__ size_t qap_witness<FieldT>::degree() const
{
    return degree_;
}

template<typename FieldT>
__host__ size_t qap_witness<FieldT>::degree_host()
{
    size_t degree;
    cudaMemcpy(&degree, &this->degree_, sizeof(size_t), cudaMemcpyDeviceToHost);
    return degree;
}

template<typename FieldT>
__device__ size_t qap_witness<FieldT>::num_inputs() const
{
    return num_inputs_;
}


template<typename FieldT>
__host__ size_t qap_witness<FieldT>::num_inputs_host()
{
    size_t num_inputs;
    cudaMemcpy(&num_inputs, &this->num_inputs_, sizeof(size_t), cudaMemcpyDeviceToHost);
    return num_inputs;
}







}


#endif