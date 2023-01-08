#ifndef __GROTH_CU__
#define __GROTH_CU__


#include "../../reductions/r1cs_to_qap/r1cs_to_qap.cuh"

namespace cuzk{

template<typename ppT>
__global__ void groth_generate_random_element(generator_params<ppT> *gp, instance_params* ip)
{
    gp->t = ip->instance.random_element();
    gp->alpha = ip->instance.random_element();
    gp->beta = ip->instance.random_element();
    gp->gamma = ip->instance.random_element();
    gp->delta = ip->instance.random_element();
    gp->gamma_inverse = gp->gamma.inverse();
    gp->delta_inverse = gp->delta.inverse();

    gp->g1_generator = ip->g1_instance.random_element();
    gp->g2_generator = ip->g2_instance.random_element();
}

// template<typename ppT>
// __global__ void r1cs_test(generator_params<ppT> *gp, instance_params* ip)
// {
//     size_t non_zero_At = 0;
//     size_t non_zero_Bt = 0;
    
//     for (size_t i = 0; i < gp->qap.num_variables() + 1; ++i)
//     {
//         if (!gp->qap.vAt[i].is_zero())
//         {
//             ++non_zero_At;
//         }
//         if (!gp->qap.vBt[i].is_zero())
//         {
//             ++non_zero_Bt;
//         }
//     }
//     printf("non zero At: %d\n", non_zero_At);
// }


template<typename ppT>
__global__ void groth_gamma_Lt_Setup(generator_params<ppT> *gp, instance_params* ip)
{
    gp->gamma_ABC_0 = (gp->beta * gp->qap.vAt[0] + gp->alpha * gp->qap.vBt[0] + gp->qap.vCt[0]) * gp->gamma_inverse;
    size_t gamma_ABC_size = gp->qap.num_inputs();
    gp->gamma_ABC.resize(gamma_ABC_size, ip->instance.zero());

    size_t Lt_size = gp->qap.num_variables() - gp->qap.num_inputs();
    gp->Lt.resize(Lt_size, ip->instance.zero());

    gp->qap.vHt.resize(gp->qap.vHt.size() - 2);
}

template<typename ppT>
__global__ void groth_gamma_Lt(generator_params<ppT> *gp, instance_params* ip)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < gp->qap.num_inputs())
    {
        gp->gamma_ABC[tid] = (gp->beta * gp->qap.vAt[tid+1] + gp->alpha * gp->qap.vBt[tid+1] + gp->qap.vCt[tid+1]) * gp->gamma_inverse;
        tid += blockDim.x * gridDim.x;
    }

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t Lt_offset = gp->qap.num_inputs() + 1;
    while(tid < gp->qap.num_variables() - gp->qap.num_inputs())
    {
        gp->Lt[tid] = (gp->beta * gp->qap.vAt[Lt_offset + tid] + gp->alpha * gp->qap.vBt[Lt_offset + tid] + gp->qap.vCt[Lt_offset + tid]) * gp->delta_inverse;
        tid += blockDim.x * gridDim.x;
    }
}

template<typename ppT>
__global__ void groth_table_multi(generator_params<ppT> *gp, instance_params* ip)
{
    const size_t g1_scalar_size = ip->instance.size_in_bits();
    const size_t g1_window_size = 17; // libff::get_exp_window_size<libff::G1<ppT> >(g1_scalar_count, *params->g1_instance) - 7;
    // const size_t g1_in_window = 1ul << g1_window_size;
    const size_t g1_outerc = (g1_scalar_size + g1_window_size - 1) / g1_window_size;
    // gp->g1_table.resize(g1_outerc, libstl::vector<libff::G1<ppT>>(g1_in_window));
    // gp->g1_table.resize(g1_outerc);
    // for(size_t i=0; i<g1_outerc; i++) gp->g1_table[i].presize(g1_in_window, 1000, 32);

    const size_t g2_scalar_size = ip->instance.size_in_bits();
    const size_t g2_window_size = 16; // libff::get_exp_window_size<libff::G1<ppT> >(g1_scalar_count, *params->g1_instance) - 7;
    // const size_t g2_in_window = 1ul << g2_window_size;
    const size_t g2_outerc = (g2_scalar_size + g2_window_size - 1) / g2_window_size;
    // gp->g2_table.resize(g2_outerc, libstl::vector<libff::G2<ppT>>(g2_in_window));
    // gp->g2_table.resize(g2_outerc);
    // for(size_t i=0; i<g2_outerc; i++) gp->g2_table[i].presize(g2_in_window, 1000, 32);

    gp->g1_table = libstl::move(*libff::p_get_window_table<libff::G1<ppT>>(g1_scalar_size, g1_window_size, gp->g1_generator, g1_outerc * 8, 32));
    gp->g2_table = libstl::move(*libff::p_get_window_table<libff::G2<ppT>>(g2_scalar_size, g2_window_size, gp->g2_generator, g2_outerc * 8, 32));
}

template<typename ppT>
__global__ void groth_table(generator_params<ppT> *gp, instance_params* ip)
{
    // const size_t g1_scalar_count = non_zero_At + non_zero_Bt + qap.num_variables();
    const size_t g1_scalar_size = ip->instance.size_in_bits();
    const size_t g1_window_size = 17; // libff::get_exp_window_size<libff::G1<ppT> >(g1_scalar_count, *params->g1_instance) - 7;
    gp->g1_table = get_window_table(g1_scalar_size, g1_window_size, gp->g1_generator);

    // const size_t g2_scalar_count = non_zero_Bt;
    const size_t g2_scalar_size = ip->instance.size_in_bits();
    const size_t g2_window_size = 16; // libff::get_exp_window_size<libff::G2<ppT> >(g2_scalar_count, *params->g2_instance) - 7;
    gp->g2_table = get_window_table(g2_scalar_size, g2_window_size, gp->g2_generator);
}

template<typename ppT>
__global__ void groth_prover_key(generator_params<ppT> *gp, instance_params* ip)
{
    const size_t g1_scalar_size = ip->instance.size_in_bits();
    const size_t g1_window_size = 17; // libff::get_exp_window_size<libff::G1<ppT> >(g1_scalar_count, *params->g1_instance) - 7;
    const size_t g2_scalar_size = ip->instance.size_in_bits();
    const size_t g2_window_size = 16; // libff::get_exp_window_size<libff::G2<ppT> >(g2_scalar_count, *params->g2_instance) - 7;

    gp->alpha_g1 = gp->alpha * gp->g1_generator;
    gp->beta_g1 = gp->beta * gp->g1_generator;
    gp->beta_g2 = gp->beta * gp->g2_generator;
    gp->delta_g1 = gp->delta * gp->g1_generator;
    gp->delta_g2 = gp->delta * gp->g2_generator;

    gp->Zt_delta_inverse = gp->qap.Zt * gp->delta_inverse;
    
    gp->A_g1_query = libstl::move(*libff::p_batch_exp<libff::G1<ppT>, libff::Fr<ppT>>(g1_scalar_size, g1_window_size, gp->g1_table, gp->qap.vAt, ip->g1_instance, 1000, 32));    
    gp->B_g1_query = libstl::move(*libff::p_batch_exp<libff::G1<ppT>, libff::Fr<ppT>>(g1_scalar_size, g1_window_size, gp->g1_table, gp->qap.vBt, ip->g1_instance, 1000, 32));
    gp->B_g2_query = libstl::move(*libff::p_batch_exp<libff::G2<ppT>, libff::Fr<ppT>>(g2_scalar_size, g2_window_size, gp->g2_table, gp->qap.vBt, ip->g2_instance, 1000, 32));
    gp->H_g1_query = libstl::move(*libff::p_batch_exp_with_coeff<libff::G1<ppT>, libff::Fr<ppT>>(g1_scalar_size, g1_window_size, gp->g1_table, gp->Zt_delta_inverse, gp->qap.vHt, ip->g1_instance, 1000, 32));
    gp->L_g1_query = libstl::move(*libff::p_batch_exp<libff::G1<ppT>, libff::Fr<ppT>>(g1_scalar_size, g1_window_size, gp->g1_table, gp->Lt, ip->g1_instance, 1000, 32));

    ip->g1_instance.p_batch_to_special(gp->A_g1_query, 160, 32);
    ip->g1_instance.p_batch_to_special(gp->B_g1_query, 160, 32);
    ip->g2_instance.p_batch_to_special(gp->B_g2_query, 160, 32);
    ip->g1_instance.p_batch_to_special(gp->H_g1_query, 160, 32);
    ip->g1_instance.p_batch_to_special(gp->L_g1_query, 160, 32);


    gp->kp.pk = libstl::move(groth_proving_key<ppT>(libstl::move(gp->alpha_g1), 
                                                            libstl::move(gp->beta_g1),
                                                            libstl::move(gp->beta_g2),
                                                            libstl::move(gp->delta_g1),
                                                            libstl::move(gp->delta_g2),
                                                            libstl::move(gp->A_g1_query),
                                                            libstl::move(gp->B_g1_query),
                                                            libstl::move(gp->B_g2_query),
                                                            libstl::move(gp->H_g1_query),
                                                            libstl::move(gp->L_g1_query)));

}


template<typename ppT>
__global__ void groth_verifier_key_pairing(generator_params<ppT> *gp, instance_params* ip)
{
    gp->alpha_g1_beta_g2 = ppT::reduced_pairing(gp->alpha_g1, gp->beta_g2);
}


template<typename ppT>
__global__ void groth_verifier_key(generator_params<ppT> *gp, instance_params* ip)
{
    const size_t g1_scalar_size = ip->instance.size_in_bits();
    const size_t g1_window_size = 17; // libff::get_exp_window_size<libff::G1<ppT> >(g1_scalar_count, *params->g1_instance) - 7;

    gp->gamma_g2 = gp->gamma * gp->g2_generator;

    gp->gamma_ABC_g1_0 = gp->gamma_ABC_0 * gp->g1_generator;


    gp->gamma_ABC_g1_values = libstl::move(*libff::p_batch_exp<libff::G1<ppT>, libff::Fr<ppT>>(g1_scalar_size, g1_window_size, gp->g1_table, gp->gamma_ABC, ip->g1_instance, 1000, 32));

    gp->gamma_ABC_g1.presize(1+gp->gamma_ABC_g1_values.size(), 1000, 32);
    gp->gamma_ABC_g1[0] = gp->gamma_ABC_g1_0;

    gp->kp.vk = libstl::move(groth_verification_key<ppT>(libstl::move(gp->alpha_g1_beta_g2), libstl::move(gp->gamma_g2), libstl::move(gp->delta_g2), libstl::move(gp->gamma_ABC_g1)));

}


template<typename ppT>
__global__ void groth_gamma_ABC(generator_params<ppT> *gp, instance_params* ip)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < gp->gamma_ABC_g1_values.size())
    {
        gp->kp.vk.gamma_ABC_g1[tid + 1] = gp->gamma_ABC_g1_values[tid];
        tid += blockDim.x * gridDim.x;
    }
}


template<typename ppT>
void groth_generator(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip)
{
    groth_generate_random_element<ppT><<<1, 1>>>(gp, ip);
    cudaDeviceSynchronize();

    // qap
    r1cs_to_qap_instance_map_with_evaluation<ppT>(gp, rp, ip);


    // // r1cs_test<ppT><<<1, 1>>>(gp, ip);


    groth_gamma_Lt_Setup<ppT><<<1, 1>>>(gp, ip);
    groth_gamma_Lt<ppT><<<1000, 32>>>(gp, ip);


    // table
    groth_table_multi<ppT><<<1, 1>>>(gp, ip);

    // prover key
    groth_prover_key<ppT><<<1, 1>>>(gp, ip);


    // verifier key
    groth_verifier_key_pairing<ppT><<<1, 1>>>(gp, ip);
    groth_verifier_key<ppT><<<1, 1>>>(gp, ip);
    groth_gamma_ABC<ppT><<<1000, 32>>>(gp, ip);

}



///////    prover   ///////
template<typename ppT>
struct MSMParams
{
    libff::G1<ppT>* evaluation_At_g1;
    libff::G1<ppT>* evaluation_Bt_g1;
    libff::G2<ppT>* evaluation_Bt_g2;
    libff::G1<ppT>* evaluation_Ht_g1;
    libff::G1<ppT>* evaluation_Lt_g1;

    size_t evaluation_At_g1_total;
    size_t evaluation_Bt_g1_total;
    size_t evaluation_Bt_g2_total;
    size_t evaluation_Ht_g1_total;
    size_t evaluation_Lt_g1_total;
};


template<typename ppT>
struct MSMParams_host
{
    libff::G1<ppT> evaluation_At_g1;
    libff::G1<ppT> evaluation_Bt_g1;
    libff::G2<ppT> evaluation_Bt_g2;
    libff::G1<ppT> evaluation_Ht_g1;
    libff::G1<ppT> evaluation_Lt_g1;

    size_t evaluation_At_g1_total;
    size_t evaluation_Bt_g1_total;
    size_t evaluation_Bt_g2_total;
    size_t evaluation_Ht_g1_total;
    size_t evaluation_Lt_g1_total;
};

template<typename ppT_host, typename ppT_device>
struct DataTransferParams{
    size_t device_id;
    r1cs_constraint_system<libff::Fr<ppT_device>>** dcs;
    groth_proving_key<ppT_device>** dpk;
    r1cs_primary_input<libff::Fr<ppT_device>>** dpi;
    r1cs_auxiliary_input<libff::Fr<ppT_device>>** dai;
    r1cs_constraint_system_host<libff::Fr<ppT_host>>* hcs;
    groth_proving_key_host<ppT_host>* hpk;
    r1cs_primary_input_host<libff::Fr<ppT_host>>* hpi;
    r1cs_auxiliary_input_host<libff::Fr<ppT_host>>* hai;

    MSMParams<ppT_device> msm;
    instance_params* ip;
    bool flag;
};




template<typename ppT_host, typename ppT_device>
void groth_prover_cpu2gpu(r1cs_constraint_system<libff::Fr<ppT_device>>** dcs, groth_proving_key<ppT_device>** dpk, r1cs_primary_input<libff::Fr<ppT_device>>** dpi, r1cs_auxiliary_input<libff::Fr<ppT_device>>** dai,
                                r1cs_constraint_system_host<libff::Fr<ppT_host>>* hcs, groth_proving_key_host<ppT_host>* hpk, r1cs_primary_input_host<libff::Fr<ppT_host>>* hpi, r1cs_auxiliary_input_host<libff::Fr<ppT_host>>* hai,
                                instance_params* ip)
{
    *dcs = libstl::create_host<r1cs_constraint_system<libff::Fr<ppT_device>>>();
    *dpk = libstl::create_host<groth_proving_key<ppT_device>>();
    *dpi = libstl::create_host<r1cs_primary_input<libff::Fr<ppT_device>>>();
    *dai = libstl::create_host<r1cs_auxiliary_input<libff::Fr<ppT_device>>>();

    r1cs_constraint_system_host2device(*dcs, hcs, &ip->instance);
    r1cs_primary_input_host2device(*dpi, hpi, &ip->instance);
    r1cs_auxiliary_input_host2device(*dai, hai, &ip->instance);
    // groth_proving_key_host2device(*dpk, hpk, &ip->g1_instance, &ip->g2_instance, 0);
}

template<typename ppT_host, typename ppT_device>
void* groth_prover_cpu2gpu_thread(void* dp)
{
    DataTransferParams<ppT_host, ppT_device>* d = (DataTransferParams<ppT_host, ppT_device>*)dp;
    cudaSetDevice(d->device_id);
    groth_prover_cpu2gpu(d->dcs, d->dpk, d->dpi, d->dai,
                                    d->hcs, d->hpk, d->hpi, d->hai, d->ip);

    return 0;
}


cudaEvent_t event1;
cudaEvent_t event2;
cudaEvent_t event3;
cudaEvent_t event4;
cudaEvent_t event5;


template<typename ppT>
void groth_multi_exp_host(MSMParams<ppT>* msm, qap_witness<libff::Fr<ppT>>* qap, groth_proving_key<ppT>* pk, r1cs_primary_input<libff::Fr<ppT>>* primary_input, r1cs_auxiliary_input<libff::Fr<ppT>>* auxiliary_input,
                                        libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    size_t const_padded_assignment_size = 1 + primary_input->size_host() + auxiliary_input->size_host();
    const_padded_assignment->presize_host(const_padded_assignment_size, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            while(idx < const_padded_assignment_size)
            {
                (*const_padded_assignment)[idx] = qap->vcoefficients_for_ABCs[idx-1];
                idx += gridDim.x * blockDim.x;
            }

            if(blockIdx.x * blockDim.x + threadIdx.x == 0)
            {
                (*const_padded_assignment)[0] = instance->one();
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_in = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_in->pcopy_host(*const_padded_assignment, gridSize, blockSize);
    libff::Fr<ppT>* zero = instance->zero_host();
    const_padded_assignment_in->presize_host(qap->num_variables_host() + 1, zero, gridSize, blockSize);

    libff::G1<ppT>* g1_zero = g1_instance->zero_host();
    cudaEventSynchronize(event1);

    pk->A_g1_query.presize_host(qap->num_variables_host() + 1, g1_zero, gridSize, blockSize);
    // msm->evaluation_At_g1 = libff::p_multi_exp_faster_multi_GPU<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(pk->A_g1_query, *const_padded_assignment_in, *instance, *g1_instance, gridSize, blockSize);
    msm->evaluation_At_g1 = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(pk->A_g1_query, *const_padded_assignment_in, *instance, *g1_instance, gridSize, blockSize);

    cudaEventSynchronize(event2);
    pk->B_g1_query.presize_host(qap->num_variables_host() + 1, g1_zero, gridSize, blockSize);
    msm->evaluation_Bt_g1 = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(pk->B_g1_query, *const_padded_assignment_in, *instance, *g1_instance, gridSize, blockSize);

    libff::G2<ppT>* g2_zero = g2_instance->zero_host();
    cudaEventSynchronize(event3);
    pk->B_g2_query.presize_host(qap->num_variables_host() + 1, g2_zero, gridSize, blockSize);
    msm->evaluation_Bt_g2 = libff::p_multi_exp_faster_multi_GPU_host<libff::G2<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(pk->B_g2_query, *const_padded_assignment_in, *instance, *g2_instance, gridSize, blockSize);

    libstl::vector<libff::G1<ppT>>* Ht_g1_query = libstl::create_host<libstl::vector<libff::G1<ppT>>>();
    libstl::vector<libff::Fr<ppT>>* Ht_g1_coefficients_for_H = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    cudaEventSynchronize(event4);
    Ht_g1_query->pcopy_host(pk->H_g1_query, gridSize, blockSize);
    Ht_g1_query->presize_host(qap->degree_host() - 1, g1_zero, gridSize, blockSize);
    Ht_g1_coefficients_for_H->pcopy_host(qap->vcoefficients_for_H, gridSize, blockSize);
    Ht_g1_coefficients_for_H->presize_host(qap->degree_host() - 1, zero, gridSize, blockSize);
    msm->evaluation_Ht_g1 = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(*Ht_g1_query, *Ht_g1_coefficients_for_H, *instance, *g1_instance, gridSize, blockSize);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_wit = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_wit->presize_host(qap->num_variables_host() - qap->num_inputs_host(), zero, gridSize, blockSize);
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < qap->num_variables() - qap->num_inputs())
            {
                (*const_padded_assignment_wit)[idx] = (*const_padded_assignment)[idx + qap->num_inputs()+1];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);
    
    cudaEventSynchronize(event5);
    msm->evaluation_Lt_g1 = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(pk->L_g1_query, *const_padded_assignment_wit, *instance, *g1_instance, gridSize, blockSize);
}

template<typename ppT>
void groth_multi_exp_host1(MSMParams<ppT>* msm, qap_witness<libff::Fr<ppT>>* qap, groth_proving_key<ppT>* pk, r1cs_primary_input<libff::Fr<ppT>>* primary_input, r1cs_auxiliary_input<libff::Fr<ppT>>* auxiliary_input,
                                        libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;

    msm->evaluation_At_g1 = g1_instance->zero_host();

    size_t msmlockMem;
    libstl::lock_host(msmlockMem);

    libff::Fr<ppT>* zero = instance->zero_host();
    libff::G1<ppT>* g1_zero = g1_instance->zero_host();

    pk->A_g1_query.presize_host(qap->num_variables_host() + 1, g1_zero, gridSize, blockSize);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    size_t const_padded_assignment_size = 1 + primary_input->size_host() + auxiliary_input->size_host();
    const_padded_assignment->presize_host(const_padded_assignment_size, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            while(idx < const_padded_assignment_size)
            {
                (*const_padded_assignment)[idx] = qap->vcoefficients_for_ABCs[idx-1];
                idx += gridDim.x * blockDim.x;
            }

            if(blockIdx.x * blockDim.x + threadIdx.x == 0)
            {
                (*const_padded_assignment)[0] = instance->one();
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_in = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_in->pcopy_host(*const_padded_assignment, gridSize, blockSize);
    const_padded_assignment_in->presize_host(qap->num_variables_host() + 1, zero, gridSize, blockSize);

    // cudaEventSynchronize(event1);
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(msm->evaluation_At_g1, pk->A_g1_query, *const_padded_assignment_in, *instance, *g1_instance, gridSize, blockSize);
    msm->evaluation_At_g1_total = qap->num_variables_host() + 1;

    libstl::resetlock_host(msmlockMem);
}


template<typename ppT>
void groth_multi_exp_host2(MSMParams<ppT>* msm, qap_witness<libff::Fr<ppT>>* qap, groth_proving_key<ppT>* pk, r1cs_primary_input<libff::Fr<ppT>>* primary_input, r1cs_auxiliary_input<libff::Fr<ppT>>* auxiliary_input,
                                        libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;

    msm->evaluation_Bt_g1 = g1_instance->zero_host();
    size_t msmlockMem;
    libstl::lock_host(msmlockMem);


    libff::Fr<ppT>* zero = instance->zero_host();
    libff::G1<ppT>* g1_zero = g1_instance->zero_host();
    pk->B_g1_query.presize_host(qap->num_variables_host() + 1, g1_zero, gridSize, blockSize);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    size_t const_padded_assignment_size = 1 + primary_input->size_host() + auxiliary_input->size_host();
    const_padded_assignment->presize_host(const_padded_assignment_size, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            while(idx < const_padded_assignment_size)
            {
                (*const_padded_assignment)[idx] = qap->vcoefficients_for_ABCs[idx-1];
                idx += gridDim.x * blockDim.x;
            }

            if(blockIdx.x * blockDim.x + threadIdx.x == 0)
            {
                (*const_padded_assignment)[0] = instance->one();
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_in = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_in->pcopy_host(*const_padded_assignment, gridSize, blockSize);

    const_padded_assignment_in->presize_host(qap->num_variables_host() + 1, zero, gridSize, blockSize);

    // cudaEventSynchronize(event2);

    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(msm->evaluation_Bt_g1, pk->B_g1_query, *const_padded_assignment_in, *instance, *g1_instance, gridSize, blockSize);
    msm->evaluation_Bt_g1_total = qap->num_variables_host() + 1;

    libstl::resetlock_host(msmlockMem);
}

template<typename ppT>
void groth_multi_exp_host3(MSMParams<ppT>* msm, qap_witness<libff::Fr<ppT>>* qap, groth_proving_key<ppT>* pk, r1cs_primary_input<libff::Fr<ppT>>* primary_input, r1cs_auxiliary_input<libff::Fr<ppT>>* auxiliary_input,
                                        libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;
    msm->evaluation_Bt_g2 = g2_instance->zero_host();
    size_t msmlockMem;
    libstl::lock_host(msmlockMem);

    libff::Fr<ppT>* zero = instance->zero_host();
    libff::G1<ppT>* g1_zero = g1_instance->zero_host();
    libff::G2<ppT>* g2_zero = g2_instance->zero_host();
    pk->B_g2_query.presize_host(qap->num_variables_host() + 1, g2_zero, gridSize, blockSize);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    size_t const_padded_assignment_size = 1 + primary_input->size_host() + auxiliary_input->size_host();
    const_padded_assignment->presize_host(const_padded_assignment_size, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            while(idx < const_padded_assignment_size)
            {
                (*const_padded_assignment)[idx] = qap->vcoefficients_for_ABCs[idx-1];
                idx += gridDim.x * blockDim.x;
            }

            if(blockIdx.x * blockDim.x + threadIdx.x == 0)
            {
                (*const_padded_assignment)[0] = instance->one();
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_in = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_in->pcopy_host(*const_padded_assignment, gridSize, blockSize);
    const_padded_assignment_in->presize_host(qap->num_variables_host() + 1, zero, gridSize, blockSize);

    // cudaEventSynchronize(event3);
    
    libff::p_multi_exp_faster_multi_GPU_host<libff::G2<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(msm->evaluation_Bt_g2, pk->B_g2_query, *const_padded_assignment_in, *instance, *g2_instance, gridSize, blockSize);
    msm->evaluation_Bt_g2_total = qap->num_variables_host() + 1;

    libstl::resetlock_host(msmlockMem);
}



template<typename ppT>
void groth_multi_exp_host4(MSMParams<ppT>* msm, qap_witness<libff::Fr<ppT>>* qap, groth_proving_key<ppT>* pk, r1cs_primary_input<libff::Fr<ppT>>* primary_input, r1cs_auxiliary_input<libff::Fr<ppT>>* auxiliary_input,
                                        libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;

    msm->evaluation_Ht_g1 = g1_instance->zero_host();
    size_t msmlockMem;
    libstl::lock_host(msmlockMem);

    libff::Fr<ppT>* zero = instance->zero_host();
    libff::G1<ppT>* g1_zero = g1_instance->zero_host();

    libstl::vector<libff::G1<ppT>>* Ht_g1_query = libstl::create_host<libstl::vector<libff::G1<ppT>>>();
    libstl::vector<libff::Fr<ppT>>* Ht_g1_coefficients_for_H = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();

    // cudaEventSynchronize(event4);
    Ht_g1_query->pcopy_host(pk->H_g1_query, gridSize, blockSize);
    Ht_g1_query->presize_host(qap->degree_host() - 1, g1_zero, gridSize, blockSize);
    Ht_g1_coefficients_for_H->pcopy_host(qap->vcoefficients_for_H, gridSize, blockSize);
    Ht_g1_coefficients_for_H->presize_host(qap->degree_host() - 1, zero, gridSize, blockSize);
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(msm->evaluation_Ht_g1, *Ht_g1_query, *Ht_g1_coefficients_for_H, *instance, *g1_instance, gridSize, blockSize);
    msm->evaluation_Ht_g1_total = qap->degree_host() - 1;

    libstl::resetlock_host(msmlockMem);
}


template<typename ppT>
void groth_multi_exp_host5(MSMParams<ppT>* msm, qap_witness<libff::Fr<ppT>>* qap, groth_proving_key<ppT>* pk, r1cs_primary_input<libff::Fr<ppT>>* primary_input, r1cs_auxiliary_input<libff::Fr<ppT>>* auxiliary_input,
                                        libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;

    msm->evaluation_Lt_g1 = g1_instance->zero_host();
    size_t msmlockMem;
    libstl::lock_host(msmlockMem);

    libff::Fr<ppT>* zero = instance->zero_host();
    libff::G1<ppT>* g1_zero = g1_instance->zero_host();

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    size_t const_padded_assignment_size = 1 + primary_input->size_host() + auxiliary_input->size_host();
    const_padded_assignment->presize_host(const_padded_assignment_size, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
            while(idx < const_padded_assignment_size)
            {
                (*const_padded_assignment)[idx] = qap->vcoefficients_for_ABCs[idx-1];
                idx += gridDim.x * blockDim.x;
            }

            if(blockIdx.x * blockDim.x + threadIdx.x == 0)
            {
                (*const_padded_assignment)[0] = instance->one();
            }
        }
    );
    cudaStreamSynchronize(0);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_in = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_in->pcopy_host(*const_padded_assignment, gridSize, blockSize);
    const_padded_assignment_in->presize_host(qap->num_variables_host() + 1, zero, gridSize, blockSize);

    libstl::vector<libff::Fr<ppT>>* const_padded_assignment_wit = libstl::create_host<libstl::vector<libff::Fr<ppT>>>();
    const_padded_assignment_wit->presize_host(qap->num_variables_host() - qap->num_inputs_host(), zero, gridSize, blockSize);
    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < qap->num_variables() - qap->num_inputs())
            {
                (*const_padded_assignment_wit)[idx] = (*const_padded_assignment)[idx + qap->num_inputs()+1];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);

    // cudaEventSynchronize(event5);
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(msm->evaluation_Lt_g1, pk->L_g1_query, *const_padded_assignment_wit, *instance, *g1_instance, gridSize, blockSize);
    msm->evaluation_Lt_g1_total = qap->num_variables_host() - qap->num_inputs_host();

    libstl::resetlock_host(msmlockMem);
}


template<typename ppT_host, typename ppT_device>
void* groth_prover_witness_map_msm_thread(void* dp)
{
    DataTransferParams<ppT_host, ppT_device>* d = (DataTransferParams<ppT_host, ppT_device>*)dp;
    cudaSetDevice(d->device_id);

    cudaEventCreate( &event1);
    cudaEventCreate( &event2);
    cudaEventCreate( &event3);
    cudaEventCreate( &event4);
    cudaEventCreate( &event5);

    // libstl::printAllocator_host();

    // key
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    groth_proving_key_host2device1(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    cudaEventRecord( event1, stream); 
    groth_proving_key_host2device2(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    cudaEventRecord( event2, stream); 
    groth_proving_key_host2device3(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    cudaEventRecord( event3, stream); 
    groth_proving_key_host2device4(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    cudaEventRecord( event4, stream); 
    groth_proving_key_host2device5(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    cudaEventRecord( event5, stream);

    // cudaEventSynchronize(event1);
    // cudaEventSynchronize(event2);
    // cudaEventSynchronize(event3);
    // cudaEventSynchronize(event4);
    // cudaEventSynchronize(event5);


    // libstl::printAllocator_host();

    // size_t lockMem;
    // libstl::lock_host(lockMem);

    // qap witness

    qap_witness<libff::Fr<ppT_device>>* qap = r1cs_to_qap_witness_map(*d->dcs, *d->dpi, *d->dai, &d->ip->instance);

    // libstl::printAllocator_host();

    // multi exp
    cudaEvent_t eventMultiExpStart, eventMultiExpEnd;
    cudaEventCreate( &eventMultiExpStart);
	cudaEventCreate( &eventMultiExpEnd);
    cudaEventRecord( eventMultiExpStart, 0); 
    cudaEventSynchronize(eventMultiExpStart);

    cudaEvent_t eventMultiExp1WaitStart, eventMultiExp1WaitEnd;
    cudaEventCreate( &eventMultiExp1WaitStart);
	cudaEventCreate( &eventMultiExp1WaitEnd);
    cudaEventRecord( eventMultiExp1WaitStart, 0); 
    cudaEventSynchronize(eventMultiExp1WaitStart);

    cudaEventSynchronize(event1);

    cudaEventRecord( eventMultiExp1WaitEnd, 0);
    cudaEventSynchronize(eventMultiExp1WaitEnd);
    float   TimeMultiExp1Wait;
    cudaEventElapsedTime( &TimeMultiExp1Wait, eventMultiExp1WaitStart, eventMultiExp1WaitEnd );
    // printf( "Time to MultiExp1Wait:  %3.5f ms\n", TimeMultiExp1Wait );

    cudaEvent_t eventMultiExp1Start, eventMultiExp1End;
    cudaEventCreate( &eventMultiExp1Start);
	cudaEventCreate( &eventMultiExp1End);
    cudaEventRecord( eventMultiExp1Start, 0); 
    cudaEventSynchronize(eventMultiExp1Start);

    // groth_proving_key_host2device2(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    // cudaEventRecord( event2, stream);
    
    groth_multi_exp_host1<ppT_device>(&d->msm, qap, *d->dpk, *d->dpi, *d->dai, &d->ip->instance, &d->ip->g1_instance, &d->ip->g2_instance);

    cudaEventRecord( eventMultiExp1End, 0);
    cudaEventSynchronize(eventMultiExp1End);
    float   TimeMultiExp1;
    cudaEventElapsedTime( &TimeMultiExp1, eventMultiExp1Start, eventMultiExp1End );
    // printf( "Time Thread %lu to MultiExp1:  %3.5f ms\n", d->device_id, TimeMultiExp1 );

    cudaEvent_t eventMultiExp2WaitStart, eventMultiExp2WaitEnd;
    cudaEventCreate( &eventMultiExp2WaitStart);
	cudaEventCreate( &eventMultiExp2WaitEnd);
    cudaEventRecord( eventMultiExp2WaitStart, 0); 
    cudaEventSynchronize(eventMultiExp2WaitStart);

    cudaEventSynchronize(event2);

    cudaEventRecord( eventMultiExp2WaitEnd, 0);
    cudaEventSynchronize(eventMultiExp2WaitEnd);
    float   TimeMultiExp2Wait;
    cudaEventElapsedTime( &TimeMultiExp2Wait, eventMultiExp2WaitStart, eventMultiExp2WaitEnd );
    // printf( "Time to MultiExp2Wait:  %3.5f ms\n", TimeMultiExp2Wait );

    cudaEvent_t eventMultiExp2Start, eventMultiExp2End;
    cudaEventCreate( &eventMultiExp2Start);
	cudaEventCreate( &eventMultiExp2End);
    cudaEventRecord( eventMultiExp2Start, 0); 
    cudaEventSynchronize(eventMultiExp2Start);

    // groth_proving_key_host2device3(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    // cudaEventRecord( event3, stream); 

    groth_multi_exp_host2<ppT_device>(&d->msm, qap, *d->dpk, *d->dpi, *d->dai, &d->ip->instance, &d->ip->g1_instance, &d->ip->g2_instance);

    cudaEventRecord( eventMultiExp2End, 0);
    cudaEventSynchronize(eventMultiExp2End);
    float   TimeMultiExp2;
    cudaEventElapsedTime( &TimeMultiExp2, eventMultiExp2Start, eventMultiExp2End );
    // printf( "Time Thread %lu to MultiExp2:  %3.5f ms\n", d->device_id, TimeMultiExp2 );

    cudaEvent_t eventMultiExp3WaitStart, eventMultiExp3WaitEnd;
    cudaEventCreate( &eventMultiExp3WaitStart);
	cudaEventCreate( &eventMultiExp3WaitEnd);
    cudaEventRecord( eventMultiExp3WaitStart, 0); 
    cudaEventSynchronize(eventMultiExp3WaitStart);

    cudaEventSynchronize(event3);

    cudaEventRecord( eventMultiExp3WaitEnd, 0);
    cudaEventSynchronize(eventMultiExp3WaitEnd);
    float   TimeMultiExp3Wait;
    cudaEventElapsedTime( &TimeMultiExp3Wait, eventMultiExp3WaitStart, eventMultiExp3WaitEnd );
    // printf( "Time to MultiExp3Wait:  %3.5f ms\n", TimeMultiExp3Wait );

    cudaEvent_t eventMultiExp3Start, eventMultiExp3End;
    cudaEventCreate( &eventMultiExp3Start);
	cudaEventCreate( &eventMultiExp3End);
    cudaEventRecord( eventMultiExp3Start, 0); 
    cudaEventSynchronize(eventMultiExp3Start);

    // groth_proving_key_host2device4(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    // cudaEventRecord( event4, stream); 
    
    groth_multi_exp_host3<ppT_device>(&d->msm, qap, *d->dpk, *d->dpi, *d->dai, &d->ip->instance, &d->ip->g1_instance, &d->ip->g2_instance);
    
    cudaEventRecord( eventMultiExp3End, 0);
    cudaEventSynchronize(eventMultiExp3End);
    float   TimeMultiExp3;
    cudaEventElapsedTime( &TimeMultiExp3, eventMultiExp3Start, eventMultiExp3End );
    // printf( "Time Thread %lu to MultiExp3:  %3.5f ms\n", d->device_id, TimeMultiExp3 );

    cudaEvent_t eventMultiExp4WaitStart, eventMultiExp4WaitEnd;
    cudaEventCreate( &eventMultiExp4WaitStart);
	cudaEventCreate( &eventMultiExp4WaitEnd);
    cudaEventRecord( eventMultiExp4WaitStart, 0); 
    cudaEventSynchronize(eventMultiExp4WaitStart);

    cudaEventSynchronize(event4);

    cudaEventRecord( eventMultiExp4WaitEnd, 0);
    cudaEventSynchronize(eventMultiExp4WaitEnd);
    float   TimeMultiExp4Wait;
    cudaEventElapsedTime( &TimeMultiExp4Wait, eventMultiExp4WaitStart, eventMultiExp4WaitEnd );
    // printf( "Time to MultiExp4Wait:  %3.5f ms\n", TimeMultiExp4Wait );

    cudaEvent_t eventMultiExp4Start, eventMultiExp4End;
    cudaEventCreate( &eventMultiExp4Start);
	cudaEventCreate( &eventMultiExp4End);
    cudaEventRecord( eventMultiExp4Start, 0); 
    cudaEventSynchronize(eventMultiExp4Start);

    // groth_proving_key_host2device5(*d->dpk, d->hpk, &d->ip->g1_instance, &d->ip->g2_instance, stream);
    // cudaEventRecord( event5, stream); 

    groth_multi_exp_host4<ppT_device>(&d->msm, qap, *d->dpk, *d->dpi, *d->dai, &d->ip->instance, &d->ip->g1_instance, &d->ip->g2_instance);

    cudaEventRecord( eventMultiExp4End, 0);
    cudaEventSynchronize(eventMultiExp4End);
    float   TimeMultiExp4;
    cudaEventElapsedTime( &TimeMultiExp4, eventMultiExp4Start, eventMultiExp4End );
    // printf( "Time Thread %lu to MultiExp4:  %3.5f ms\n", d->device_id, TimeMultiExp4 );

    cudaEvent_t eventMultiExp5WaitStart, eventMultiExp5WaitEnd;
    cudaEventCreate( &eventMultiExp5WaitStart);
	cudaEventCreate( &eventMultiExp5WaitEnd);
    cudaEventRecord( eventMultiExp5WaitStart, 0); 
    cudaEventSynchronize(eventMultiExp5WaitStart);

    cudaEventSynchronize(event5);

    cudaEventRecord( eventMultiExp5WaitEnd, 0);
    cudaEventSynchronize(eventMultiExp5WaitEnd);
    float   TimeMultiExp5Wait;
    cudaEventElapsedTime( &TimeMultiExp5Wait, eventMultiExp5WaitStart, eventMultiExp5WaitEnd );
    // printf( "Time to MultiExp5Wait:  %3.5f ms\n", TimeMultiExp5Wait );


    cudaEvent_t eventMultiExp5Start, eventMultiExp5End;
    cudaEventCreate( &eventMultiExp5Start);
	cudaEventCreate( &eventMultiExp5End);
    cudaEventRecord( eventMultiExp5Start, 0); 
    cudaEventSynchronize(eventMultiExp5Start);

    // cudaEventSynchronize(event5);
    groth_multi_exp_host5<ppT_device>(&d->msm, qap, *d->dpk, *d->dpi, *d->dai, &d->ip->instance, &d->ip->g1_instance, &d->ip->g2_instance);
    
    cudaEventRecord( eventMultiExp5End, 0);
    cudaEventSynchronize(eventMultiExp5End);
    float   TimeMultiExp5;
    cudaEventElapsedTime( &TimeMultiExp5, eventMultiExp5Start, eventMultiExp5End );
    // printf( "Time Thread %lu to MultiExp5:  %3.5f ms\n", d->device_id, TimeMultiExp5 );

    cudaEventRecord( eventMultiExpEnd, 0);
    cudaEventSynchronize(eventMultiExpEnd);
    float   TimeMultiExp;
    cudaEventElapsedTime( &TimeMultiExp, eventMultiExpStart, eventMultiExpEnd );
    printf( "Time to MSM:  %3.5f ms\n", TimeMultiExp );

    // libstl::printAllocator_host();

    // if(d->flag == false)
    // {
    //     libstl::resetlock_host(lockMem);
    // }

    // libstl::printAllocator_host();
    return 0;
}

template<typename ppT_host, typename ppT_device>
void groth_prover_msm_device2host(MSMParams_host<ppT_host>* hmsm, MSMParams<ppT_device>* dmsm, libff::G1<ppT_host>* g1_instance, libff::G2<ppT_host>* g2_instance)
{
    cudaMemcpy(&hmsm->evaluation_At_g1, dmsm->evaluation_At_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hmsm->evaluation_Bt_g1, dmsm->evaluation_Bt_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hmsm->evaluation_Bt_g2, dmsm->evaluation_Bt_g2, sizeof(libff::G2<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hmsm->evaluation_Ht_g1, dmsm->evaluation_Ht_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hmsm->evaluation_Lt_g1, dmsm->evaluation_Lt_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);

    hmsm->evaluation_At_g1.set_params(g1_instance->params);
    hmsm->evaluation_Bt_g1.set_params(g1_instance->params);
    hmsm->evaluation_Bt_g2.set_params(g2_instance->params);
    hmsm->evaluation_Ht_g1.set_params(g1_instance->params);
    hmsm->evaluation_Lt_g1.set_params(g1_instance->params);

    hmsm->evaluation_At_g1_total = dmsm->evaluation_At_g1_total;
    hmsm->evaluation_Bt_g1_total = dmsm->evaluation_Bt_g1_total;
    hmsm->evaluation_Bt_g2_total = dmsm->evaluation_Bt_g2_total;
    hmsm->evaluation_Ht_g1_total = dmsm->evaluation_Ht_g1_total;
    hmsm->evaluation_Lt_g1_total = dmsm->evaluation_Lt_g1_total;
}

template<typename ppT>
void groth_prover_reduce_result(groth_proving_key_host<ppT>* hpk, MSMParams_host<ppT>* msm, libff::Fr<ppT>* instance, libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    libff::Fr<ppT> r = instance->random_element();
    libff::Fr<ppT> s = instance->random_element();

    int device_count;
    cudaGetDeviceCount(&device_count);

    hpk->alpha_g1.set_params(g1_instance->params);
    hpk->beta_g1.set_params(g1_instance->params);
    hpk->delta_g1.set_params(g1_instance->params);
    hpk->beta_g2.set_params(g2_instance->params);
    hpk->delta_g2.set_params(g2_instance->params);

    libff::G1<ppT> evaluation_At_g1 = msm[device_count - 1].evaluation_At_g1;
    libff::G1<ppT> evaluation_Bt_g1 = msm[device_count - 1].evaluation_Bt_g1;
    libff::G2<ppT> evaluation_Bt_g2 = msm[device_count - 1].evaluation_Bt_g2;
    libff::G1<ppT> evaluation_Ht_g1 = msm[device_count - 1].evaluation_Ht_g1;
    libff::G1<ppT> evaluation_Lt_g1 = msm[device_count - 1].evaluation_Lt_g1;

    if(device_count != 1)
    {
        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t total = msm[i].evaluation_At_g1_total;
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(sgroup > num_groups) sgroup = num_groups;
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup == egroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                evaluation_At_g1 = evaluation_At_g1.dbl();
            }
            evaluation_At_g1 = evaluation_At_g1 + msm[i].evaluation_At_g1;
        }

        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t total = msm[i].evaluation_Bt_g1_total;
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup == egroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                evaluation_Bt_g1 = evaluation_Bt_g1.dbl();
            }
            evaluation_Bt_g1 = evaluation_Bt_g1 + msm[i].evaluation_Bt_g1;
        }

        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t total = msm[i].evaluation_Bt_g2_total;
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup == egroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                evaluation_Bt_g2 = evaluation_Bt_g2.dbl();
            }
            evaluation_Bt_g2 = evaluation_Bt_g2 + msm[i].evaluation_Bt_g2;
        }

        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t total = msm[i].evaluation_Ht_g1_total;
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup == egroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                evaluation_Ht_g1 = evaluation_Ht_g1.dbl();
            }
            evaluation_Ht_g1 = evaluation_Ht_g1 + msm[i].evaluation_Ht_g1;
        }

        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t total = msm[i].evaluation_Lt_g1_total;
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup == egroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                evaluation_Lt_g1 = evaluation_Lt_g1.dbl();
            }
            evaluation_Lt_g1 = evaluation_Lt_g1 + msm[i].evaluation_Lt_g1;
        }
    }

    libff::G1<ppT> g1_A = hpk->alpha_g1 + evaluation_At_g1 + r * hpk->delta_g1;
    libff::G1<ppT> g1_B = hpk->beta_g1 +  evaluation_Bt_g1 + s * hpk->delta_g1;
    libff::G2<ppT> g2_B = hpk->beta_g2 + evaluation_Bt_g2  + s * hpk->delta_g2;
    libff::G1<ppT> g1_C = evaluation_Ht_g1 + evaluation_Lt_g1 +s * g1_A + r * g1_B - (r * s) * hpk->delta_g1;

    // g1_A.to_special();
    // printf("g_A:\n");
    // printf("X: "); g1_A.X.as_bigint().print();
    // printf("Y: "); g1_A.Y.as_bigint().print();

    // g2_B.to_special();
    // printf("g_B:\n");
    // printf("X1: "); g2_B.X.c0.as_bigint().print();
    // printf("X2: "); g2_B.X.c1.as_bigint().print();

    // printf("Y1: "); g2_B.Y.c0.as_bigint().print(); 
    // printf("Y2: "); g2_B.Y.c1.as_bigint().print();

    // g1_C.to_special();
    // printf("g_C:\n");
    // printf("X: "); g1_C.X.as_bigint().print();
    // printf("Y: "); g1_C.Y.as_bigint().print();
}


template<typename ppT_host, typename ppT_device>
void groth_prover(r1cs_constraint_system_host<libff::Fr<ppT_host>>* hcs, groth_proving_key_host<ppT_host>* hpk, r1cs_primary_input_host<libff::Fr<ppT_host>>* hpi, r1cs_auxiliary_input_host<libff::Fr<ppT_host>>* hai,
                            instance_params* ip[], h_instance_params* hip)
{
    int device_count;
    cudaGetDeviceCount( &device_count );
    CUTThread  thread[device_count];

    // data CPU to GPU
    DataTransferParams<ppT_host, ppT_device> dtp[device_count];
    // DeviceParams<ppT_device> dp[device_count];

    r1cs_constraint_system<libff::Fr<ppT_device>>* dcs[device_count];
    groth_proving_key<ppT_device>* dpk[device_count];
    r1cs_primary_input<libff::Fr<ppT_device>>* dpi[device_count];
    r1cs_auxiliary_input<libff::Fr<ppT_device>>* dai[device_count];

    // cpu-gpu
    for(size_t i=0; i<device_count; i++)
    {
        dtp[i].device_id = i;
        dtp[i].dcs = &dcs[i];
        dtp[i].dpk = &dpk[i];
        dtp[i].dpi = &dpi[i];
        dtp[i].dai = &dai[i];
        dtp[i].hcs = hcs;
        dtp[i].hpk = hpk;
        dtp[i].hpi = hpi;
        dtp[i].hai = hai;
        dtp[i].ip = ip[i];
        dtp[i].flag = false;

        thread[i] = start_thread(groth_prover_cpu2gpu_thread<ppT_host, ppT_device>, &dtp[i]);
    }
    for(size_t i=0; i<device_count; i++)
    {   
        end_thread(thread[i]);
    }

    // libstl::printAllocator_host();


    // // qap witness msm
    // for(size_t j=0; j < 1; j++)
    // {
    //     for(size_t i=0; i<device_count; i++)
    //     {
    //         thread[i] = start_thread(groth_prover_witness_map_msm_thread<ppT_host, ppT_device>, &dtp[i]);
    //     }
    //     for(size_t i=0; i<device_count; i++)
    //     {   
    //         end_thread(thread[i]);
    //     }
    // }

    for(size_t i=0; i<device_count; i++)
    {
        dtp[i].flag = true;
        thread[i] = start_thread(groth_prover_witness_map_msm_thread<ppT_host, ppT_device>, &dtp[i]);
    }
    for(size_t i=0; i<device_count; i++)
    {   
        end_thread(thread[i]);
    }
    
    // reduce result
    MSMParams_host<ppT_host> hmsm[device_count];
    for(size_t i=0; i<device_count; i++)
    {
        groth_prover_msm_device2host(&hmsm[i], &dtp[i].msm, &hip->h_g1_instance, &hip->h_g2_instance);
    }

    groth_prover_reduce_result(hpk, hmsm, &hip->h_instance, &hip->h_g1_instance, &hip->h_g2_instance);

}


//////    device --- host   //////
template<typename ppT_host, typename ppT_device>
void groth_proving_key_device2host(groth_proving_key_host<ppT_host>* hpk, groth_proving_key<ppT_device>* dpk)
{
    cudaMemcpy(&hpk->alpha_g1, &dpk->alpha_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hpk->beta_g1, &dpk->beta_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hpk->beta_g2, &dpk->beta_g2, sizeof(libff::G2<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hpk->delta_g1, &dpk->delta_g1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hpk->delta_g2, &dpk->delta_g2, sizeof(libff::G2<ppT_device>), cudaMemcpyDeviceToHost);

    vector_device2host(&hpk->A_g1_query, &dpk->A_g1_query);
    vector_device2host(&hpk->B_g1_query, &dpk->B_g1_query);
    vector_device2host(&hpk->B_g2_query, &dpk->B_g2_query);
    vector_device2host(&hpk->H_g1_query, &dpk->H_g1_query);
    vector_device2host(&hpk->L_g1_query, &dpk->L_g1_query);
}

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device1(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, cudaStream_t stream = 0)
{
    cudaMemcpy(&dpk->alpha_g1, &hpk->alpha_g1, sizeof(libff::G1<ppT_host>), cudaMemcpyHostToDevice);
    cudaMemcpy(&dpk->beta_g1, &hpk->beta_g1, sizeof(libff::G1<ppT_host>), cudaMemcpyHostToDevice);
    cudaMemcpy(&dpk->beta_g2, &hpk->beta_g2, sizeof(libff::G2<ppT_host>), cudaMemcpyHostToDevice);
    cudaMemcpy(&dpk->delta_g1, &hpk->delta_g1, sizeof(libff::G1<ppT_host>), cudaMemcpyHostToDevice);
    cudaMemcpy(&dpk->delta_g2, &hpk->delta_g2, sizeof(libff::G2<ppT_host>), cudaMemcpyHostToDevice);

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            dpk->alpha_g1.set_params(g1_instance->params);
            dpk->beta_g1.set_params(g1_instance->params);
            dpk->beta_g2.set_params(g2_instance->params);
            dpk->delta_g1.set_params(g1_instance->params);
            dpk->delta_g2.set_params(g2_instance->params);
        }
    );
    cudaStreamSynchronize(0);


    vector_host2device(&dpk->A_g1_query, &hpk->A_g1_query, stream);

    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize, 0, stream>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dpk->A_g1_query.size())
            {
                dpk->A_g1_query[idx].set_params(g1_instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
}

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device2(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, cudaStream_t stream = 0)
{
    vector_host2device(&dpk->B_g1_query, &hpk->B_g1_query, stream);

    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize, 0, stream>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dpk->B_g1_query.size())
            {
                dpk->B_g1_query[idx].set_params(g1_instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
}

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device3(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, cudaStream_t stream = 0)
{
    vector_host2device(&dpk->B_g2_query, &hpk->B_g2_query, stream);

    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize, 0, stream>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dpk->B_g2_query.size())
            {
                dpk->B_g2_query[idx].set_params(g2_instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
}

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device4(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, cudaStream_t stream = 0)
{
    vector_host2device(&dpk->H_g1_query, &hpk->H_g1_query, stream);

    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize, 0, stream>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dpk->H_g1_query.size())
            {
                dpk->H_g1_query[idx].set_params(g1_instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
}

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device5(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, cudaStream_t stream = 0)
{
    vector_host2device(&dpk->L_g1_query, &hpk->L_g1_query, stream);


    size_t gridSize = 512;
    size_t blockSize = 32;
    libstl::launch<<<gridSize, blockSize, 0, stream>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dpk->L_g1_query.size())
            {
                dpk->L_g1_query[idx].set_params(g1_instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
}

template<typename ppT_host, typename ppT_device>
void groth_proving_key_host2device(groth_proving_key<ppT_device>* dpk, groth_proving_key_host<ppT_host>* hpk,
                                            libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, cudaStream_t stream = 0)
{
    groth_proving_key_host2device1(dpk, hpk, g1_instance, g2_instance, stream);
    groth_proving_key_host2device2(dpk, hpk, g1_instance, g2_instance, stream);
    groth_proving_key_host2device3(dpk, hpk, g1_instance, g2_instance, stream);
    groth_proving_key_host2device4(dpk, hpk, g1_instance, g2_instance, stream);
    groth_proving_key_host2device5(dpk, hpk, g1_instance, g2_instance, stream);
}


template<typename ppT_host, typename ppT_device>
void groth_verification_key_device2host(groth_verification_key_host<ppT_host>* hvk, groth_verification_key<ppT_device>* dvk)
{
    cudaMemcpy(&hvk->alpha_g1_beta_g2, &dvk->alpha_g1_beta_g2, sizeof(libff::GT<ppT_device>), cudaMemcpyHostToDevice);
    cudaMemcpy(&hvk->gamma_g2, &dvk->gamma_g2, sizeof(libff::G2<ppT_device>), cudaMemcpyHostToDevice);
    cudaMemcpy(&hvk->delta_g2, &dvk->delta_g2, sizeof(libff::G2<ppT_device>), cudaMemcpyHostToDevice);

    vector_device2host(&hvk->gamma_ABC_g1, &dvk->gamma_ABC_g1);
}



template<typename ppT_host, typename ppT_device>
void groth_verification_key_host2device(groth_verification_key<ppT_device>* dvk, groth_verification_key_host<ppT_host>* hvk,
                                                libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, libff::GT<ppT_device>* gt_instance)
{
    cudaMemcpy(&dvk->alpha_g1_beta_g2, &hvk->alpha_g1_beta_g2, sizeof(libff::GT<ppT_host>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dvk->gamma_g2, &hvk->gamma_g2, sizeof(libff::G2<ppT_host>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dvk->delta_g2, &hvk->delta_g2, sizeof(libff::G2<ppT_host>), cudaMemcpyDeviceToHost);

    vector_host2device(&dvk->gamma_ABC_g1, &hvk->gamma_ABC_g1);
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            dvk->alpha_g1_beta_g2.set_params(gt_instance->params);
            dvk->gamma_g2.set_params(g2_instance->params);
            dvk->delta_g2.set_params(g2_instance->params);
        }
    );
    cudaDeviceSynchronize();

    size_t gridSize = 512;
    size_t blockSize = 32;

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < dvk->gamma_ABC_g1.size())
            {
                dvk->gamma_ABC_g1[idx].set_params(g1_instance->params);
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaDeviceSynchronize();
}

template<typename ppT_host, typename ppT_device>
void groth_keypair_device2host(groth_keypair_host<ppT_host>* hkp, groth_keypair<ppT_device>* dkp)
{
    groth_proving_key_device2host(&hkp->pk, &dkp->pk);
    groth_verification_key_device2host(&hkp->vk, &dkp->vk);
}

template<typename ppT_host, typename ppT_device>
void groth_keypair_host2device(groth_keypair<ppT_device>* dkp, groth_keypair_host<ppT_host>* hkp, 
                                    libff::G1<ppT_device>* g1_instance, libff::G2<ppT_device>* g2_instance, libff::GT<ppT_device>* gt_instance)
{
    groth_proving_key_host2device(&dkp->pk, &hkp->pk, g1_instance, g2_instance);
    groth_verification_key_host2device(&dkp->vk, &hkp->vk, g1_instance, g2_instance, gt_instance);
}

template<typename ppT>
void groth_proof_device2host(groth_proof<ppT>* hpf, groth_proof<ppT>* dpf)
{
    cudaMemcpy(&hpf->g_A, &dpf->g_A, sizeof(libff::G1<ppT>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hpf->g_B, &dpf->g_B, sizeof(libff::G2<ppT>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hpf->g_C, &dpf->g_C, sizeof(libff::G1<ppT>), cudaMemcpyDeviceToHost);
}


template<typename ppT>
void groth_proof_host2device(groth_proof<ppT>* dpf, groth_proof<ppT>* hpf, 
                                    libff::G1<ppT>* g1_instance, libff::G2<ppT>* g2_instance)
{
    cudaMemcpy(&dpf->g_A, &hpf->g_A, sizeof(libff::G1<ppT>), cudaMemcpyHostToDevice);
    cudaMemcpy(&dpf->g_B, &hpf->g_B, sizeof(libff::G2<ppT>), cudaMemcpyHostToDevice);
    cudaMemcpy(&dpf->g_C, &hpf->g_C, sizeof(libff::G1<ppT>), cudaMemcpyHostToDevice);

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            dpf->g_A.set_params(g1_instance->params);
            dpf->g_B.set_params(g2_instance->params);
            dpf->g_C.set_params(g1_instance->params);
        }
    );
    cudaDeviceSynchronize();
}

}

#endif
