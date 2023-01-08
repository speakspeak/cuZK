#ifndef __R1CS_TO_QAP_CU__
#define __R1CS_TO_QAP_CU__

#include "../../../depends/libfqfft-cuda/libfqfft/evaluation_domain/get_evaluation_domain.cuh"

namespace cuzk{

template<typename ppT>
__global__ void r1cs_to_qap_instance_map_with_evaluation_phase_one(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip)
{
    size_t gridSize = 512;
    size_t blockSize = 32;
    const libff::Fr<ppT>& instance = gp->t;

    const r1cs_constraint_system<libff::Fr<ppT>> &cs = rp->r1cs.constraint_system;
    size_t m = cs.num_constraints() + cs.num_inputs() + 1;
    if(m != 1ul << libff::log2(m))
    {
        size_t big = 1ul<<(libff::log2(m)-1);
        size_t small = m - big;
        size_t rounded_small = (1ul << libff::log2(small));
        if(rounded_small < gridSize * blockSize) rounded_small = gridSize * blockSize;
        m = big + rounded_small;
    }

    libff::Fr<ppT>* zero = libstl::create<libff::Fr<ppT>>(instance.zero());

    if(libfqfft::basic_radix2_domain<libff::Fr<ppT>>::valid(m))
    {
        libfqfft::basic_radix2_domain<libff::Fr<ppT>>* domain;
        domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>, libfqfft::basic_radix2_domain>(m, &ip->instance);
        gp->qap.Zt = domain->compute_vanishing_polynomial(gp->t);
        gp->qap.degree_ = domain->m;
        gp->qap.vHt.presize(domain->m + 1, *zero, 512, 32);
    }
    else if(libfqfft::step_radix2_domain<libff::Fr<ppT>>::valid(m))
    {
        libfqfft::step_radix2_domain<libff::Fr<ppT>>* domain;
        domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>, libfqfft::step_radix2_domain>(m, &ip->instance);
        gp->qap.Zt = domain->compute_vanishing_polynomial(gp->t);
        gp->qap.degree_ = domain->m;
        gp->qap.vHt.presize(domain->m + 1, *zero, 512, 32);
    }

    gp->qap.vAt.presize(rp->r1cs.constraint_system.num_variables()+1, *zero, 512, 32);
    gp->qap.vBt.presize(rp->r1cs.constraint_system.num_variables()+1, *zero, 512, 32);
    gp->qap.vCt.presize(rp->r1cs.constraint_system.num_variables()+1, *zero, 512, 32);
    // gp->qap.vHt.presize(domain->m + 1, *zero, 512, 32);

    // gp->qap.Zt = domain->compute_vanishing_polynomial(gp->t);

    gp->qap.num_variables_ = rp->r1cs.constraint_system.num_variables();
    // gp->qap.degree_ = domain->m;
    gp->qap.num_inputs_ = rp->r1cs.constraint_system.num_inputs();
}

template<typename ppT>
__global__ void r1cs_to_qap_instance_map_with_evaluation_phase_two(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip)
{
    size_t gridSize = 512;
    size_t blockSize = 32;
    const r1cs_constraint_system<libff::Fr<ppT>> &cs = rp->r1cs.constraint_system;
    size_t m = cs.num_constraints() + cs.num_inputs() + 1;
    if(m != 1ul << libff::log2(m))
    {
        size_t big = 1ul<<(libff::log2(m)-1);
        size_t small = m - big;
        size_t rounded_small = (1ul << libff::log2(small));
        if(rounded_small < gridSize * blockSize) rounded_small = gridSize * blockSize;
        m = big + rounded_small;
    }
    if(libfqfft::basic_radix2_domain<libff::Fr<ppT>>::valid(m))
    {
        libfqfft::basic_radix2_domain<libff::Fr<ppT>>* domain;
        domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>, libfqfft::basic_radix2_domain>(m, &ip->instance);
        gp->u = domain->p_evaluate_all_lagrange_polynomials(gp->t, 512, 32);
    }
    else if(libfqfft::step_radix2_domain<libff::Fr<ppT>>::valid(m))
    {
        libfqfft::step_radix2_domain<libff::Fr<ppT>>* domain;
        domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>, libfqfft::step_radix2_domain>(m, &ip->instance);
        gp->u = domain->p_evaluate_all_lagrange_polynomials(gp->t, 512, 32);
    }
}

template<typename ppT>
__global__ void r1cs_to_qap_instance_map_with_evaluation_phase_matrix(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip)
{
    size_t gridSize = 512;
    size_t blockSize = 32;
    const r1cs_constraint_system<libff::Fr<ppT>> &cs = rp->r1cs.constraint_system;
    size_t m = cs.num_constraints() + cs.num_inputs() + 1;
    if(m != 1ul << libff::log2(m))
    {
        size_t big = 1ul<<(libff::log2(m)-1);
        size_t small = m - big;
        size_t rounded_small = (1ul << libff::log2(small));
        if(rounded_small < gridSize * blockSize) rounded_small = gridSize * blockSize;
        m = big + rounded_small;
    }
    

    libff::Fr<ppT>* zero = libstl::create<libff::Fr<ppT>>(ip->instance.zero());
    libmatrix::CSR_matrix<libff::Fr<ppT>>* ma_T = libmatrix::p_transpose_csr2csr(cs.ma, 512, 32);
    libmatrix::CSR_matrix<libff::Fr<ppT>>* mb_T = libmatrix::p_transpose_csr2csr(cs.mb, 512, 32);
    libmatrix::CSR_matrix<libff::Fr<ppT>>* mc_T = libmatrix::p_transpose_csr2csr(cs.mc, 512, 32);
    gp->qap.vAt = *libmatrix::p_spmv_csr_balanced(*ma_T, gp->u, *zero, 512, 32);
    gp->qap.vBt = *libmatrix::p_spmv_csr_balanced(*mb_T, gp->u, *zero, 512, 32);
    gp->qap.vCt = *libmatrix::p_spmv_csr_balanced(*mc_T, gp->u, *zero, 512, 32);

    libstl::launch<<<512, 32>>>
    (
        [=, &cs]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx <= cs.num_inputs())
            {
                gp->qap.vAt[idx] += gp->u[cs.num_constraints() + idx];
                idx += blockDim.x * gridDim.x;
            }
        }
    );
    cudaDeviceSynchronize();

    if(libfqfft::basic_radix2_domain<libff::Fr<ppT>>::valid(m))
    {
        libfqfft::basic_radix2_domain<libff::Fr<ppT>>* domain;
        domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>, libfqfft::basic_radix2_domain>(m, &ip->instance);
        libstl::launch<<<512, 32>>>
        (
            [=]
            __device__ ()
            {
                size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                libff::Fr<ppT> ts = gp->t ^ (blockDim.x * gridDim.x);
                libff::Fr<ppT> tl = gp->t ^ tid;
                while(tid < domain->m + 1)
                {
                    gp->qap.vHt[tid] = tl;
                    tl *= ts;
                    tid += blockDim.x * gridDim.x;
                }
            }
        );
        cudaDeviceSynchronize();
    }
    else if(libfqfft::step_radix2_domain<libff::Fr<ppT>>::valid(m))
    {
        libfqfft::step_radix2_domain<libff::Fr<ppT>>* domain;
        domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>, libfqfft::step_radix2_domain>(m, &ip->instance);
        libstl::launch<<<512, 32>>>
        (
            [=]
            __device__ ()
            {
                size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                libff::Fr<ppT> ts = gp->t ^ (blockDim.x * gridDim.x);
                libff::Fr<ppT> tl = gp->t ^ tid;
                while(tid < domain->m + 1)
                {
                    gp->qap.vHt[tid] = tl;
                    tl *= ts;
                    tid += blockDim.x * gridDim.x;
                }
            }
        );
        cudaDeviceSynchronize();
    }


}

template<typename ppT>
void r1cs_to_qap_instance_map_with_evaluation(generator_params<ppT> *gp, r1cs_params<libff::Fr<ppT>>* rp, instance_params* ip)
{
    r1cs_to_qap_instance_map_with_evaluation_phase_one<ppT><<<1, 1>>>(gp, rp, ip);

    r1cs_to_qap_instance_map_with_evaluation_phase_two<ppT><<<1, 1>>>(gp, rp, ip);

    r1cs_to_qap_instance_map_with_evaluation_phase_matrix<ppT><<<1, 1>>>(gp, rp, ip);

}

template<typename FieldT>
qap_witness<FieldT>* r1cs_to_qap_witness_map(r1cs_constraint_system<FieldT>* cs,
                                            r1cs_primary_input<FieldT>* primary_input,
                                            r1cs_auxiliary_input<FieldT>* auxiliary_input,
                                            FieldT* instance)
{
    size_t gridSize = 512;
    size_t blockSize = 32;

    size_t m = cs->num_constraints_host() + cs->num_inputs_host() + 1;
    if(m != 1ul << libff::log2(m))
    {
        size_t big = 1ul<<(libff::log2(m)-1);
        size_t small = m - big;
        size_t rounded_small = (1ul << libff::log2(small));
        if(rounded_small < gridSize * blockSize) rounded_small = gridSize * blockSize;
        m = big + rounded_small;
    }

    size_t primary_input_size = primary_input->size_host();
    size_t auxiliary_input_size = auxiliary_input->size_host();
    size_t coefficients_for_H_size = m + 1;
    FieldT* zero = instance->zero_host();

    libstl::vector<FieldT>* vcoefficients_for_H = libstl::create_host<libstl::vector<FieldT>>();
    vcoefficients_for_H->presize_host(coefficients_for_H_size, zero, gridSize, blockSize);
    qap_witness<FieldT>* qap = libstl::create_host<qap_witness<FieldT>>();
    qap->vcoefficients_for_ABCs.presize_host(primary_input_size + auxiliary_input_size, gridSize, blockSize);

    size_t lockMem;
    libstl::lock_host(lockMem);

    libstl::vector<FieldT>* vaA = libstl::create_host<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* vaB = libstl::create_host<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* vaC = libstl::create_host<libstl::vector<FieldT>>();
    libstl::vector<FieldT>* full_variable_assignment = libstl::create_host<libstl::vector<FieldT>>();

    // cudaEvent_t eventQapPhase1Start, eventQapPhase1End;
    // cudaEventCreate( &eventQapPhase1Start);
	// cudaEventCreate( &eventQapPhase1End);
    // cudaEventRecord( eventQapPhase1Start, 0);
    // cudaEventSynchronize(eventQapPhase1Start);


    full_variable_assignment->presize_host(primary_input_size + auxiliary_input_size + 1, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx == 0) (*full_variable_assignment)[0] = instance->one();
            while(idx < primary_input_size + auxiliary_input_size)
            {
                if(idx < primary_input_size)
                {
                    (*full_variable_assignment)[idx + 1] = (*primary_input)[idx];
                }    
                else
                {
                    (*full_variable_assignment)[idx + 1] = (*auxiliary_input)[idx - primary_input_size];
                }
                idx += blockDim.x * gridDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);


    size_t * dm = libstl::create_host<size_t>();
    cudaMemcpy( dm , &m, sizeof(size_t), cudaMemcpyHostToDevice);
    FieldT** pinstance = libstl::create_host<FieldT*>();
    cudaMemcpy( pinstance , &instance, sizeof(FieldT*), cudaMemcpyHostToDevice);

    size_t aA_size = m;
    size_t aB_size = m;
    size_t aC_size = m;

    vaA->presize_host(aA_size, zero, gridSize, blockSize);
    vaB->presize_host(aB_size, zero, gridSize, blockSize);
    vaC->presize_host(aC_size, zero, gridSize, blockSize);
    

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            size_t input_size = cs->num_inputs();
            size_t cs_size = cs->num_constraints();
            while(idx <= input_size)
            {
                (*vaA)[idx+cs_size] = (*full_variable_assignment)[idx];
                idx += blockDim.x * gridDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);

    // cudaEventRecord( eventQapPhase1End, 0);
    // cudaEventSynchronize(eventQapPhase1End);
    // float   TimeQapPhase1;
    // cudaEventElapsedTime( &TimeQapPhase1, eventQapPhase1Start, eventQapPhase1End );
    // printf( "Time to QAPW 1:  %3.5f ms\n", TimeQapPhase1);


    cudaEvent_t eventQapPhase2Start, eventQapPhase2End;
    cudaEventCreate( &eventQapPhase2Start);
	cudaEventCreate( &eventQapPhase2End);
    cudaEventRecord( eventQapPhase2Start, 0); 
    cudaEventSynchronize(eventQapPhase2Start);

    // Here we can also choose csr_balance.
    libstl::vector<FieldT>* p_res_a = libmatrix::p_spmv_csr_scalar_host(cs->ma, *full_variable_assignment, *zero, gridSize, blockSize);
    libstl::vector<FieldT>* p_res_b = libmatrix::p_spmv_csr_scalar_host(cs->mb, *full_variable_assignment, *zero, gridSize, blockSize);
    libstl::vector<FieldT>* p_res_c = libmatrix::p_spmv_csr_scalar_host(cs->mc, *full_variable_assignment, *zero, gridSize, blockSize);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < cs->num_constraints())
            {
                (*vaA)[idx] += (*p_res_a)[idx];
                (*vaB)[idx] += (*p_res_b)[idx];
                (*vaC)[idx] += (*p_res_c)[idx];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);

    cudaEventRecord( eventQapPhase2End, 0);
    cudaEventSynchronize(eventQapPhase2End);
    float   TimeQapPhase2;
    cudaEventElapsedTime( &TimeQapPhase2, eventQapPhase2Start, eventQapPhase2End );
    printf( "Time to MUL %3.5f ms\n", TimeQapPhase2);


    cudaEvent_t eventQapPhase3Start, eventQapPhase3End;
    cudaEventCreate( &eventQapPhase3Start);
	cudaEventCreate( &eventQapPhase3End);
    cudaEventRecord( eventQapPhase3Start, 0); 
    cudaEventSynchronize(eventQapPhase3Start);

    FieldT* g = libstl::create_host<FieldT>();

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *g = FieldT(instance->params, *instance->params->multiplicative_generator);
        }
    );
    cudaStreamSynchronize(0);


    if(libfqfft::basic_radix2_domain<FieldT>::valid(m))
    {
        libfqfft::basic_radix2_domain<FieldT>* domain = libstl::create_host<libfqfft::basic_radix2_domain<FieldT>>();

        libstl::launch<<<1, 1>>>
        (
            [=]
            __device__ ()
            {
                *domain = *libfqfft::get_evaluation_domain<FieldT, libfqfft::basic_radix2_domain>(m, instance);
            }
        );
        cudaStreamSynchronize(0);


        domain->ppiFFT_host(*vaA, gridSize * blockSize);
        domain->ppiFFT_host(*vaB, gridSize * blockSize);

        
        domain->ppcosetFFT_host(*vaA, *g, gridSize * blockSize);
        domain->ppcosetFFT_host(*vaB, *g, gridSize * blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                size_t m = cs->num_constraints() + cs->num_inputs() + 1;
                while(idx < m)
                {
                    (*vaA)[idx] = (*vaA)[idx] * (*vaB)[idx];
                    idx += gridDim.x * blockDim.x;
                }
            }
        );
        cudaStreamSynchronize(0);

        domain->ppiFFT_host(*vaC, gridSize * blockSize);
        domain->ppcosetFFT_host(*vaC, *g, gridSize * blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                size_t m = cs->num_constraints() + cs->num_inputs() + 1;
                while(idx < m)
                {
                    (*vaA)[idx] = (*vaA)[idx] - (*vaC)[idx];
                    idx += gridDim.x * blockDim.x;
                }
            }
        );
        cudaStreamSynchronize(0);

        domain->p_divide_by_Z_on_coset_host(*vaA, gridSize, blockSize);
        domain->ppicosetFFT_host(*vaA, *g, gridSize * blockSize);
    }
    else if(libfqfft::step_radix2_domain<FieldT>::valid(m))
    {
        libfqfft::step_radix2_domain<FieldT>* domain = libstl::create_host<libfqfft::step_radix2_domain<FieldT>>();

        libstl::launch<<<1, 1>>>
        (
            [=]
            __device__ ()
            {
                *domain = *libfqfft::get_evaluation_domain<FieldT, libfqfft::step_radix2_domain>(m, instance);
            }
        );
        cudaStreamSynchronize(0);

        domain->ppiFFT_host(*vaA, gridSize * blockSize);
        domain->ppiFFT_host(*vaB, gridSize * blockSize);

        domain->ppcosetFFT_host(*vaA, *g, gridSize * blockSize);
        domain->ppcosetFFT_host(*vaB, *g, gridSize * blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                while(idx < m)
                {
                    (*vaA)[idx] = (*vaA)[idx] * (*vaB)[idx];
                    idx += gridDim.x * blockDim.x;
                }
            }
        );
        cudaStreamSynchronize(0);

        domain->ppiFFT_host(*vaC, gridSize * blockSize);
        domain->ppcosetFFT_host(*vaC, *g, gridSize * blockSize);

        libstl::launch<<<gridSize, blockSize>>>
        (
            [=]
            __device__ ()
            {
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                while(idx < m)
                {
                    (*vaA)[idx] = (*vaA)[idx] - (*vaC)[idx];
                    idx += gridDim.x * blockDim.x;
                }
            }
        );
        cudaStreamSynchronize(0);

        domain->p_divide_by_Z_on_coset_host(*vaA, gridSize, blockSize);
        domain->ppicosetFFT_host(*vaA, *g, gridSize * blockSize);
    }

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < m)
            {
                (*vcoefficients_for_H)[idx] += (*vaA)[idx];
                idx += gridDim.x * blockDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);

    cudaEventRecord( eventQapPhase3End, 0);
    cudaEventSynchronize(eventQapPhase3End);
    float   TimeQapPhase3;
    cudaEventElapsedTime( &TimeQapPhase3, eventQapPhase3Start, eventQapPhase3End );
    printf( "Time to FFT:  %3.5f ms\n", TimeQapPhase3);

    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            qap->num_variables_ = cs->num_variables();
            qap->degree_ = m;
            qap->num_inputs_ = cs->num_inputs();
            qap->d1 = instance->zero();
            qap->d2 = instance->zero();
            qap->d3 = instance->zero();

            qap->vcoefficients_for_H = libstl::move(*vcoefficients_for_H);
        }
    );
    cudaStreamSynchronize(0);


    libstl::launch<<<gridSize, blockSize>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            while(idx < primary_input_size + auxiliary_input_size)
            {
                if(idx < primary_input_size)
                {
                    qap->vcoefficients_for_ABCs[idx] = (*primary_input)[idx];
                }    
                else
                {
                    qap->vcoefficients_for_ABCs[idx] = (*auxiliary_input)[idx - primary_input_size];
                }
                idx += blockDim.x * gridDim.x;
            }
        }
    );
    cudaStreamSynchronize(0);


    libstl::resetlock_host(lockMem);

    return qap;
}

}


#endif

