struct instance_params;
struct h_instance_params;

template<typename FieldT>
struct r1cs_params;

template<typename ppT>
struct generator_params;

template<typename ppT>
struct prover_params;


#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND

//Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void * data){
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void end_thread(CUTThread thread){
    pthread_join(thread, NULL);
}

//Destroy thread
void destroy_thread( CUTThread thread ){
    pthread_cancel(thread);
}

//Wait for multiple threads
void wait_for_threads(const CUTThread * threads, int num){
    for(int i = 0; i < num; i++)
        end_thread( threads[i] );
}



#include <stdio.h>
#include <math.h>

#include "../depends/libstl-cuda/memory.cuh"
#include "../depends/libstl-cuda/vector.cuh"
#include "../depends/libstl-cuda/utility.cuh"

#include "../depends/libff-cuda/fields/bigint_host.cuh"
#include "../depends/libff-cuda/fields/fp_host.cuh"
#include "../depends/libff-cuda/fields/fp2_host.cuh"
#include "../depends/libff-cuda/fields/fp6_3over2_host.cuh"
#include "../depends/libff-cuda/fields/fp12_2over3over2_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-balanced.cuh"
#include "../depends/libmatrix-cuda/transpose/transpose_ell2csr.cuh"

#include "../cuzk/relations/variable.cuh"
#include "../cuzk/relations/constraint_satisfaction_problems/r1cs/r1cs.cuh"
#include "../cuzk/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.cuh"
#include "../cuzk/relations/arithmetic_programs/qap/qap.cuh"
#include "../cuzk/reductions/r1cs_to_qap/r1cs_to_qap.cuh"
#include "../cuzk/zk_proof_systems/groth/groth.cuh"

#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp.cuh"

#include <time.h>

using namespace libff;
using namespace cuzk;

__device__ static const mp_size_t_ n = 8;


struct instance_params
{
    bls12_381_Fr instance;
    bls12_381_G1 g1_instance;
    bls12_381_G2 g2_instance;
    bls12_381_GT gt_instance;
};

struct h_instance_params
{
    bls12_381_Fr_host h_instance;
    bls12_381_G1_host h_g1_instance;
    bls12_381_G2_host h_g2_instance;
    bls12_381_GT_host h_gt_instance;
};

template<typename FieldT>
struct r1cs_params
{
    r1cs_example<FieldT> r1cs;
};


template<typename ppT>
struct generator_params
{
    libff::Fr<ppT> t;
    libff::Fr<ppT> alpha;
    libff::Fr<ppT> beta;
    libff::Fr<ppT> gamma;
    libff::Fr<ppT> delta;
    libff::Fr<ppT> gamma_inverse;
    libff::Fr<ppT> delta_inverse;

    libff::G1<ppT> g1_generator;
    libff::G2<ppT> g2_generator;

    qap_instance_evaluation<libff::Fr<ppT>> qap;
    libstl::vector<libff::Fr<ppT>> u;

    libstl::vector<libff::Fr<ppT>> Lt;
    libff::Fr<ppT> gamma_ABC_0;
    libstl::vector<libff::Fr<ppT>> gamma_ABC;

    libff::window_table<libff::G1<ppT>> g1_table;
    libff::window_table<libff::G2<ppT>> g2_table;

    libff::G1<ppT> alpha_g1;
    libff::G1<ppT> beta_g1;
    libff::G2<ppT> beta_g2;
    libff::G1<ppT> delta_g1;
    libff::G2<ppT> delta_g2;
    
    libstl::vector<libff::G1<ppT>> A_g1_query;
    libstl::vector<libff::G1<ppT>> B_g1_query;
    libstl::vector<libff::G2<ppT>> B_g2_query;
    libstl::vector<libff::G1<ppT>> H_g1_query;
    libstl::vector<libff::G1<ppT>> L_g1_query;
    libstl::vector<libff::G1<ppT>> gamma_ABC_g1_values;

    libff::Fr<ppT> Zt_delta_inverse;

    libff::GT<ppT> alpha_g1_beta_g2;
    libff::G2<ppT> gamma_g2; 
    libff::G1<ppT> gamma_ABC_g1_0;
    libstl::vector<libff::G1<ppT>> gamma_ABC_g1;

    groth_keypair<ppT> kp;
};

template<typename ppT>
struct prover_params
{
    libff::Fr<ppT> d1;
    libff::Fr<ppT> d2; 
    libff::Fr<ppT> d3;

    libfqfft::basic_radix2_domain<libff::Fr<ppT>>* domain;
    
    libstl::vector<libff::Fr<ppT>> full_variable_assignment;
    libstl::vector<libff::Fr<ppT>> full_variable_assignment_without_one;
    size_t primary_input_size;
    size_t auxiliary_input_size;
    libstl::vector<libff::Fr<ppT>> vaA;
    libstl::vector<libff::Fr<ppT>> vaB;
    libstl::vector<libff::Fr<ppT>> vaC;
    libstl::vector<libff::Fr<ppT>> vcoefficients_for_H;

    qap_witness<libff::Fr<ppT> > qap_wit;

    libff::Fr<ppT> r;
    libff::Fr<ppT> s;

    libstl::vector<libff::Fr<ppT>> const_padded_assignment;


    libstl::vector<libff::Fr<ppT>> const_padded_assignment_in;
    libstl::vector<libff::Fr<ppT>> Ht_g1_coefficients_for_H;
    
    libstl::vector<libff::G1<ppT>> At_g1_query;
    libstl::vector<libff::G1<ppT>> Bt_g1_query;
    libstl::vector<libff::G2<ppT>> Bt_g2_query;
    libstl::vector<libff::G1<ppT>> Ht_g1_query;

    libstl::vector<libff::G1<ppT>> evaluation_At_g1_mid_res;
    libstl::vector<libff::G1<ppT>> evaluation_Bt_g1_mid_res;
    libstl::vector<libff::G2<ppT>> evaluation_Bt_g2_mid_res;
    libstl::vector<libff::G1<ppT>> evaluation_Ht_g1_mid_res;
    libstl::vector<libff::G1<ppT>> evaluation_Lt_g1_mid_res;

    libff::G1<ppT> evaluation_At_g1;
    libff::G1<ppT> evaluation_Bt_g1;
    libff::G2<ppT> evaluation_Bt_g2;
    libff::G1<ppT> evaluation_Ht_g1;
    libff::G1<ppT> evaluation_Lt_g1;

    libstl::vector<libff::Fr<ppT>> const_padded_assignment_wit;

    libff::G1<ppT> g1_A;
    libff::G1<ppT> g1_B;
    libff::G2<ppT> g2_B;
    libff::G1<ppT> g1_C;
};

__global__ void init_params()
{
    gmp_init_allocator_();
    bls12_381_pp::init_public_params();
}

__global__ void instance_init(instance_params* ip)
{
    ip->instance = bls12_381_Fr(&bls12_381_fp_params_r);
    ip->g1_instance = bls12_381_G1(&g1_params);
    ip->g2_instance = bls12_381_G2(&g2_params);
    ip->gt_instance = bls12_381_GT(&bls12_381_fp12_params_q);
}

void instance_init_host(h_instance_params* ip)
{
    ip->h_instance = bls12_381_Fr_host(&bls12_381_fp_params_r_host);
    ip->h_g1_instance = bls12_381_G1_host(&g1_params_host);
    ip->h_g2_instance = bls12_381_G2_host(&g2_params_host);
    ip->h_gt_instance = bls12_381_GT_host(&bls12_381_fp12_params_q_host);
}

template<typename ppT>
__global__ void generator_init(generator_params<ppT>* gp)
{
    new ((void*)gp) generator_params<ppT>();
}

template<typename FieldT>
__global__ void generate_r1cs_construction(r1cs_params<bls12_381_Fr>* rp, const instance_params* ip, size_t log_size)
{
    // rp->r1cs = generate_r1cs_example_with_field_input(10, 5, ip->instance);
    // rp->r1cs = generate_r1cs_example_with_field_input(100, 27, ip->instance);
    // rp->r1cs = generate_r1cs_example_with_field_input(1000, 23, ip->instance);
    // rp->r1cs = generate_r1cs_example_with_field_input(10000, 6383, ip->instance);
    // rp->r1cs = generate_r1cs_example_with_field_input(100000, 31071, ip->instance);
    // rp->r1cs = generate_r1cs_example_with_field_input(1000000, 48575, ip->instance);
    // rp->r1cs = generate_r1cs_example_with_field_input(4000000, 194303, ip->instance);

    rp->r1cs = generate_r1cs_example_like_bellperson(1ul << log_size, ip->instance);
}

struct Mem
{
    size_t device_id;
    void* mem;
    size_t init_size;
};

void* multi_init_params(void* params)
{
    Mem* device_mem = (Mem*) params;
    cudaSetDevice(device_mem->device_id);
    size_t init_size = 1024 * 1024 * 1024;
    init_size *= device_mem->init_size;
    if( cudaMalloc( (void**)&device_mem->mem, init_size ) != cudaSuccess) printf("device malloc error!\n");
    libstl::initAllocator(device_mem->mem, init_size);
    init_params<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

struct Instance
{
    size_t device_id;
    instance_params** ip;
};


void* multi_instance_init(void* instance)
{
    Instance* it = (Instance*)instance;
    cudaSetDevice(it->device_id);
    if( cudaMalloc( (void**)it->ip, sizeof(instance_params)) != cudaSuccess) printf("ip malloc error!\n");
    instance_init<<<1, 1>>>(*it->ip);
    cudaDeviceSynchronize();
    return 0;
}


int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Please enter the number of constraints (e.g. 20 represents 2^20) \n");
		return 1;
	}

    int log_size = atoi(argv[1]);
    int init_size = 20;

    int deviceCount;
    cudaGetDeviceCount( &deviceCount );
    CUTThread  thread[deviceCount];

    bls12_381_pp_host::init_public_params();
    size_t lockMem[deviceCount];

    cudaSetDevice(0);

    // params init 
    Mem device_mem[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        device_mem[i].device_id = i;
        device_mem[i].mem = NULL;
        device_mem[i].init_size = init_size;
        thread[i] = start_thread( multi_init_params, &device_mem[i] );
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    // instance init
    instance_params* ip[deviceCount];
    Instance instance[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        instance[i].device_id = i;
        instance[i].ip = &ip[i];
        thread[i] = start_thread( multi_instance_init, &instance[i] );

    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    h_instance_params hip;
    instance_init_host(&hip);

    for(size_t i=0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        libstl::lock_host(lockMem[i]);
    }
    cudaSetDevice(0);

    // r1cs construction
    r1cs_params<bls12_381_Fr>* rp;
    if( cudaMalloc( (void**)&rp, sizeof(r1cs_params<bls12_381_Fr>)) != cudaSuccess) printf("rp malloc error!\n");

    generate_r1cs_construction<bls12_381_Fr><<<1, 1>>>(rp, ip[0], log_size);

    r1cs_params<bls12_381_Fr>* rpt[deviceCount];

    for(size_t i=1; i<deviceCount; i++)
    {
        cudaSetDevice(i);
        if( cudaMalloc( (void**)&rpt[i], sizeof(r1cs_params<bls12_381_Fr>)) != cudaSuccess) printf("rp malloc error!\n");
        generate_r1cs_construction<bls12_381_Fr><<<1, 1>>>(rpt[i], ip[i], log_size);
    }

    cudaSetDevice(0);

    // generator
    generator_params<bls12_381_pp>* gp;
    if( cudaMalloc( (void**)&gp, sizeof(generator_params<bls12_381_pp>)) != cudaSuccess) printf("gp malloc error!\n");
    generator_init<<<1, 1>>>(gp);
    cudaDeviceSynchronize();
    groth_generator<bls12_381_pp>(gp, rp, ip[0]);

    generator_params<bls12_381_pp>* gpt[deviceCount];
    for(size_t i=1; i<deviceCount; i++)
    {
        cudaSetDevice(i);
        if( cudaMalloc( (void**)&gpt[i], sizeof(generator_params<bls12_381_pp>)) != cudaSuccess) printf("gp malloc error!\n");
        generator_init<<<1, 1>>>(gpt[i]);
        cudaDeviceSynchronize();
        groth_generator<bls12_381_pp>(gpt[i], rpt[i], ip[i]);
    }

    cudaSetDevice(0);


    groth_keypair_host<bls12_381_pp_host> hkp;
    groth_keypair_device2host<bls12_381_pp_host, bls12_381_pp>(&hkp, &gp->kp);
    r1cs_constraint_system_host<bls12_381_Fr_host> hcs;
    r1cs_constraint_system_device2host<bls12_381_Fr_host, bls12_381_Fr>(&hcs, &rp->r1cs.constraint_system);
    r1cs_primary_input_host<bls12_381_Fr_host> hpi;
    r1cs_primary_input_device2host<bls12_381_Fr_host, bls12_381_Fr>(&hpi, &rp->r1cs.r1cs_primary_input);
    r1cs_auxiliary_input_host<bls12_381_Fr_host> hai;
    r1cs_auxiliary_input_device2host<bls12_381_Fr_host, bls12_381_Fr>(&hai, &rp->r1cs.r1cs_auxiliary_input);
    
    for(size_t i=0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        libstl::presetlock_host(lockMem[i], 512, 32);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);

    cudaEvent_t eventProveStart, eventProveEnd;
    cudaEventCreate( &eventProveStart);
	cudaEventCreate( &eventProveEnd);
    cudaEventRecord( eventProveStart, 0); 
    cudaEventSynchronize(eventProveStart);

    groth_prover<bls12_381_pp_host, bls12_381_pp>(&hcs, &hkp.pk, &hpi, &hai, ip, &hip);

    cudaEventRecord( eventProveEnd, 0);
    cudaEventSynchronize(eventProveEnd);
    float   TimeProve;
    cudaEventElapsedTime( &TimeProve, eventProveStart, eventProveEnd );
    printf( "Time to Prove:  %3.5f ms\n", TimeProve );

    for(size_t i=0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        libstl::presetlock_host(lockMem[i], 512, 32);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);

    // prove
    cudaEvent_t eventProveFStart, eventProveFEnd;
    cudaEventCreate( &eventProveFStart);
	cudaEventCreate( &eventProveFEnd);
    cudaEventRecord( eventProveFStart, 0); 
    cudaEventSynchronize(eventProveFStart);

    groth_prover<bls12_381_pp_host, bls12_381_pp>(&hcs, &hkp.pk, &hpi, &hai, ip, &hip);

    cudaEventRecord( eventProveFEnd, 0);
    cudaEventSynchronize(eventProveFEnd);
    float   TimeProveF;
    cudaEventElapsedTime( &TimeProveF, eventProveFStart, eventProveFEnd );
    printf( "Time to Prove:  %3.5f ms\n", TimeProveF );

    cudaDeviceReset();
    return 0;
}
