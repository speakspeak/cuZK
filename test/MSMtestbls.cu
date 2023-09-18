struct instance_params;
struct h_instance_params;

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
#include <string.h>

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
#include "../depends/libmatrix-cuda/transpose/transpose_ell2csr.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-balanced.cuh"
#include "../depends/libff-cuda/scalar_multiplication/multiexp.cuh"


#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp.cuh"

#include <time.h>

#include <nvml.h>

using namespace libff;

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


template<typename ppT>
struct MSM_params
{
    libstl::vector<libff::Fr<ppT>> vf;
    libstl::vector<libff::G1<ppT>> vg;
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
__global__ void generate_MP(MSM_params<ppT>* mp, instance_params* ip, size_t size)
{
    new ((void*)mp) MSM_params<ppT>();
    mp->vf.presize(size, 512, 32);
    mp->vg.presize(size, 512, 32);

    libstl::launch<<<512, 32>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            libff::Fr<ppT> f = ip->instance.random_element();
            libff::G1<ppT> g = ip->g1_instance.random_element();
            f ^= idx;
            g = g * idx;
            while(idx < size)
            {
                mp->vf[idx] = f;
                mp->vg[idx] = g;
                f = f + f;
                g = g + g;
                idx += tnum;
            }
        }
    );
    cudaDeviceSynchronize();

    ip->g1_instance.p_batch_to_special(mp->vg, 160, 32);
}

struct Mem
{
    size_t device_id;
    void* mem;
};

void* multi_init_params(void* params)
{
    Mem* device_mem = (Mem*) params;
    cudaSetDevice(device_mem->device_id);
    size_t init_size = 1024 * 1024 * 1024;
    init_size *= 5;
    if( cudaMalloc( (void**)&device_mem->mem, init_size ) != cudaSuccess) printf("device malloc error!");
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

template<typename ppT>
struct MSM
{
    size_t device_id;
    MSM_params<ppT>* mp;
    instance_params* ip;
    libff::G1<ppT>* res;
};

template<typename ppT>
void* multi_MSM(void* msm)
{
    MSM<ppT>* it = (MSM<ppT>*)msm;
    cudaSetDevice(it->device_id);

    size_t lockMem;
    libstl::lock_host(lockMem);
    printf( "warming up (1).. \n");
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    printf( "warming up (2).. \n");
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    printf( "warming up (3).. \n");
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libstl::resetlock_host(lockMem);


    printf( "testing.. \n");
    unsigned long long energy_start, energy_end;
    nvmlDevice_t device;
    nvmlInit();
    nvmlDeviceGetHandleByIndex(it->device_id, &device);
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);

    cudaEvent_t eventMSMStart, eventMSMEnd;
    cudaEventCreate( &eventMSMStart);
    cudaEventCreate( &eventMSMEnd);
    cudaEventRecord( eventMSMStart, 0);
    cudaEventSynchronize(eventMSMStart);

    for(size_t i=0; i<1; i++)
    {
        it->res = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
        cudaDeviceSynchronize();
    }

    cudaEventRecord( eventMSMEnd, 0);
    cudaEventSynchronize(eventMSMEnd);
    float   TimeMSM;
    cudaEventElapsedTime( &TimeMSM, eventMSMStart, eventMSMEnd );
    unsigned long long EnergyMSM;
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_end);
    EnergyMSM = energy_end - energy_start;
    printf( "Time thread %lu for MSM:  %3.5f ms\n, Energy: %llu mJ", it->device_id, TimeMSM, EnergyMSM );
    nvmlShutdown();
    return 0;
}

template<typename ppT_host, typename ppT_device>
void D2H(libff::G1<ppT_host>* hg1, libff::G1<ppT_device>* dg1, libff::G1<ppT_host>* g1_instance)
{
    cudaMemcpy(hg1, dg1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    hg1->set_params(g1_instance->params);
}


template<typename ppT>
void Reduce(libff::G1<ppT>* hg1, libff::Fr<ppT>* instance, size_t total)
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    libff::G1<ppT> g1 = hg1[device_count-1];

    if(device_count != 1)
    {
        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup > num_groups) sgroup = num_groups;
            if(egroup == sgroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                g1 = g1.dbl();
            }
            g1 = g1 + hg1[i];
        }
    }

    g1.to_special();

}

int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
		return 1;
	}

    int log_size = atoi(argv[1]);

    int deviceCount;
    cudaGetDeviceCount( &deviceCount );
    CUTThread  thread[deviceCount];
    printf( "We have %d devices \n", deviceCount);

    bls12_381_pp_host::init_public_params();
    printf( "Start on device=0 \n");
    cudaSetDevice(0);

    size_t num_v = (size_t) (1 << log_size);

    // params init 
    printf( "Init params... \n");
    Mem device_mem[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        device_mem[i].device_id = i;
        device_mem[i].mem = NULL;
        thread[i] = start_thread( multi_init_params, &device_mem[i] );
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    // instance init
    printf( "Init instances... \n");
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

    // elements generation
    MSM_params<bls12_381_pp>* mp[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        printf( "Device %d: Generate elements... \n", i);
        cudaSetDevice(i);
        if( cudaMalloc( (void**)&mp[i], sizeof(MSM_params<bls12_381_pp>)) != cudaSuccess) printf("mp malloc error!\n");
        generate_MP<bls12_381_pp><<<1, 1>>>(mp[i], ip[i], num_v);
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
    
    // msm
    MSM<bls12_381_pp> msm[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        printf( "Device %d: Computing MSM... \n", i);
        msm[i].device_id = i;
        msm[i].mp = mp[i];
        msm[i].ip = ip[i];
        thread[i] = start_thread( multi_MSM<bls12_381_pp>, &msm[i] );
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    // reduction on host

    libff::G1<bls12_381_pp_host> hg1[deviceCount];
    for(size_t i=0; i < deviceCount; i++)
    {
        printf( "Device %d: Transfering to Host... \n", i);
        cudaSetDevice(i);
        D2H<bls12_381_pp_host, bls12_381_pp>(&hg1[i], msm[i].res, &hip.h_g1_instance);
    }

    printf( "Host: Reduction... \n");
    Reduce<bls12_381_pp_host>(hg1, &hip.h_instance, num_v);

    cudaDeviceReset();
    return 0;
}
