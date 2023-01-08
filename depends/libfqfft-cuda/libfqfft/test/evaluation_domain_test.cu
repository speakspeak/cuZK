// #include "../../depends/libff-cuda/scalar_multiplication/wnaf.cuh"
// #include "../../depends/libff-cuda/scalar_multiplication/multiexp.cuh"
#include "../../depends/libff-cuda/curves/alt_bn128/alt_bn128_pairing.cuh"
#include "../../depends/libff-cuda/curves/alt_bn128/alt_bn128_pp.cuh"
#include "../../depends/libff-cuda/curves/alt_bn128/alt_bn128_g2.cuh"
#include "../../depends/libff-cuda/curves/alt_bn128/alt_bn128_g1.cuh"
#include "../../depends/libff-cuda/curves/alt_bn128/alt_bn128_init.cuh"
#include "../../depends/libff-cuda/fields/bigint.cuh"
#include "../../depends/libff-cuda/fields/fp.cuh"
#include "../../depends/libff-cuda/fields/fp2.cuh"
// #include "../../depends/libff-cuda/fields/fp3.cuh"
// #include "../../depends/libff-cuda/fields/fp4.cuh"
// #include "../../depends/libff-cuda/fields/fp6_2over3.cuh"
#include "../../depends/libff-cuda/fields/fp6_3over2.cuh"
#include "../../depends/libff-cuda/fields/fp12_2over3over2.cuh"
#include "../evaluation_domain/get_evaluation_domain.cuh"
#include "../../depends/libff-cuda/common/utils.cuh"
#include "../polynomial_arithmetic/naive_evaluate.cuh"
#include <stdio.h>
// #include <math.h>

using namespace libstl;
using namespace libff;
using namespace libfqfft;

__device__ static const mp_size_t_ n = 8;

__device__ Fp_params<n> fp_params;
__device__ Fp2_params<n> fp2_params;
__device__ Fp6_3over2_params<n> fp6_params;
__device__ Fp12_params<n> fp12_params;

__device__ void init()
{
    initAllocator(100000);

    gmp_init_allocator_();

    bigint<n>* modulus= new bigint<n>("21888242871839275222246405745257275088548364400416034343698204186575808495617");    
    bigint<n>* euler = new bigint<n>("10944121435919637611123202872628637544274182200208017171849102093287904247808");
    bigint<n>* t = new bigint<n>("81540058820840996586704275553141814055101440848469862132140264610111");
    bigint<n>* t_minus_1_over_2 = new bigint<n>("40770029410420498293352137776570907027550720424234931066070132305055");
    bigint<n>* nqr = new bigint<n>("5");
    bigint<n>* nqr_to_t = new bigint<n>("19103219067921713944291392827692070036145651957329286315305642004821462161904");
    bigint<n>* multiplicative_generator = new bigint<n>("5");
    bigint<n>* root_of_unity = new bigint<n>("19103219067921713944291392827692070036145651957329286315305642004821462161904");
    bigint<n>* Rsquared = new bigint<n>("944936681149208446651664254269745548490766851729442924617792859073125903783");
    bigint<n>* Rcubed = new bigint<n>("5866548545943845227489894872040244720403868105578784105281690076696998248512");
        
    fp_params.modulus = modulus;    //
    fp_params.euler = euler;
    fp_params.t = t;
    fp_params.t_minus_1_over_2 = t_minus_1_over_2;
    fp_params.nqr = nqr;
    fp_params.nqr_to_t = nqr_to_t;
    fp_params.multiplicative_generator = multiplicative_generator;
    fp_params.root_of_unity = root_of_unity;
    fp_params.Rsquared = Rsquared;
    fp_params.Rcubed = Rcubed;

    fp_params.num_limbs = 8;
    fp_params.num_bits = 254;
    fp_params.s = 28;
    fp_params.inv = 0xefffffff;
}


__device__ void testpFFT()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    const size_t m = 4;

    vector<alt_bn128_Fr> f(m, instance);
    f[0] = alt_bn128_Fr(instance.params, 2);
    f[1] = alt_bn128_Fr(instance.params, 5);
    f[2] = alt_bn128_Fr(instance.params, 3);
    f[3] = alt_bn128_Fr(instance.params, 8);

    basic_radix2_domain<alt_bn128_Fr>* domain = get_evaluation_domain<alt_bn128_Fr, basic_radix2_domain>(m, instance);

    vector<alt_bn128_Fr>* a = (vector<alt_bn128_Fr>*)allocate(sizeof(vector<alt_bn128_Fr>));
    construct(a, f);
    domain->pFFT(*a, 2, 1);

    vector<alt_bn128_Fr> b(f);
    domain->FFT(b);

    for (size_t i = 0; i < m; i++)
    {
         if (b[i] == (*a)[i])
             printf("testpFFT OK %d\n", i);
         else
             printf("testpFFT not OK %d\n", i);
    }

    delete domain;
}

__device__ void testpiFFT()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    const size_t m = 4;

    vector<alt_bn128_Fr> f(m, instance);
    f[0] = alt_bn128_Fr(instance.params, 2);
    f[1] = alt_bn128_Fr(instance.params, 5);
    f[2] = alt_bn128_Fr(instance.params, 3);
    f[3] = alt_bn128_Fr(instance.params, 8);

    basic_radix2_domain<alt_bn128_Fr>* domain = get_evaluation_domain<alt_bn128_Fr, basic_radix2_domain>(m, instance);

    vector<alt_bn128_Fr>* a = (vector<alt_bn128_Fr>*)allocate(sizeof(vector<alt_bn128_Fr>));
    construct(a, f);
    domain->piFFT(*a, 2, 2);

    vector<alt_bn128_Fr> b(f);
    domain->iFFT(b);

    for (size_t i = 0; i < m; i++)
    {
        if (b[i] == (*a)[i])
            printf("testpiFFT OK %d\n", i);
        else
            printf("testpiFFT not OK %d\n", i);
    }

    delete domain;
}

__device__ void testpCosetFFT()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    const size_t m = 4;

    vector<alt_bn128_Fr> f(m, instance);
    f[0] = alt_bn128_Fr(instance.params, 2);
    f[1] = alt_bn128_Fr(instance.params, 5);
    f[2] = alt_bn128_Fr(instance.params, 3);
    f[3] = alt_bn128_Fr(instance.params, 8);

    basic_radix2_domain<alt_bn128_Fr>* domain = get_evaluation_domain<alt_bn128_Fr, basic_radix2_domain>(m, instance);

    alt_bn128_Fr* coset = (alt_bn128_Fr*)allocate(sizeof(alt_bn128_Fr));
    construct(coset, instance.params, *instance.params->multiplicative_generator);
    
    vector<alt_bn128_Fr>* a = (vector<alt_bn128_Fr>*)allocate(sizeof(vector<alt_bn128_Fr>));
    construct(a, f);

    domain->pcosetFFT(*a, *coset, 1, 2);

    vector<alt_bn128_Fr> b(f);
    domain->cosetFFT(b, *coset);

    for (size_t i = 0; i < m; i++)
    {
        if (b[i] == (*a)[i])
            printf("testpCosetFFT OK %d\n", i);
        else
            printf("testpCosetFFT not OK %d\n", i);
    }

    delete domain;
}

__device__ void testpiCosetFFT()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    const size_t m = 4;

    vector<alt_bn128_Fr> f(m, instance);
    f[0] = alt_bn128_Fr(instance.params, 2);
    f[1] = alt_bn128_Fr(instance.params, 5);
    f[2] = alt_bn128_Fr(instance.params, 3);
    f[3] = alt_bn128_Fr(instance.params, 8);

    basic_radix2_domain<alt_bn128_Fr>* domain = get_evaluation_domain<alt_bn128_Fr, basic_radix2_domain>(m, instance);

    alt_bn128_Fr* coset = (alt_bn128_Fr*)allocate(sizeof(alt_bn128_Fr));
    construct(coset, instance.params, *instance.params->multiplicative_generator);
    
    vector<alt_bn128_Fr>* a = (vector<alt_bn128_Fr>*)allocate(sizeof(vector<alt_bn128_Fr>));
    construct(a, f);

    domain->picosetFFT(*a, *coset, 1, 2);

    vector<alt_bn128_Fr> b(f);
    domain->icosetFFT(b, *coset);

    for (size_t i = 0; i < m; i++)
    {
        if (b[i] == (*a)[i])
            printf("testpiCosetFFT OK %d\n", i);
        else
            printf("testpiCosetFFT not OK %d\n", i);
    }

    delete domain;
}

__device__ void testLagrangeCoefficients()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    const size_t m = 8;

    alt_bn128_Fr t = alt_bn128_Fr(instance.params, 10);

    basic_radix2_domain<alt_bn128_Fr>* domain = get_evaluation_domain<alt_bn128_Fr, basic_radix2_domain>(m, instance);
    vector<alt_bn128_Fr> a = domain->evaluate_all_lagrange_polynomials(t);

    vector<alt_bn128_Fr> d(m, instance);
    for (size_t i = 0; i < m; i++)
        d[i] = domain->get_domain_element(i);

    for (size_t i = 0; i < m; i++)
    {
        alt_bn128_Fr e = evaluate_lagrange_polynomial(m, d, t, i);
        if (e == a[i])
            printf("testLagrangeCoefficients OK %d\n", i);
        else
            printf("testLagrangeCoefficients not OK %d\n", i);
    }

    delete domain;
}

__device__ void testComputeZ()
{   
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    const size_t m = 8;

    alt_bn128_Fr t = alt_bn128_Fr(instance.params, 10);

    basic_radix2_domain<alt_bn128_Fr>* domain = get_evaluation_domain<alt_bn128_Fr, basic_radix2_domain>(m, instance);

    alt_bn128_Fr a = domain->compute_vanishing_polynomial(t);
    alt_bn128_Fr Z = instance.one();

    for (size_t i = 0; i < m; i++)
        Z *= (t - domain->get_domain_element(i));

    if (Z == a)
        printf("testComputeZ OK\n");
    else
        printf("testComputeZ not OK\n");

    delete domain;
}

__global__ void test(void) {
    init();

    alt_bn128_pp::init_public_params();

    printf("initOK\n");

    testpFFT();
    testpiFFT();
    testpCosetFFT();
    testpiCosetFFT();

    //testLagrangeCoefficients();
    //testComputeZ();
}

int main()
{
    test<<<1, 1>>>();

    cudaDeviceReset();
}