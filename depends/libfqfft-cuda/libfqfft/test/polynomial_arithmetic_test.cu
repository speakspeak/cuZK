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
#include "../../depends/libff-cuda/common/utils.cuh"
#include "../polynomial_arithmetic/basic_operations.cuh"
#include "../polynomial_arithmetic/xgcd.cuh"
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

__device__ void testPolynomialAdditionSame()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(8, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);
    a[5] = alt_bn128_Fr(instance.params, 7);
    a[6] = alt_bn128_Fr(instance.params, 7);
    a[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> b(8, instance);
    b[0] = alt_bn128_Fr(instance.params, 9);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 11);
    b[3] = alt_bn128_Fr(instance.params, 14);
    b[4] = alt_bn128_Fr(instance.params, 7);
    b[5] = alt_bn128_Fr(instance.params, 1);
    b[6] = alt_bn128_Fr(instance.params, 5);
    b[7] = alt_bn128_Fr(instance.params, 8);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_addition(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, 10);
    c_ans[1] = alt_bn128_Fr(instance.params, 6);
    c_ans[2] = alt_bn128_Fr(instance.params, 15);
    c_ans[3] = alt_bn128_Fr(instance.params, 39);
    c_ans[4] = alt_bn128_Fr(instance.params, 13);
    c_ans[5] = alt_bn128_Fr(instance.params, 8);
    c_ans[6] = alt_bn128_Fr(instance.params, 12);
    c_ans[7] = alt_bn128_Fr(instance.params, 10);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialAdditionSame OK %d\n", i);
        else
            printf("testPolynomialAdditionSame not OK %d\n", i);
    }
}

__device__ void testPolynomialAdditionBiggerA()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(8, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);
    a[5] = alt_bn128_Fr(instance.params, 7);
    a[6] = alt_bn128_Fr(instance.params, 7);
    a[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> b(5, instance);
    b[0] = alt_bn128_Fr(instance.params, 9);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 11);
    b[3] = alt_bn128_Fr(instance.params, 14);
    b[4] = alt_bn128_Fr(instance.params, 7);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_addition(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, 10);
    c_ans[1] = alt_bn128_Fr(instance.params, 6);
    c_ans[2] = alt_bn128_Fr(instance.params, 15);
    c_ans[3] = alt_bn128_Fr(instance.params, 39);
    c_ans[4] = alt_bn128_Fr(instance.params, 13);
    c_ans[5] = alt_bn128_Fr(instance.params, 7);
    c_ans[6] = alt_bn128_Fr(instance.params, 7);
    c_ans[7] = alt_bn128_Fr(instance.params, 2);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialAdditionBiggerA OK %d\n", i);
        else
            printf("testPolynomialAdditionBiggerA not OK %d\n", i);
    }
}

__device__ void testPolynomialAdditionBiggerB()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(5, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);

    vector<alt_bn128_Fr> b(8, instance);
    b[0] = alt_bn128_Fr(instance.params, 9);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 11);
    b[3] = alt_bn128_Fr(instance.params, 14);
    b[4] = alt_bn128_Fr(instance.params, 7);
    b[5] = alt_bn128_Fr(instance.params, 1);
    b[6] = alt_bn128_Fr(instance.params, 5);
    b[7] = alt_bn128_Fr(instance.params, 8);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_addition(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, 10);
    c_ans[1] = alt_bn128_Fr(instance.params, 6);
    c_ans[2] = alt_bn128_Fr(instance.params, 15);
    c_ans[3] = alt_bn128_Fr(instance.params, 39);
    c_ans[4] = alt_bn128_Fr(instance.params, 13);
    c_ans[5] = alt_bn128_Fr(instance.params, 1);
    c_ans[6] = alt_bn128_Fr(instance.params, 5);
    c_ans[7] = alt_bn128_Fr(instance.params, 8);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialAdditionBiggerB OK %d\n", i);
        else
            printf("testPolynomialAdditionBiggerB not OK %d\n", i);
    }
}

__device__ void testPolynomialAdditionZeroA()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(3, instance.zero());

    vector<alt_bn128_Fr> b(8, instance);
    b[0] = alt_bn128_Fr(instance.params, 1);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 4);
    b[3] = alt_bn128_Fr(instance.params, 25);
    b[4] = alt_bn128_Fr(instance.params, 6);
    b[5] = alt_bn128_Fr(instance.params, 7);
    b[6] = alt_bn128_Fr(instance.params, 7);
    b[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_addition(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, 1);
    c_ans[1] = alt_bn128_Fr(instance.params, 3);
    c_ans[2] = alt_bn128_Fr(instance.params, 4);
    c_ans[3] = alt_bn128_Fr(instance.params, 25);
    c_ans[4] = alt_bn128_Fr(instance.params, 6);
    c_ans[5] = alt_bn128_Fr(instance.params, 7);
    c_ans[6] = alt_bn128_Fr(instance.params, 7);
    c_ans[7] = alt_bn128_Fr(instance.params, 2);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialAdditionZeroA OK %d\n", i);
        else
            printf("testPolynomialAdditionZeroA not OK %d\n", i);
    }
}

__device__ void testPolynomialAdditionZeroB()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(8, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);
    a[5] = alt_bn128_Fr(instance.params, 7);
    a[6] = alt_bn128_Fr(instance.params, 7);
    a[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> b(3, instance.zero());

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_addition(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, 1);
    c_ans[1] = alt_bn128_Fr(instance.params, 3);
    c_ans[2] = alt_bn128_Fr(instance.params, 4);
    c_ans[3] = alt_bn128_Fr(instance.params, 25);
    c_ans[4] = alt_bn128_Fr(instance.params, 6);
    c_ans[5] = alt_bn128_Fr(instance.params, 7);
    c_ans[6] = alt_bn128_Fr(instance.params, 7);
    c_ans[7] = alt_bn128_Fr(instance.params, 2);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialAdditionZeroB OK %d\n", i);
        else
            printf("testPolynomialAdditionZeroB not OK %d\n", i);
    }
}

__device__ void testPolynomialSubtractionSame()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(8, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);
    a[5] = alt_bn128_Fr(instance.params, 7);
    a[6] = alt_bn128_Fr(instance.params, 7);
    a[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> b(8, instance);
    b[0] = alt_bn128_Fr(instance.params, 9);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 11);
    b[3] = alt_bn128_Fr(instance.params, 14);
    b[4] = alt_bn128_Fr(instance.params, 7);
    b[5] = alt_bn128_Fr(instance.params, 1);
    b[6] = alt_bn128_Fr(instance.params, 5);
    b[7] = alt_bn128_Fr(instance.params, 8);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_subtraction(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, -8);
    c_ans[1] = alt_bn128_Fr(instance.params, 0);
    c_ans[2] = alt_bn128_Fr(instance.params, -7);
    c_ans[3] = alt_bn128_Fr(instance.params, 11);
    c_ans[4] = alt_bn128_Fr(instance.params, -1);
    c_ans[5] = alt_bn128_Fr(instance.params, 6);
    c_ans[6] = alt_bn128_Fr(instance.params, 2);
    c_ans[7] = alt_bn128_Fr(instance.params, -6);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialSubtractionSame OK %d\n", i);
        else
            printf("testPolynomialSubtractionSame not OK %d\n", i);
    }
}

__device__ void testPolynomialSubtractionBiggerA()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(8, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);
    a[5] = alt_bn128_Fr(instance.params, 7);
    a[6] = alt_bn128_Fr(instance.params, 7);
    a[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> b(5, instance);
    b[0] = alt_bn128_Fr(instance.params, 9);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 11);
    b[3] = alt_bn128_Fr(instance.params, 14);
    b[4] = alt_bn128_Fr(instance.params, 7);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_subtraction(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, -8);
    c_ans[1] = alt_bn128_Fr(instance.params, 0);
    c_ans[2] = alt_bn128_Fr(instance.params, -7);
    c_ans[3] = alt_bn128_Fr(instance.params, 11);
    c_ans[4] = alt_bn128_Fr(instance.params, -1);
    c_ans[5] = alt_bn128_Fr(instance.params, 7);
    c_ans[6] = alt_bn128_Fr(instance.params, 7);
    c_ans[7] = alt_bn128_Fr(instance.params, 2);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialSubtractionBiggerA OK %d\n", i);
        else
            printf("testPolynomialSubtractionBiggerA not OK %d\n", i);
    }
}

__device__ void testPolynomialSubtractionBiggerB()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(5, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);

    vector<alt_bn128_Fr> b(8, instance);
    b[0] = alt_bn128_Fr(instance.params, 9);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 11);
    b[3] = alt_bn128_Fr(instance.params, 14);
    b[4] = alt_bn128_Fr(instance.params, 7);
    b[5] = alt_bn128_Fr(instance.params, 1);
    b[6] = alt_bn128_Fr(instance.params, 5);
    b[7] = alt_bn128_Fr(instance.params, 8);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_subtraction(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, -8);
    c_ans[1] = alt_bn128_Fr(instance.params, 0);
    c_ans[2] = alt_bn128_Fr(instance.params, -7);
    c_ans[3] = alt_bn128_Fr(instance.params, 11);
    c_ans[4] = alt_bn128_Fr(instance.params, -1);
    c_ans[5] = alt_bn128_Fr(instance.params, -1);
    c_ans[6] = alt_bn128_Fr(instance.params, -5);
    c_ans[7] = alt_bn128_Fr(instance.params, -8);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialSubtractionBiggerB OK %d\n", i);
        else
            printf("testPolynomialSubtractionBiggerB not OK %d\n", i);
    }
}

__device__ void testPolynomialSubtractionZeroA()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(3, instance.zero());

    vector<alt_bn128_Fr> b(8, instance);
    b[0] = alt_bn128_Fr(instance.params, 1);
    b[1] = alt_bn128_Fr(instance.params, 3);
    b[2] = alt_bn128_Fr(instance.params, 4);
    b[3] = alt_bn128_Fr(instance.params, 25);
    b[4] = alt_bn128_Fr(instance.params, 6);
    b[5] = alt_bn128_Fr(instance.params, 7);
    b[6] = alt_bn128_Fr(instance.params, 7);
    b[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_subtraction(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, -1);
    c_ans[1] = alt_bn128_Fr(instance.params, -3);
    c_ans[2] = alt_bn128_Fr(instance.params, -4);
    c_ans[3] = alt_bn128_Fr(instance.params, -25);
    c_ans[4] = alt_bn128_Fr(instance.params, -6);
    c_ans[5] = alt_bn128_Fr(instance.params, -7);
    c_ans[6] = alt_bn128_Fr(instance.params, -7);
    c_ans[7] = alt_bn128_Fr(instance.params, -2);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialSubtractionZeroA OK %d\n", i);
        else
            printf("testPolynomialSubtractionZeroA not OK %d\n", i);
    }
}

__device__ void testPolynomialSubtractionZeroB()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(8, instance);
    a[0] = alt_bn128_Fr(instance.params, 1);
    a[1] = alt_bn128_Fr(instance.params, 3);
    a[2] = alt_bn128_Fr(instance.params, 4);
    a[3] = alt_bn128_Fr(instance.params, 25);
    a[4] = alt_bn128_Fr(instance.params, 6);
    a[5] = alt_bn128_Fr(instance.params, 7);
    a[6] = alt_bn128_Fr(instance.params, 7);
    a[7] = alt_bn128_Fr(instance.params, 2);

    vector<alt_bn128_Fr> b(3, instance.zero());

    vector<alt_bn128_Fr> c(1, instance.zero());

    _polynomial_subtraction(c, a, b);

    vector<alt_bn128_Fr> c_ans(8, instance);
    c_ans[0] = alt_bn128_Fr(instance.params, 1);
    c_ans[1] = alt_bn128_Fr(instance.params, 3);
    c_ans[2] = alt_bn128_Fr(instance.params, 4);
    c_ans[3] = alt_bn128_Fr(instance.params, 25);
    c_ans[4] = alt_bn128_Fr(instance.params, 6);
    c_ans[5] = alt_bn128_Fr(instance.params, 7);
    c_ans[6] = alt_bn128_Fr(instance.params, 7);
    c_ans[7] = alt_bn128_Fr(instance.params, 2);

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c_ans[i] == c[i])
            printf("testPolynomialSubtractionZeroB OK %d\n", i);
        else
            printf("testPolynomialSubtractionZeroB not OK %d\n", i);
    }
}

__device__ void testPolynomialDivision()
{
    alt_bn128_Fr instance(&alt_bn128_fp_params_r);

    vector<alt_bn128_Fr> a(6, instance);
    a[0] = alt_bn128_Fr(instance.params, 5);
    a[1] = alt_bn128_Fr(instance.params, 0);
    a[2] = alt_bn128_Fr(instance.params, 0);
    a[3] = alt_bn128_Fr(instance.params, 13);
    a[4] = alt_bn128_Fr(instance.params, 0);
    a[5] = alt_bn128_Fr(instance.params, 1);

    vector<alt_bn128_Fr> b(3, instance);
    b[0] = alt_bn128_Fr(instance.params, 13);
    b[1] = alt_bn128_Fr(instance.params, 0);
    b[2] = alt_bn128_Fr(instance.params, 1);

    vector<alt_bn128_Fr> Q(1, instance.zero());
    vector<alt_bn128_Fr> R(1, instance.zero());

    _polynomial_division(Q, R, a, b);

    vector<alt_bn128_Fr> Q_ans(4, instance);
    Q_ans[0] = alt_bn128_Fr(instance.params, 0);
    Q_ans[1] = alt_bn128_Fr(instance.params, 0);
    Q_ans[2] = alt_bn128_Fr(instance.params, 0);
    Q_ans[3] = alt_bn128_Fr(instance.params, 1);

    vector<alt_bn128_Fr> R_ans(1, instance);
    R_ans[0] = alt_bn128_Fr(instance.params, 5);

    for (int i = 0; i < Q.size(); i++)
    {
        if (Q_ans[i] == Q[i])
            printf("testPolynomialDivision Q OK %d\n", i);
        else
            printf("testPolynomialDivision Q not OK %d\n", i);
    }

    for (int i = 0; i < R.size(); i++)
    {
        if (R_ans[i] == R[i])
            printf("testPolynomialDivision R OK %d\n", i);
        else
            printf("testPolynomialDivision R not OK %d\n", i);
    }
}

__global__ void test(void) {
    init();

    alt_bn128_pp::init_public_params();

    testPolynomialAdditionSame();
    testPolynomialAdditionBiggerA();
    testPolynomialAdditionBiggerB();
    testPolynomialAdditionZeroA();
    testPolynomialAdditionZeroB();

    testPolynomialSubtractionSame();
    testPolynomialSubtractionBiggerA();
    testPolynomialSubtractionBiggerB();
    testPolynomialSubtractionZeroA();
    testPolynomialSubtractionZeroB();

    testPolynomialDivision();

}

int main()
{
    test<<<1, 1>>>();
    cudaDeviceReset();
}