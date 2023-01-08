#include "../scalar_multiplication/wnaf.cuh"
// #include "../scalar_multiplication/multiexp.cuh"
#include "../curves/curve_utils.cuh"
#include "../curves/mnt4/mnt4_pairing.cuh"
#include "../curves/mnt4/mnt4_pp.cuh"
#include "../curves/mnt4/mnt4_g2.cuh"
#include "../curves/mnt4/mnt4_g1.cuh"
#include "../curves/mnt4/mnt4_init.cuh"
#include "../fields/bigint.cuh"
#include "../fields/fp.cuh"
#include "../fields/fp2.cuh"
// #include "../fields/fp3.cuh"
// #include "../fields/fp4.cuh"
// #include "../fields/fp6_2over3.cuh"
#include "../fields/fp6_3over2.cuh"
#include "../fields/fp12_2over3over2.cuh"
#include "../fields/field_utils.cuh"
#include <stdio.h>

using namespace libff;


__device__ static const mp_size_t_ n = 8;


template<mp_size_t_ n>
__device__ void fp_test()
{
    mnt4_Fr f(&mnt4_fp_params_r);
    printf("module is valid: %d\n", f.modulus_is_valid());
    f.set_ulong(7);
    printf("f value: %d\n", f.as_ulong());
    printf("f value: %d\n", f.as_bigint().as_ulong());
    printf("is zero: %d\n", f.is_zero());
    f.set_ulong(0);
    printf("is zero: %d\n", f.is_zero());
    f.set_ulong(7);

    mnt4_Fr g(&mnt4_fp_params_r, 9);
    printf("g + f: %d\n", (f + g).as_ulong());
    printf("g - f: %d\n", (g - f).as_ulong());
    printf("g * f: %d\n", (g * f).as_ulong());
    printf("g ^ 3: %d\n", (g ^ 3).as_ulong());
    f.set_ulong(3);
    printf("g ^ f: %d\n", (g ^ f.as_bigint()).as_ulong());
    mnt4_Fr h(&mnt4_fp_params_r, -9);
    printf("-h: %d\n", (-h).as_ulong());

    printf("g ^ 2: %d\n", g.squared().as_ulong());
    printf("g-1: %d\n", g.inverse().as_ulong());
    printf("g: %d\n",g.inverse().inverse().as_ulong());
    printf("g ^ 1/2 : %d\n", g.sqrt().as_ulong());
    printf("g : %d\n", g.sqrt().squared().as_ulong());


    mnt4_Fr r = f.random_element();
    printf("r value: %d\n", r.as_ulong());
}


template<mp_size_t_ n>
__device__ void fp2_test()
{
    mnt4_Fq f(&mnt4_fp_params_q, 3);
    mnt4_Fq g(&mnt4_fp_params_q, 4);
    mnt4_Fq2 f2(&mnt4_fp2_params_q, f, g);
    mnt4_Fq2 g2(&mnt4_fp2_params_q, g, f);

    printf("g2 + f2: %d\n", (f2 + g2).c0.as_ulong());
    printf("g2 - f2: %d\n", (g2 - f2).c0.as_ulong());
    printf("g2 * g2: %d\n", (g2 * g2).c0.as_ulong());
    mnt4_Fq pow(&mnt4_fp_params_q, 3);
    printf("g2 ^ pow: %d\n", (- (g2 ^ pow.as_bigint())).c0.as_ulong());
    printf("g-1: %d\n", g2.inverse().c0.as_ulong());
    printf("g2-1 * g2 :%d\n", (g2.inverse() * g2).c0.as_ulong());
    mnt4_Fq m(&mnt4_fp_params_q, 7);
    mnt4_Fq t(&mnt4_fp_params_q, 24);
    mnt4_Fq2 m2(&mnt4_fp2_params_q, m, t);
    //printf("m2 : %d\n", m2.sqrt().squared().c0.as_ulong());

    printf("m2 ^ p: %d\n", (m2 ^ *m2.params->fp_params->modulus).c0.as_ulong());
    printf("m2 ^ p: %d\n", (-(m2 ^ *m2.params->fp_params->modulus)).c1.as_ulong());
}

template<mp_size_t_ n>
__device__ void fp4_test()
{
    mnt4_Fq f(&mnt4_fp_params_q, 3);
    mnt4_Fq g(&mnt4_fp_params_q, 4);
    mnt4_Fq2 f2(&mnt4_fp2_params_q, f, g);
    mnt4_Fq2 g2(&mnt4_fp2_params_q, g, f);

    mnt4_Fq4 f4(&mnt4_fp4_params_q, f2, g2);
    mnt4_Fq4 g4(&mnt4_fp4_params_q, g2, f2);

    printf("g4 + f4: %d\n", (f4 + g4).c0.c0.as_ulong());
    printf("g4 - f4: %d\n", (g4 - f4).c0.c0.as_ulong());
    printf("f4 * g4: %d\n", (f4 * g4).c0.c0.as_ulong());
    printf("g4 * g4: %d\n", (g4 * g4).c0.c0.as_ulong());

    mnt4_Fq h(&mnt4_fp_params_q, 2);
    printf("g4 ^ 2: %d\n", (g4 ^ h.as_bigint()).c0.c0.as_ulong());

    mnt4_Fq4 h4 = g4.squared();
    printf("h4: %d\n", h4.c0.c0.as_ulong());
    printf("h4-1 * h4: %d\n", (h4.inverse() * h4).c0.c0.as_ulong());
}


// template<mp_size_t_ n>
// __device__ void fp6_3over2_test()
// {
//     mnt4_Fq f(&mnt4_fp_params_q, 3);
//     mnt4_Fq g(&mnt4_fp_params_q, 4);
//     mnt4_Fq2 f2(&mnt4_fp2_params_q, f, g);
//     mnt4_Fq2 g2(&mnt4_fp2_params_q, g, f);
//     mnt4_Fq2 h2(&mnt4_fp2_params_q, g, g);

//     mnt4_Fq6 f6(&mnt4_fp6_params_q, f2, g2, h2);
//     mnt4_Fq6 g6(&mnt4_fp6_params_q, g2, h2, f2);

//     printf("g6 + f6: %d\n", (f6 + g6).c0.c0.as_ulong());
//     printf("g6 - f6: %d\n", (g6 - f6).c0.c0.as_ulong());
//     printf("f6 * g6: %d\n", (-(f6 * g6)).c0.c0.as_ulong());
//     printf("g6 * g6: %d\n", (g6 * g6).c0.c0.as_ulong());
//     mnt4_Fq h(&mnt4_fp_params_q, 2);
//     printf("g6 ^ 2: %d\n", (g6 ^ h.as_bigint()).c0.c0.as_ulong());

//     mnt4_Fq6 h6 = g6.squared();
//     printf("h6: %d\n", h6.c0.c0.as_ulong());
//     // printf("h6: %d\n", h6.sqrt().squared().c0.c0.as_ulong());
//     printf("h6-1 * h6: %d\n", (h6.inverse() * h6).c0.c0.as_ulong());
// }

// template<mp_size_t_ n>
// __device__ void fp12_test()
// {
//     mnt4_Fq f(&mnt4_fp_params_q, 3);
//     mnt4_Fq g(&mnt4_fp_params_q, 4);
//     mnt4_Fq2 f2(&mnt4_fp2_params_q, f, g);
//     mnt4_Fq2 g2(&mnt4_fp2_params_q, g, f);
//     mnt4_Fq2 h2(&mnt4_fp2_params_q, g, g);

//     mnt4_Fq6 f6(&mnt4_fp6_params_q, f2, g2, h2);
//     mnt4_Fq6 g6(&mnt4_fp6_params_q, g2, h2, f2);

//     mnt4_Fq12 f12(&mnt4_fp12_params_q, f6, g6);
//     printf("f12 ^ 2: %u\n", f12.squared().c0.c0.c1.as_ulong());
//     printf("f12 cyclotomic_squared : %u\n", f12.cyclotomic_squared().c0.c0.c1.as_ulong());   

//     mnt4_Fq12 g12 = f12.Frobenius_map(12/2) * f12.inverse();
//     assert(g12.inverse() == g12.unitary_inverse());
// }



template<mp_size_t_ n>
__device__ void g1_test()
{
    mnt4_G1 t(&g1_params);
    mnt4_G1 zero = t.zero();
    
    mnt4_G1 one = t.one();
    mnt4_G1 two = bigint<1>(2l) * t.one();
    assert(two == two);
    mnt4_G1 five = bigint<1>(5l) * t.one();
    mnt4_G1 three = bigint<1>(3l) * t.one();
    mnt4_G1 four = bigint<1>(4l) * t.one();

    assert(two+five == three+four);

    mnt4_G1 a = five;
    mnt4_G1 b = four;
    
    assert(a.dbl() == a + a);
    assert(b.dbl() == b + b);
    assert(one.add(two) == three);
    assert(two.add(one) == three);
    assert(a + b == b + a);
    assert(a - a == zero);
    assert(a - b == a + (-b));
    assert(a - b == (-b) + a);

    //
    assert(zero + (-a) == -a);
    assert(zero - a == -a);
    assert(a - zero == a);
    assert(a + zero == a);
    assert(zero + a == a);

    assert((a + b).dbl() == (a + b) + (b + a));
    assert(bigint<1>("2") * (a + b) == (a + b) + (b + a));


    assert(a.order() * a == zero);
    assert(a.order() * one == zero);
    assert((a.order() * a) - a != zero);
    assert((a.order() * one) - one != zero);
}

template<mp_size_t_ n>
__device__ void g2_test()
{
    mnt4_G2 t(&g2_params);
    mnt4_G2 zero = t.zero();
    
    mnt4_G2 one = t.one();
    mnt4_G2 two = bigint<1>(2l) * t.one();
    assert(two == two);
    mnt4_G2 five = bigint<1>(5l) * t.one();
    mnt4_G2 three = bigint<1>(3l) * t.one();
    mnt4_G2 four = bigint<1>(4l) * t.one();

    assert(two+five == three+four);

    mnt4_G2 a = five;
    mnt4_G2 b = four;
    
    assert(a.dbl() == a + a);
    assert(b.dbl() == b + b);
    assert(one.add(two) == three);
    assert(two.add(one) == three);
    assert(a + b == b + a);
    assert(a - a == zero);
    assert(a - b == a + (-b));
    assert(a - b == (-b) + a);

    //
    assert(zero + (-a) == -a);
    assert(zero - a == -a);
    assert(a - zero == a);
    assert(a + zero == a);
    assert(zero + a == a);

    assert((a + b).dbl() == (a + b) + (b + a));
    assert(bigint<1>("2") * (a + b) == (a + b) + (b + a));

    assert(a.order() * a == zero);
    assert(a.order() * one == zero);
    assert((a.order() * a) - a != zero);
    assert((a.order() * one) - one != zero);
}

template<mp_size_t_ n>
__device__ void pairing_test()
{
    mnt4_Fr s(&mnt4_fp_params_r, 2);
    mnt4_G1 g1_t(&g1_params);
    mnt4_G2 g2_t(&g2_params);

    printf("begin\n");
    mnt4_G1 P = bigint<1>(3l) * g1_t.one();
    mnt4_G2 Q = bigint<1>(4l) * g2_t.one();
    printf("P: %d\n", P.X.as_ulong());
    printf("mid\n");
    mnt4_G1 sP = s * P;
    mnt4_G2 sQ = s * Q;

    printf("sP: %d\n", sP.X.as_ulong());
    printf("Q: %d\n", Q.X.c0.as_ulong());

    mnt4_GT ans1 = mnt4_ate_reduced_pairing(sP, Q);
    mnt4_GT ans2 = mnt4_ate_reduced_pairing(P, sQ);
    mnt4_GT ans3 = mnt4_ate_reduced_pairing(P, Q) ^ s;

    // printf("ans1: %d\n", ans1.c0.c0.c0.as_ulong());

    printf("1 == 2: %d\n", ans1 == ans2);
    printf("2 == 3: %d\n", ans2 == ans3);
    // // assert(ans2 == ans3);
    assert(ans1 != ans1.one());
    assert((ans1^s.field_char()) == ans1.one());
    // printf("ok\n");

    const mnt4_G1 P1 = bigint<1>(3l) * g1_t.one();
    const mnt4_G1 P2 = bigint<1>(4l) * g1_t.one();
    const mnt4_G2 Q1 = bigint<1>(5l) * g2_t.one();
    const mnt4_G2 Q2 = bigint<1>(6l) * g2_t.one();

    const mnt4_ate_G1_precomp prec_P1 = mnt4_precompute_G1(P1);
    const mnt4_ate_G1_precomp prec_P2 = mnt4_precompute_G1(P2);
    const mnt4_ate_G2_precomp prec_Q1 = mnt4_precompute_G2(Q1);
    const mnt4_ate_G2_precomp prec_Q2 = mnt4_precompute_G2(Q2);

    const mnt4_GT ans_1 = mnt4_miller_loop(prec_P1, prec_Q1);
    const mnt4_GT ans_2 = mnt4_miller_loop(prec_P2, prec_Q2);
    const mnt4_GT ans_12 = mnt4_double_miller_loop(prec_P1, prec_Q1, prec_P2, prec_Q2);
    assert(ans_1 * ans_2 == ans_12);
}


__device__ void random_test()
{
    mnt4_Fr s(&mnt4_fp_params_r);
    mnt4_G1 g1(&g1_params);
    mnt4_G2 g2(&g2_params);

    mnt4_Fr t = s.random_element();
    mnt4_G1 g1t = g1.random_element();
    mnt4_G2 g2t = g2.random_element();

    g1t.to_special();
    g2t.to_special();

    printf("Fr: "); t.as_bigint().print();

    printf("G1:\n");
    printf("X: "); g1t.X.as_bigint().print();
    printf("Y: "); g1t.Y.as_bigint().print();

    printf("G2:\n");
    printf("X1: "); g2t.X.c0.as_bigint().print(); 
    printf("X2: "); g2t.X.c1.as_bigint().print();

    printf("Y1: "); g2t.Y.c0.as_bigint().print(); 
    printf("Y2: "); g2t.Y.c1.as_bigint().print();

    // g2.random_element()
}

// template<mp_size_t_ n>
// __device__ void multiexp_test()
// {
//     mnt4_Fr s1(&mnt4_fp_params_r, 1);
//     mnt4_Fr s2(&mnt4_fp_params_r, 1);
//     mnt4_Fr s3(&mnt4_fp_params_r, 1);
//     mnt4_G1 g1_t(&g1_params);
//     // mnt4_G2 g2_t(g2_params);

//     mnt4_G1* p_P = libstl::create<mnt4_G1>(bigint<1>(3l) * g1_t.one());

//     mnt4_G1& P = *p_P;
//     // mnt4_G2 Q = bigint<1>(4l) * g2_t.one();


//     libstl::vector<mnt4_Fr>* p_scalar = libstl::create<libstl::vector<mnt4_Fr>>(libstl::vector<mnt4_Fr>(3));
//     libstl::vector<mnt4_G1>* p_vec = libstl::create<libstl::vector<mnt4_G1>>(libstl::vector<mnt4_G1>(3));

//     (*p_scalar)[0] = s1;
//     (*p_scalar)[1] = s2;
//     (*p_scalar)[2] = s3;
//     (*p_vec)[0] = P;
//     (*p_vec)[1] = P;
//     (*p_vec)[2] = P;

//     mnt4_G1 res = p_multi_exp_faster<mnt4_G1, mnt4_Fr, multi_exp_method_naive_plain>(*p_vec, *p_scalar, s1, P, 1, 5);

//     printf("multiexp: %d\n", res == bigint<1>(9) * g1_t.one());
// }

__global__ void warmup(void) {

    printf("begin\n");
    size_t init_size = 100000000;
    libstl::initAllocator(init_size);
    printf("init ok\n");

    gmp_init_allocator_();
    mnt4_pp::init_public_params();

    printf("init params ok\n");

    fp_test<n>();
    fp2_test<n>();
    fp4_test<n>();
    // fp6_3over2_test<n>();
    // fp12_test<n>();
    g1_test<n>();
    g2_test<n>();
    pairing_test<n>();
    //multiexp_test<n>();
    //random_test();
    printf("ok\n");

}

__global__ void testFq()
{
    mnt4_Fq f(&mnt4_fp_params_q);
    mnt4_Fq a = f.one().dbl();
    mnt4_Fq b = f.one().dbl().dbl();

    for (size_t i = 0; i < 100000; ++i)
    {
        a = a * b;
    }
}

__global__ void testG1()
{
    mnt4_Fq f(&mnt4_fp_params_q);

    mnt4_G1 t(&g1_params);

    mnt4_G1 g10 = t.zero();
    mnt4_G1 g11 = t.zero();

    g10.X = g10.Y = g10.Z = f.one();
    g11.X = g11.Y = g11.Z = f.one();

    for (size_t i = 0; i < 100000; ++i)
    {
       g10 = g10.mixed_add(g11);
    }
}

__global__ void testG2()
{
    mnt4_Fq2 f(&mnt4_fp2_params_q);

    mnt4_G2 t(&g2_params);

    mnt4_G2 g10 = t.zero();
    mnt4_G2 g11 = t.zero();

    g10.X = g10.Y = g10.Z = f.one();
    g11.X = g11.Y = g11.Z = f.one();

    for (size_t i = 0; i < 100000; ++i)
    {
       g10 = g10.mixed_add(g11);
    }
}

int main()
{
    cudaSetDevice(0);

    size_t heap_size = 1024*1024*1024;
    heap_size *= 4;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

    warmup<<<1, 1>>>();

    // cudaEvent_t fqStart, fqEnd;
    // cudaEventCreate( &fqStart);
	// cudaEventCreate( &fqEnd);
    // cudaEventRecord( fqStart, 0); 
    // cudaEventSynchronize(fqStart);

    // testFq<<<1,1>>>();

    // cudaEventRecord( fqEnd, 0);
    // cudaEventSynchronize(fqEnd);
    // float   Timefq;
    // cudaEventElapsedTime( &Timefq, fqStart, fqEnd );
    // printf( "Time to fq:  %3.5f ms\n", Timefq );



    // cudaEvent_t g1Start, g1End;
    // cudaEventCreate( &g1Start);
	// cudaEventCreate( &g1End);
    // cudaEventRecord( g1Start, 0); 
    // cudaEventSynchronize(g1Start);

    // testG1<<<1,1>>>();

    // cudaEventRecord( g1End, 0);
    // cudaEventSynchronize(g1End);
    // float   TimeG1;
    // cudaEventElapsedTime( &TimeG1, g1Start, g1End );
    // printf( "Time to g1:  %3.5f ms\n", TimeG1 );


    // cudaEvent_t g2Start, g2End;
    // cudaEventCreate( &g2Start);
	// cudaEventCreate( &g2End);
    // cudaEventRecord( g2Start, 0); 
    // cudaEventSynchronize(g1Start);

    // testG2<<<1,1>>>();

    // cudaEventRecord( g2End, 0);
    // cudaEventSynchronize(g2End);
    // float   TimeG2;
    // cudaEventElapsedTime( &TimeG2, g2Start, g2End );
    // printf( "Time to g2:  %3.5f ms\n", TimeG2 );

    cudaDeviceReset();
}