#include "../scalar_multiplication/wnaf.cuh"
#include "../scalar_multiplication/multiexp.cuh"
#include "../curves/curve_utils.cuh"
#include "../curves/alt_bn128/alt_bn128_pairing.cuh"
#include "../curves/alt_bn128/alt_bn128_pp.cuh"
#include "../curves/alt_bn128/alt_bn128_g2.cuh"
#include "../curves/alt_bn128/alt_bn128_g1.cuh"
#include "../curves/alt_bn128/alt_bn128_init.cuh"
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


template<mp_size_t_ n>
__device__ void axy()
{
    // Fp6_3over2_model<n> tt(fp6_params);
    // tt.inverse();
    Fp_model<n> f(fp_params);
    f.inverse();
    f.one();
    printf("%d", f.fp_params.num_limbs);
    Fp2_model<n> t(fp2_params, f, f);
    Fp6_3over2_model<n> tt(fp6_params, t, t, t);
    tt.inverse();
    // Fp_model<n, fp_params> f(-7);
    // printf("final value: %d\n", f.as_ulong());
    // Fp_model<n, fp_params> s = f.inverse();
    // Fp_model<n, fp_params> t = s * f;

    // Fp_model<n, fp_params> tf = -f;
    // // tf = -tf;

    // printf("%d\n", s.as_ulong());
    // printf("%d\n", t.as_ulong());
    // printf("%d\n", tf.as_ulong());
}


template<mp_size_t_ n>
__device__ void fp_test()
{
    alt_bn128_Fr f(alt_bn128_fp_params_r);
    printf("module is valid: %d\n", f.modulus_is_valid());
    f.set_ulong(7);
    printf("f value: %d\n", f.as_ulong());
    printf("f value: %d\n", f.as_bigint().as_ulong());
    printf("is zero: %d\n", f.is_zero());
    f.set_ulong(0);
    printf("is zero: %d\n", f.is_zero());
    f.set_ulong(7);

    alt_bn128_Fr g(alt_bn128_fp_params_r, 9);
    printf("g + f: %d\n", (f + g).as_ulong());
    printf("g - f: %d\n", (g - f).as_ulong());
    printf("g * f: %d\n", (g * f).as_ulong());
    printf("g ^ 3: %d\n", (g ^ 3).as_ulong());
    f.set_ulong(3);
    printf("g ^ f: %d\n", (g ^ f.as_bigint()).as_ulong());
    alt_bn128_Fr h(alt_bn128_fp_params_r, -9);
    printf("-h: %d\n", (-h).as_ulong());

    printf("g ^ 2: %d\n", g.squared().as_ulong());
    printf("g-1: %d\n", g.inverse().as_ulong());
    printf("g: %d\n",g.inverse().inverse().as_ulong());
    printf("g ^ 1/2 : %d\n", g.sqrt().as_ulong());
    printf("g : %d\n", g.sqrt().squared().as_ulong());


    alt_bn128_Fr r = f.random_element();
    printf("r value: %d\n", r.as_ulong());
}


template<mp_size_t_ n>
__device__ void fp2_test()
{
    alt_bn128_Fq f(alt_bn128_fp_params_q, 3);
    alt_bn128_Fq g(alt_bn128_fp_params_q, 4);
    alt_bn128_Fq2 f2(alt_bn128_fp2_params_q, f, g);
    alt_bn128_Fq2 g2(alt_bn128_fp2_params_q, g, f);

    printf("g2 + f2: %d\n", (f2 + g2).c0.as_ulong());
    printf("g2 - f2: %d\n", (g2 - f2).c0.as_ulong());
    printf("g2 * g2: %d\n", (g2 * g2).c0.as_ulong());
    alt_bn128_Fq pow(alt_bn128_fp_params_q, 3);
    printf("g2 ^ pow: %d\n", (- (g2 ^ pow.as_bigint())).c0.as_ulong());
    printf("g-1: %d\n", g2.inverse().c0.as_ulong());
    printf("g2-1 * g2 :%d\n", (g2.inverse() * g2).c0.as_ulong());
    alt_bn128_Fq m(alt_bn128_fp_params_q, 7);
    alt_bn128_Fq t(alt_bn128_fp_params_q, 24);
    alt_bn128_Fq2 m2(alt_bn128_fp2_params_q, m, t);
    printf("m2 : %d\n", m2.sqrt().squared().c0.as_ulong());

    printf("m2 ^ p: %d\n", (m2 ^ *m2.fp_params.modulus).c0.as_ulong());
    printf("m2 ^ p: %d\n", (-(m2 ^ *m2.fp_params.modulus)).c1.as_ulong());
}


template<mp_size_t_ n>
__device__ void fp6_3over2_test()
{
    alt_bn128_Fq f(alt_bn128_fp_params_q, 3);
    alt_bn128_Fq g(alt_bn128_fp_params_q, 4);
    alt_bn128_Fq2 f2(alt_bn128_fp2_params_q, f, g);
    alt_bn128_Fq2 g2(alt_bn128_fp2_params_q, g, f);
    alt_bn128_Fq2 h2(alt_bn128_fp2_params_q, g, g);

    alt_bn128_Fq6 f6(alt_bn128_fp6_params_q, f2, g2, h2);
    alt_bn128_Fq6 g6(alt_bn128_fp6_params_q, g2, h2, f2);

    printf("g6 + f6: %d\n", (f6 + g6).c0.c0.as_ulong());
    printf("g6 - f6: %d\n", (g6 - f6).c0.c0.as_ulong());
    printf("f6 * g6: %d\n", (-(f6 * g6)).c0.c0.as_ulong());
    printf("g6 * g6: %d\n", (g6 * g6).c0.c0.as_ulong());
    alt_bn128_Fq h(alt_bn128_fp_params_q, 2);
    printf("g6 ^ 2: %d\n", (g6 ^ h.as_bigint()).c0.c0.as_ulong());

    alt_bn128_Fq6 h6 = g6.squared();
    printf("h6: %d\n", h6.c0.c0.as_ulong());
    // printf("h6: %d\n", h6.sqrt().squared().c0.c0.as_ulong());
    printf("h6-1 * h6: %d\n", (h6.inverse() * h6).c0.c0.as_ulong());
}


template<mp_size_t_ n>
__device__ void fp12_test()
{
    alt_bn128_Fq f(alt_bn128_fp_params_q, 3);
    alt_bn128_Fq g(alt_bn128_fp_params_q, 4);
    alt_bn128_Fq2 f2(alt_bn128_fp2_params_q, f, g);
    alt_bn128_Fq2 g2(alt_bn128_fp2_params_q, g, f);
    alt_bn128_Fq2 h2(alt_bn128_fp2_params_q, g, g);

    alt_bn128_Fq6 f6(alt_bn128_fp6_params_q, f2, g2, h2);
    alt_bn128_Fq6 g6(alt_bn128_fp6_params_q, g2, h2, f2);

    alt_bn128_Fq12 f12(alt_bn128_fp12_params_q, f6, g6);
    printf("f12 ^ 2: %u\n", f12.squared().c0.c0.c1.as_ulong());
    printf("f12 cyclotomic_squared : %u\n", f12.cyclotomic_squared().c0.c0.c1.as_ulong());   

    alt_bn128_Fq12 g12 = f12.Frobenius_map(12/2) * f12.inverse();
    assert(g12.inverse() == g12.unitary_inverse());
}

template<mp_size_t_ n>
__device__ void g1_test()
{
    alt_bn128_G1 t(g1_params);
    alt_bn128_G1 zero = t.zero();
    
    alt_bn128_G1 one = t.one();
    alt_bn128_G1 two = bigint<1>(2l) * t.one();
    assert(two == two);
    alt_bn128_G1 five = bigint<1>(5l) * t.one();
    alt_bn128_G1 three = bigint<1>(3l) * t.one();
    alt_bn128_G1 four = bigint<1>(4l) * t.one();

    assert(two+five == three+four);

    alt_bn128_G1 a = five;
    alt_bn128_G1 b = four;
    
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
    alt_bn128_G2 t(g2_params);
    alt_bn128_G2 zero = t.zero();
    
    alt_bn128_G2 one = t.one();
    alt_bn128_G2 two = bigint<1>(2l) * t.one();
    assert(two == two);
    alt_bn128_G2 five = bigint<1>(5l) * t.one();
    alt_bn128_G2 three = bigint<1>(3l) * t.one();
    alt_bn128_G2 four = bigint<1>(4l) * t.one();

    assert(two+five == three+four);

    alt_bn128_G2 a = five;
    alt_bn128_G2 b = four;
    
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
    alt_bn128_Fr s(alt_bn128_fp_params_r, 2);
    alt_bn128_G1 g1_t(g1_params);
    alt_bn128_G2 g2_t(g2_params);

    printf("begin\n");
    alt_bn128_G1 P = bigint<1>(3l) * g1_t.one();
    alt_bn128_G2 Q = bigint<1>(4l) * g2_t.one();
    printf("P: %d\n", P.X.as_ulong());
    printf("mid\n");
    alt_bn128_G1 sP = s * P;
    alt_bn128_G2 sQ = s * Q;

    printf("sP: %d\n", sP.X.as_ulong());
    printf("Q: %d\n", Q.X.c0.as_ulong());

    alt_bn128_GT ans1 = alt_bn128_ate_reduced_pairing(sP, Q);
    alt_bn128_GT ans2 = alt_bn128_ate_reduced_pairing(P, sQ);
    alt_bn128_GT ans3 = alt_bn128_ate_reduced_pairing(P, Q) ^ s;
    
    // printf("ans1: %d\n", ans1.c0.c0.c0.as_ulong());

    printf("1 == 2: %d\n", ans1 == ans2);
    printf("2 == 3: %d\n", ans2 == ans3);
    // // assert(ans2 == ans3);
    assert(ans1 != ans1.one());
    assert((ans1^s.field_char()) == ans1.one());
    // printf("ok\n");

    const alt_bn128_G1 P1 = bigint<1>(3l) * g1_t.one();
    const alt_bn128_G1 P2 = bigint<1>(4l) * g1_t.one();
    const alt_bn128_G2 Q1 = bigint<1>(5l) * g2_t.one();
    const alt_bn128_G2 Q2 = bigint<1>(6l) * g2_t.one();

    const alt_bn128_ate_G1_precomp prec_P1 = alt_bn128_precompute_G1(P1);
    const alt_bn128_ate_G1_precomp prec_P2 = alt_bn128_precompute_G1(P2);
    const alt_bn128_ate_G2_precomp prec_Q1 = alt_bn128_precompute_G2(Q1);
    const alt_bn128_ate_G2_precomp prec_Q2 = alt_bn128_precompute_G2(Q2);

    const alt_bn128_GT ans_1 = alt_bn128_miller_loop(prec_P1, prec_Q1);
    const alt_bn128_GT ans_2 = alt_bn128_miller_loop(prec_P2, prec_Q2);
    const alt_bn128_GT ans_12 = alt_bn128_double_miller_loop(prec_P1, prec_Q1, prec_P2, prec_Q2);
    assert(ans_1 * ans_2 == ans_12);
}


template<mp_size_t_ n>
__device__ void multiexp_test()
{
    printf("multiexp_begin\n");

    alt_bn128_Fr s(alt_bn128_fp_params_r, 2);
    alt_bn128_G1 g1_t(g1_params);
    alt_bn128_G2 g2_t(g2_params);

    printf("begin\n");
    alt_bn128_G1 P = bigint<1>(3l) * g1_t.one();
    alt_bn128_G2 Q = bigint<1>(4l) * g2_t.one();

    
}


__global__ void warmup(void) {
    printf("warmup begin!\n");
    // init();
    // init_alt_bn128_params();
    // alt_bn128_pp t;
    // alt_bn128_pp::init_public_params();
    // printf("%d\n", alt_bn128_fp_params_r.num_bits);  
    // axy<n>();
    alt_bn128_pp::init_public_params();
    fp_test<n>();
    fp2_test<n>();
    fp6_3over2_test<n>();
    fp12_test<n>();
    g1_test<n>();
    g2_test<n>();
    pairing_test<n>();
    multiexp_test<n>();
    // bxy<n, fp_params, fp2_params>();
}

#define N   10

int main()
{
    warmup<<<1, 1>>>();

    cudaDeviceReset();
}