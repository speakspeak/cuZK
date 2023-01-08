// #include "../bigint.cuh"
// #include "../fp.cuh"
// #include <stdio.h>


// __device__ static const mp_size_t_ n = 8;

// __device__ Fp_params<n> fp_params;


 // 
// {8,
//     "21888242871839275222246405745257275088548364400416034343698204186575808495617",
//     254,  \
//     "10944121435919637611123202872628637544274182200208017171849102093287904247808", \
//     28, \
//     "81540058820840996586704275553141814055101440848469862132140264610111", \
//     "40770029410420498293352137776570907027550720424234931066070132305055",   \
//     "5",  \
//     "19103219067921713944291392827692070036145651957329286315305642004821462161904",  \
//     "5",  \
//     "19103219067921713944291392827692070036145651957329286315305642004821462161904",  \
//     0xefffffff,     \
//     "944936681149208446651664254269745548490766851729442924617792859073125903783",   \
//     "5866548545943845227489894872040244720403868105578784105281690076696998248512" \
//     };


// __device__ void init()
// {

//     bigint<n>* modulus= new bigint<n>("21888242871839275222246405745257275088548364400416034343698204186575808495617");
//     bigint<n>* euler = new bigint<n>("10944121435919637611123202872628637544274182200208017171849102093287904247808");
//     bigint<n>* t = new bigint<n>("81540058820840996586704275553141814055101440848469862132140264610111");
//     bigint<n>* t_minus_1_over_2 = new bigint<n>("40770029410420498293352137776570907027550720424234931066070132305055");
//     bigint<n>* nqr = new bigint<n>("5");
//     bigint<n>* nqr_to_t = new bigint<n>("19103219067921713944291392827692070036145651957329286315305642004821462161904");
//     bigint<n>* multiplicative_generator = new bigint<n>("5");
//     bigint<n>* root_of_unity = new bigint<n>("19103219067921713944291392827692070036145651957329286315305642004821462161904");
//     bigint<n>* Rsquared = new bigint<n>("944936681149208446651664254269745548490766851729442924617792859073125903783");
//     bigint<n>* Rcubed = new bigint<n>("5866548545943845227489894872040244720403868105578784105281690076696998248512");
    
//     fp_params.modulus = modulus;
//     fp_params.euler = euler;
//     fp_params.t = t;
//     fp_params.t_minus_1_over_2 = t_minus_1_over_2;
//     fp_params.nqr = nqr;
//     fp_params.nqr_to_t = nqr_to_t;
//     fp_params.multiplicative_generator = multiplicative_generator;
//     fp_params.root_of_unity = root_of_unity;
//     fp_params.Rsquared = Rsquared;
//     fp_params.Rcubed = Rcubed;

//     fp_params.num_limbs = 8;
//     fp_params.num_bits = 254;
//     fp_params.s = 28;
//     fp_params.inv = 0xefffffff;
// }



// template<mp_size_t_ n, const Fp_params<n>& fp_params>
// __device__ void axy()
// {
//     // Fp_model<n, fp_params> f(10);
//     // Fp_model<n, fp_params> s = f.squared();

//     // printf("%d\n", f.as_ulong());
//     // printf("%d\n", s.as_ulong());
//     // printf("%d", f.is_zero());
// }



// __global__ void warmup(void) {
//     init();
//     axy<n, fp_params>();
// }


int main()
{
   // warmup<<<1, 1>>>();
}






