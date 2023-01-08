#include "alt_bn128_init_host.cuh"
#include "alt_bn128_g1_host.cuh"
#include "alt_bn128_g2_host.cuh"

#include <gmp.h>

namespace libff {
 
Fp_params_host<alt_bn128_r_limbs_host> alt_bn128_fp_params_r_host;
Fp_params_host<alt_bn128_q_limbs_host> alt_bn128_fp_params_q_host;
Fp2_params_host<alt_bn128_q_limbs_host> alt_bn128_fp2_params_q_host;
Fp6_3over2_params_host<alt_bn128_q_limbs_host> alt_bn128_fp6_params_q_host;
Fp12_params_host<alt_bn128_q_limbs_host> alt_bn128_fp12_params_q_host;

alt_bn128_Fq_host* alt_bn128_coeff_b_host;
alt_bn128_Fq2_host* alt_bn128_twist_host;
alt_bn128_Fq2_host* alt_bn128_twist_coeff_b_host;
alt_bn128_Fq_host* alt_bn128_twist_mul_by_b_c0_host;
alt_bn128_Fq_host* alt_bn128_twist_mul_by_b_c1_host;
alt_bn128_Fq2_host* alt_bn128_twist_mul_by_q_X_host;
alt_bn128_Fq2_host* alt_bn128_twist_mul_by_q_Y_host;

alt_bn128_G1_params_host g1_params_host;
alt_bn128_G2_params_host g2_params_host;


void init_alt_bn128_params_host()
{
    typedef bigint_host<alt_bn128_r_limbs_host> bigint_r;
    typedef bigint_host<alt_bn128_q_limbs_host> bigint_q;

    // fr
    alt_bn128_fp_params_r_host.modulus = new bigint_r("21888242871839275222246405745257275088548364400416034343698204186575808495617");
    if (sizeof(mp_limb_t) == 8)
    {
        alt_bn128_fp_params_r_host.Rsquared = new bigint_r("944936681149208446651664254269745548490766851729442924617792859073125903783");
        alt_bn128_fp_params_r_host.Rcubed = new bigint_r("5866548545943845227489894872040244720403868105578784105281690076696998248512");
        alt_bn128_fp_params_r_host.inv = 0xc2e1f593efffffff;
    }
    if (sizeof(mp_limb_t) == 4)
    {
        alt_bn128_fp_params_r_host.Rsquared = new bigint_r("944936681149208446651664254269745548490766851729442924617792859073125903783");
        alt_bn128_fp_params_r_host.Rcubed = new bigint_r("5866548545943845227489894872040244720403868105578784105281690076696998248512");
        alt_bn128_fp_params_r_host.inv = 0xefffffff;
    }
    alt_bn128_fp_params_r_host.num_limbs = alt_bn128_r_limbs_host;
    alt_bn128_fp_params_r_host.num_bits = 254;
    alt_bn128_fp_params_r_host.euler = new bigint_r("10944121435919637611123202872628637544274182200208017171849102093287904247808");
    alt_bn128_fp_params_r_host.s = 28;
    alt_bn128_fp_params_r_host.t = new bigint_r("81540058820840996586704275553141814055101440848469862132140264610111");
    alt_bn128_fp_params_r_host.t_minus_1_over_2 = new bigint_r("40770029410420498293352137776570907027550720424234931066070132305055");
    alt_bn128_fp_params_r_host.multiplicative_generator = new bigint_r("5");
    alt_bn128_fp_params_r_host.root_of_unity = new bigint_r("19103219067921713944291392827692070036145651957329286315305642004821462161904");
    alt_bn128_fp_params_r_host.nqr = new bigint_r("5");
    alt_bn128_fp_params_r_host.nqr_to_t = new bigint_r("19103219067921713944291392827692070036145651957329286315305642004821462161904");
    alt_bn128_fp_params_r_host.zero = new alt_bn128_Fr_host(&alt_bn128_fp_params_r_host);
    alt_bn128_fp_params_r_host.zero->init_zero();
    alt_bn128_fp_params_r_host.one = new alt_bn128_Fr_host(&alt_bn128_fp_params_r_host);
    alt_bn128_fp_params_r_host.one->init_one();

    //fq
    alt_bn128_fp_params_q_host.modulus = new bigint_q("21888242871839275222246405745257275088696311157297823662689037894645226208583");
    if (sizeof(mp_limb_t) == 8)
    {
        alt_bn128_fp_params_q_host.Rsquared = new bigint_q("3096616502983703923843567936837374451735540968419076528771170197431451843209");
        alt_bn128_fp_params_q_host.Rcubed = new bigint_q("14921786541159648185948152738563080959093619838510245177710943249661917737183");
        alt_bn128_fp_params_q_host.inv = 0x87d20782e4866389;
    }
    if (sizeof(mp_limb_t) == 4)
    {
        alt_bn128_fp_params_q_host.Rsquared = new bigint_q("3096616502983703923843567936837374451735540968419076528771170197431451843209");
        alt_bn128_fp_params_q_host.Rcubed = new bigint_q("14921786541159648185948152738563080959093619838510245177710943249661917737183");
        alt_bn128_fp_params_q_host.inv = 0xe4866389;
    }
    alt_bn128_fp_params_q_host.num_limbs = alt_bn128_q_limbs_host;
    alt_bn128_fp_params_q_host.num_bits = 254;
    alt_bn128_fp_params_q_host.euler = new bigint_q("10944121435919637611123202872628637544348155578648911831344518947322613104291");
    alt_bn128_fp_params_q_host.s = 1;
    alt_bn128_fp_params_q_host.t = new bigint_q("10944121435919637611123202872628637544348155578648911831344518947322613104291");
    alt_bn128_fp_params_q_host.t_minus_1_over_2 = new bigint_q("5472060717959818805561601436314318772174077789324455915672259473661306552145");
    alt_bn128_fp_params_q_host.multiplicative_generator = new bigint_q("3");
    alt_bn128_fp_params_q_host.root_of_unity = new bigint_q("21888242871839275222246405745257275088696311157297823662689037894645226208582");
    alt_bn128_fp_params_q_host.nqr = new bigint_q("3");
    alt_bn128_fp_params_q_host.nqr_to_t = new bigint_q("21888242871839275222246405745257275088696311157297823662689037894645226208582");
    alt_bn128_fp_params_q_host.zero = new alt_bn128_Fq_host(&alt_bn128_fp_params_q_host);
    alt_bn128_fp_params_q_host.zero->init_zero();
    alt_bn128_fp_params_q_host.one = new alt_bn128_Fq_host(&alt_bn128_fp_params_q_host);
    alt_bn128_fp_params_q_host.one->init_one();

    //fq2
    alt_bn128_fp2_params_q_host.fp_params = &alt_bn128_fp_params_q_host;
    alt_bn128_fp2_params_q_host.euler = new bigint_host<2*alt_bn128_q_limbs_host>("239547588008311421220994022608339370399626158265550411218223901127035046843189118723920525909718935985594116157406550130918127817069793474323196511433944");
    alt_bn128_fp2_params_q_host.s = 4;
    alt_bn128_fp2_params_q_host.t = new bigint_host<2*alt_bn128_q_limbs_host>("29943448501038927652624252826042421299953269783193801402277987640879380855398639840490065738714866998199264519675818766364765977133724184290399563929243");
    alt_bn128_fp2_params_q_host.t_minus_1_over_2 = new bigint_host<2 * alt_bn128_q_limbs_host>("14971724250519463826312126413021210649976634891596900701138993820439690427699319920245032869357433499099632259837909383182382988566862092145199781964621");
    alt_bn128_fp2_params_q_host.non_residue = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275222246405745257275088696311157297823662689037894645226208582");
    alt_bn128_fp2_params_q_host.nqr_c0 = new bigint_host<alt_bn128_q_limbs_host>("2");
    alt_bn128_fp2_params_q_host.nqr_c1 = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp2_params_q_host.nqr_to_t_c0 = new bigint_host<alt_bn128_q_limbs_host>("5033503716262624267312492558379982687175200734934877598599011485707452665730");
    alt_bn128_fp2_params_q_host.nqr_to_t_c1 = new bigint_host<alt_bn128_q_limbs_host>("314498342015008975724433667930697407966947188435857772134235984660852259084");
    alt_bn128_fp2_params_q_host.Frobenius_coeffs_c1[0] = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp2_params_q_host.Frobenius_coeffs_c1[1] = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275222246405745257275088696311157297823662689037894645226208582");

    //fq6
    alt_bn128_fp6_params_q_host.fp2_params = &alt_bn128_fp2_params_q_host;
    alt_bn128_fp6_params_q_host.non_residue_c0 = new bigint_host<alt_bn128_q_limbs_host>("9");
    alt_bn128_fp6_params_q_host.non_residue_c1 = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c0[0] = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c0[1] = new bigint_host<alt_bn128_q_limbs_host>("21575463638280843010398324269430826099269044274347216827212613867836435027261");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c0[2] = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275220042445260109153167277707414472061641714758635765020556616");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c0[3] = new bigint_host<alt_bn128_q_limbs_host>("3772000881919853776433695186713858239009073593817195771773381919316419345261");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c0[4] = new bigint_host<alt_bn128_q_limbs_host>("2203960485148121921418603742825762020974279258880205651966");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c0[5] = new bigint_host<alt_bn128_q_limbs_host>("18429021223477853657660792034369865839114504446431234726392080002137598044644");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c1[0] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c1[1] = new bigint_host<alt_bn128_q_limbs_host>("10307601595873709700152284273816112264069230130616436755625194854815875713954");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c1[2] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c1[3] = new bigint_host<alt_bn128_q_limbs_host>("2236595495967245188281701248203181795121068902605861227855261137820944008926");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c1[4] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c1_c1[5] = new bigint_host<alt_bn128_q_limbs_host>("9344045779998320333812420223237981029506012124075525679208581902008406485703");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c0[0] = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c0[1] = new bigint_host<alt_bn128_q_limbs_host>("2581911344467009335267311115468803099551665605076196740867805258568234346338");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c0[2] = new bigint_host<alt_bn128_q_limbs_host>("2203960485148121921418603742825762020974279258880205651966");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c0[3] = new bigint_host<alt_bn128_q_limbs_host>("5324479202449903542726783395506214481928257762400643279780343368557297135718");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c0[4] = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275220042445260109153167277707414472061641714758635765020556616");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c0[5] = new bigint_host<alt_bn128_q_limbs_host>("13981852324922362344252311234282257507216387789820983642040889267519694726527");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c1[0] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c1[1] = new bigint_host<alt_bn128_q_limbs_host>("19937756971775647987995932169929341994314640652964949448313374472400716661030");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c1[2] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c1[3] = new bigint_host<alt_bn128_q_limbs_host>("16208900380737693084919495127334387981393726419856888799917914180988844123039");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c1[4] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp6_params_q_host.Frobenius_coeffs_c2_c1[5] = new bigint_host<alt_bn128_q_limbs_host>("7629828391165209371577384193250820201684255241773809077146787135900891633097");

    //fp12
    alt_bn128_fp12_params_q_host.fp6_params = &alt_bn128_fp6_params_q_host;
    alt_bn128_fp12_params_q_host.non_residue_c0 = new bigint_host<alt_bn128_q_limbs_host>("9");
    alt_bn128_fp12_params_q_host.non_residue_c1 = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[0] = new bigint_host<alt_bn128_q_limbs_host>("1");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[1] = new bigint_host<alt_bn128_q_limbs_host>("8376118865763821496583973867626364092589906065868298776909617916018768340080");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[2] = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275220042445260109153167277707414472061641714758635765020556617");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[3] = new bigint_host<alt_bn128_q_limbs_host>("11697423496358154304825782922584725312912383441159505038794027105778954184319");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[4] = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275220042445260109153167277707414472061641714758635765020556616");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[5] = new bigint_host<alt_bn128_q_limbs_host>("3321304630594332808241809054958361220322477375291206261884409189760185844239");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[6] = new bigint_host<alt_bn128_q_limbs_host>("21888242871839275222246405745257275088696311157297823662689037894645226208582");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[7] = new bigint_host<alt_bn128_q_limbs_host>("13512124006075453725662431877630910996106405091429524885779419978626457868503");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[8] = new bigint_host<alt_bn128_q_limbs_host>("2203960485148121921418603742825762020974279258880205651966");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[9] = new bigint_host<alt_bn128_q_limbs_host>("10190819375481120917420622822672549775783927716138318623895010788866272024264");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[10] = new bigint_host<alt_bn128_q_limbs_host>("2203960485148121921418603742825762020974279258880205651967");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c0[11] = new bigint_host<alt_bn128_q_limbs_host>("18566938241244942414004596690298913868373833782006617400804628704885040364344");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[0] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[1] = new bigint_host<alt_bn128_q_limbs_host>("16469823323077808223889137241176536799009286646108169935659301613961712198316");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[2] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[3] = new bigint_host<alt_bn128_q_limbs_host>("303847389135065887422783454877609941456349188919719272345083954437860409601");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[4] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[5] = new bigint_host<alt_bn128_q_limbs_host>("5722266937896532885780051958958348231143373700109372999374820235121374419868");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[6] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[7] = new bigint_host<alt_bn128_q_limbs_host>("5418419548761466998357268504080738289687024511189653727029736280683514010267");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[8] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[9] = new bigint_host<alt_bn128_q_limbs_host>("21584395482704209334823622290379665147239961968378104390343953940207365798982");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[10] = new bigint_host<alt_bn128_q_limbs_host>("0");
    alt_bn128_fp12_params_q_host.Frobenius_coeffs_c1_c1[11] = new bigint_host<alt_bn128_q_limbs_host>("16165975933942742336466353786298926857552937457188450663314217659523851788715");


    // g1
    alt_bn128_Fq_host fq_t(&alt_bn128_fp_params_q_host);
    g1_params_host.fr_params = &alt_bn128_fp_params_r_host;
    g1_params_host.fq_params = &alt_bn128_fp_params_q_host;
    g1_params_host.G1_zero_X = new alt_bn128_Fq_host(fq_t.zero());
    g1_params_host.G1_zero_Y = new alt_bn128_Fq_host(fq_t.one());
    g1_params_host.G1_zero_Z = new alt_bn128_Fq_host(fq_t.zero());
    g1_params_host.G1_one_X = new alt_bn128_Fq_host(&alt_bn128_fp_params_q_host, "1");
    g1_params_host.G1_one_Y = new alt_bn128_Fq_host(&alt_bn128_fp_params_q_host, "2");
    g1_params_host.G1_one_Z = new alt_bn128_Fq_host(fq_t.one());

    g1_params_host.wnaf_window_table = new size_t[4];
    g1_params_host.wnaf_window_table_size = 4;
    g1_params_host.wnaf_window_table[0] = 11;
    g1_params_host.wnaf_window_table[1] = 24;
    g1_params_host.wnaf_window_table[2] = 60;
    g1_params_host.wnaf_window_table[3] = 127;

    g1_params_host.fixed_base_exp_window_table_length = new size_t[22];
    g1_params_host.fixed_base_exp_window_table_length_size = 22;
    g1_params_host.fixed_base_exp_window_table_length[0] = 1;
    g1_params_host.fixed_base_exp_window_table_length[1] = 5;
    g1_params_host.fixed_base_exp_window_table_length[2] = 11;
    g1_params_host.fixed_base_exp_window_table_length[3] = 32;
    g1_params_host.fixed_base_exp_window_table_length[4] = 55;
    g1_params_host.fixed_base_exp_window_table_length[5] = 162;
    g1_params_host.fixed_base_exp_window_table_length[6] = 360;
    g1_params_host.fixed_base_exp_window_table_length[7] = 815;
    g1_params_host.fixed_base_exp_window_table_length[8] = 2373;
    g1_params_host.fixed_base_exp_window_table_length[9] = 6978;
    g1_params_host.fixed_base_exp_window_table_length[10] = 7122;
    g1_params_host.fixed_base_exp_window_table_length[11] = 0;
    g1_params_host.fixed_base_exp_window_table_length[12] = 57818;
    g1_params_host.fixed_base_exp_window_table_length[13] = 0;
    g1_params_host.fixed_base_exp_window_table_length[14] = 169679;
    g1_params_host.fixed_base_exp_window_table_length[15] = 439759;
    g1_params_host.fixed_base_exp_window_table_length[16] = 936073;
    g1_params_host.fixed_base_exp_window_table_length[17] = 0;
    g1_params_host.fixed_base_exp_window_table_length[18] = 4666555;
    g1_params_host.fixed_base_exp_window_table_length[19] = 7580404;
    g1_params_host.fixed_base_exp_window_table_length[20] = 0;
    g1_params_host.fixed_base_exp_window_table_length[21] = 34552892;

    // // g2
    alt_bn128_Fq2_host fq2_t(&alt_bn128_fp2_params_q_host);
    g2_params_host.fr_params = &alt_bn128_fp_params_r_host;
    g2_params_host.fq_params = &alt_bn128_fp_params_q_host;
    g2_params_host.fq2_params = &alt_bn128_fp2_params_q_host;

    g2_params_host.G2_zero_X = new alt_bn128_Fq2_host(fq2_t.zero());
    g2_params_host.G2_zero_Y = new alt_bn128_Fq2_host(fq2_t.one());
    g2_params_host.G2_zero_Z = new alt_bn128_Fq2_host(fq2_t.zero());

    g2_params_host.G2_one_X = new alt_bn128_Fq2_host(&alt_bn128_fp2_params_q_host, 
                                            alt_bn128_Fq_host(&alt_bn128_fp_params_q_host, "10857046999023057135944570762232829481370756359578518086990519993285655852781"),
                                            alt_bn128_Fq_host(&alt_bn128_fp_params_q_host, "11559732032986387107991004021392285783925812861821192530917403151452391805634"));
    g2_params_host.G2_one_Y = new alt_bn128_Fq2_host(&alt_bn128_fp2_params_q_host, 
                                            alt_bn128_Fq_host(&alt_bn128_fp_params_q_host, "8495653923123431417604973247489272438418190587263600148770280649306958101930"),
                                            alt_bn128_Fq_host(&alt_bn128_fp_params_q_host, "4082367875863433681332203403145435568316851327593401208105741076214120093531"));
    g2_params_host.G2_one_Z = new alt_bn128_Fq2_host(fq2_t.one());

    g2_params_host.wnaf_window_table = new size_t[4];
    g2_params_host.wnaf_window_table_size = 4;
    g2_params_host.wnaf_window_table[0] = 5;
    g2_params_host.wnaf_window_table[1] = 15;
    g2_params_host.wnaf_window_table[2] = 39;
    g2_params_host.wnaf_window_table[3] = 109;

    g2_params_host.fixed_base_exp_window_table_length = new size_t[22];
    g2_params_host.fixed_base_exp_window_table_length_size = 22;
    g2_params_host.fixed_base_exp_window_table_length[0] = 1;
    g2_params_host.fixed_base_exp_window_table_length[1] = 5;
    g2_params_host.fixed_base_exp_window_table_length[2] = 10;
    g2_params_host.fixed_base_exp_window_table_length[3] = 25;
    g2_params_host.fixed_base_exp_window_table_length[4] = 59;
    g2_params_host.fixed_base_exp_window_table_length[5] = 154;
    g2_params_host.fixed_base_exp_window_table_length[6] = 334;
    g2_params_host.fixed_base_exp_window_table_length[7] = 743;
    g2_params_host.fixed_base_exp_window_table_length[8] = 2034;
    g2_params_host.fixed_base_exp_window_table_length[9] = 4988;
    g2_params_host.fixed_base_exp_window_table_length[10] = 8888;
    g2_params_host.fixed_base_exp_window_table_length[11] = 26271;
    g2_params_host.fixed_base_exp_window_table_length[12] = 39768;
    g2_params_host.fixed_base_exp_window_table_length[13] = 106276;
    g2_params_host.fixed_base_exp_window_table_length[14] = 141703;
    g2_params_host.fixed_base_exp_window_table_length[15] = 462423;
    g2_params_host.fixed_base_exp_window_table_length[16] = 926872;
    g2_params_host.fixed_base_exp_window_table_length[17] = 0;
    g2_params_host.fixed_base_exp_window_table_length[18] = 4873049;
    g2_params_host.fixed_base_exp_window_table_length[19] = 5706708;
    g2_params_host.fixed_base_exp_window_table_length[20] = 0;
    g2_params_host.fixed_base_exp_window_table_length[21] = 31673815;
}

}
