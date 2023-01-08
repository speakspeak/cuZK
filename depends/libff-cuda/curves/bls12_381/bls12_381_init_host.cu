#include "bls12_381_init_host.cuh"
#include "bls12_381_g1_host.cuh"
#include "bls12_381_g2_host.cuh"

#include <gmp.h>

namespace libff {

Fp_params_host<bls12_381_r_limbs_host> bls12_381_fp_params_r_host;
Fp_params_host<bls12_381_q_limbs_host> bls12_381_fp_params_q_host;
Fp2_params_host<bls12_381_q_limbs_host> bls12_381_fp2_params_q_host;
Fp6_3over2_params_host<bls12_381_q_limbs_host> bls12_381_fp6_params_q_host;
Fp12_params_host<bls12_381_q_limbs_host> bls12_381_fp12_params_q_host;


bls12_381_Fq_host* bls12_381_coeff_b_host;
bls12_381_Fq2_host* bls12_381_twist_host;
bls12_381_Fq2_host* bls12_381_twist_coeff_b_host;
bls12_381_Fq_host* bls12_381_twist_mul_by_b_c0_host;
bls12_381_Fq_host* bls12_381_twist_mul_by_b_c1_host;
bls12_381_Fq2_host* bls12_381_twist_mul_by_q_X_host;
bls12_381_Fq2_host* bls12_381_twist_mul_by_q_Y_host;

bls12_381_G1_params_host g1_params_host;
bls12_381_G2_params_host g2_params_host;



void init_bls12_381_params_host()
{
    typedef bigint_host<bls12_381_r_limbs_host> bigint_r;
    typedef bigint_host<bls12_381_q_limbs_host> bigint_q;

    // fr
    bls12_381_fp_params_r_host.modulus = new bigint_r("52435875175126190479447740508185965837690552500527637822603658699938581184513");
    if (sizeof(mp_limb_t) == 8)
    {
        bls12_381_fp_params_r_host.Rsquared = new bigint_r("3294906474794265442129797520630710739278575682199800681788903916070560242797");
        bls12_381_fp_params_r_host.Rcubed = new bigint_r("49829253988540319354550742249276084460127446355315915089527227471280320770991");
        bls12_381_fp_params_r_host.inv = 0xfffffffeffffffff; // (-1/modulus) mod W
    }
    if (sizeof(mp_limb_t) == 4)
    {
        bls12_381_fp_params_r_host.Rsquared = new bigint_r("3294906474794265442129797520630710739278575682199800681788903916070560242797");
        bls12_381_fp_params_r_host.Rcubed = new bigint_r("49829253988540319354550742249276084460127446355315915089527227471280320770991");
        bls12_381_fp_params_r_host.inv = 0xffffffff;
    }
    bls12_381_fp_params_r_host.num_bits = 255;
    bls12_381_fp_params_r_host.euler = new bigint_r("26217937587563095239723870254092982918845276250263818911301829349969290592256");
    bls12_381_fp_params_r_host.s = 32; // 2-adic order of modulus-1
    bls12_381_fp_params_r_host.t = new bigint_r("12208678567578594777604504606729831043093128246378069236549469339647"); //(modulus-1)/2^s
    bls12_381_fp_params_r_host.t_minus_1_over_2 = new bigint_r("6104339283789297388802252303364915521546564123189034618274734669823");
    bls12_381_fp_params_r_host.multiplicative_generator = new bigint_r("7");
    bls12_381_fp_params_r_host.root_of_unity = new bigint_r("10238227357739495823651030575849232062558860180284477541189508159991286009131");
    bls12_381_fp_params_r_host.nqr = new bigint_r("5");
    bls12_381_fp_params_r_host.nqr_to_t = new bigint_r("937917089079007706106976984802249742464848817460758522850752807661925904159");
    bls12_381_fp_params_r_host.zero = new bls12_381_Fr_host(&bls12_381_fp_params_r_host);
    bls12_381_fp_params_r_host.zero->init_zero();
    bls12_381_fp_params_r_host.one = new bls12_381_Fr_host(&bls12_381_fp_params_r_host);
    bls12_381_fp_params_r_host.one->init_one();

    // fq
    bls12_381_fp_params_q_host.modulus = new bigint_q("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787");
    if (sizeof(mp_limb_t) == 8)
    {
        bls12_381_fp_params_q_host.Rsquared = new bigint_q("2708263910654730174793787626328176511836455197166317677006154293982164122222515399004018013397331347120527951271750"); // k=6
        bls12_381_fp_params_q_host.Rcubed = new bigint_q("1639067542774625894236716575548084905938753837211594095883637014582201460755008380976950835174037649440777609978336");
        bls12_381_fp_params_q_host.inv = 0x89f3fffcfffcfffd;
    }
    if (sizeof(mp_limb_t) == 4)
    {
        bls12_381_fp_params_q_host.Rsquared = new bigint_q("2708263910654730174793787626328176511836455197166317677006154293982164122222515399004018013397331347120527951271750");
        bls12_381_fp_params_q_host.Rcubed = new bigint_q("1639067542774625894236716575548084905938753837211594095883637014582201460755008380976950835174037649440777609978336");
        bls12_381_fp_params_q_host.inv = 0xfffcfffd;
    }
    bls12_381_fp_params_q_host.num_limbs = bls12_381_q_limbs_host;
    bls12_381_fp_params_q_host.num_bits = 381;
    bls12_381_fp_params_q_host.euler = new bigint_q("2001204777610833696708894912867952078278441409969503942666029068062015825245418932221343814564507832018947136279893");
    bls12_381_fp_params_q_host.s = 1;
    bls12_381_fp_params_q_host.t = new bigint_q("2001204777610833696708894912867952078278441409969503942666029068062015825245418932221343814564507832018947136279893");
    bls12_381_fp_params_q_host.t_minus_1_over_2 = new bigint_q("1000602388805416848354447456433976039139220704984751971333014534031007912622709466110671907282253916009473568139946");
    bls12_381_fp_params_q_host.multiplicative_generator = new bigint_q("2");
    bls12_381_fp_params_q_host.root_of_unity = new bigint_q("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786");
    bls12_381_fp_params_q_host.nqr = new bigint_q("2");
    bls12_381_fp_params_q_host.nqr_to_t = new bigint_q("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786");
    bls12_381_fp_params_q_host.zero = new bls12_381_Fq_host(&bls12_381_fp_params_q_host);
    bls12_381_fp_params_q_host.zero->init_zero();
    bls12_381_fp_params_q_host.one = new bls12_381_Fq_host(&bls12_381_fp_params_q_host);
    bls12_381_fp_params_q_host.one->init_one();
    
    // fq2
    bls12_381_fp2_params_q_host.fp_params = &bls12_381_fp_params_q_host;
    bls12_381_fp2_params_q_host.euler = new bigint_host<2 * bls12_381_q_limbs_host>("8009641123864852705971874322159486308847560049665276329931192268492988374245678571700328039651096714987477192770085365265551942269853452968100101210518217905546506517135906379008203984748165830709270511838887449985712996744742684");
    bls12_381_fp2_params_q_host.s = 3;
    bls12_381_fp2_params_q_host.t = new bigint_host<2 * bls12_381_q_limbs_host>("2002410280966213176492968580539871577211890012416319082482798067123247093561419642925082009912774178746869298192521341316387985567463363242025025302629554476386626629283976594752050996187041457677317627959721862496428249186185671");
    bls12_381_fp2_params_q_host.t_minus_1_over_2 = new bigint_host<2 * bls12_381_q_limbs_host>("1001205140483106588246484290269935788605945006208159541241399033561623546780709821462541004956387089373434649096260670658193992783731681621012512651314777238193313314641988297376025498093520728838658813979860931248214124593092835");
    bls12_381_fp2_params_q_host.non_residue = new bigint_host<bls12_381_q_limbs_host>("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786");
    bls12_381_fp2_params_q_host.nqr_c0 = new bigint_host<bls12_381_q_limbs_host>("1"); 
    bls12_381_fp2_params_q_host.nqr_c1 = new bigint_host<bls12_381_q_limbs_host>("1"); 
    bls12_381_fp2_params_q_host.nqr_to_t_c0 = new bigint_host<bls12_381_q_limbs_host>("1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257");
    bls12_381_fp2_params_q_host.nqr_to_t_c1 = new bigint_host<bls12_381_q_limbs_host>("2973677408986561043442465346520108879172042883009249989176415018091420807192182638567116318576472649347015917690530");
    bls12_381_fp2_params_q_host.Frobenius_coeffs_c1[0] = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp2_params_q_host.Frobenius_coeffs_c1[1] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786");


    // fq6
    bls12_381_fp6_params_q_host.fp2_params = &bls12_381_fp2_params_q_host;
    bls12_381_fp6_params_q_host.non_residue_c0 = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp6_params_q_host.non_residue_c1 = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c0[0] = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c0[1] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c0[2] = new bigint_host<bls12_381_q_limbs_host>("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c0[3] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c0[4] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c0[5] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c0[0] = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c0[1] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c0[2] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c0[3] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c0[4] = new bigint_host<bls12_381_q_limbs_host>("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c0[5] = new bigint_host<bls12_381_q_limbs_host>("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620351");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c1[0] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c1[1] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c1[2] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c1[3] = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c1[4] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c1_c1[5] = new bigint_host<bls12_381_q_limbs_host>("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c1[0] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c1[1] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c1[2] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c1[3] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c1[4] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp6_params_q_host.Frobenius_coeffs_c2_c1[5] = new bigint_host<bls12_381_q_limbs_host>("0");

    // fq12
    bls12_381_fp12_params_q_host.fp6_params = &bls12_381_fp6_params_q_host;
    bls12_381_fp12_params_q_host.non_residue_c0 = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp12_params_q_host.non_residue_c1 = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[0] = new bigint_host<bls12_381_q_limbs_host>("1");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[1] = new bigint_host<bls12_381_q_limbs_host>("3850754370037169011952147076051364057158807420970682438676050522613628423219637725072182697113062777891589506424760");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[2] = new bigint_host<bls12_381_q_limbs_host>("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620351");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[3] = new bigint_host<bls12_381_q_limbs_host>("2973677408986561043442465346520108879172042883009249989176415018091420807192182638567116318576472649347015917690530");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[4] = new bigint_host<bls12_381_q_limbs_host>("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[5] = new bigint_host<bls12_381_q_limbs_host>("3125332594171059424908108096204648978570118281977575435832422631601824034463382777937621250592425535493320683825557");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[6] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[7] = new bigint_host<bls12_381_q_limbs_host>("151655185184498381465642749684540099398075398968325446656007613510403227271200139370504932015952886146304766135027");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[8] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[9] = new bigint_host<bls12_381_q_limbs_host>("1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[10] = new bigint_host<bls12_381_q_limbs_host>("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c0[11] = new bigint_host<bls12_381_q_limbs_host>("877076961050607968509681729531255177986764537961432449499635504522207616027455086505066378536590128544573588734230");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[0] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[1] = new bigint_host<bls12_381_q_limbs_host>("151655185184498381465642749684540099398075398968325446656007613510403227271200139370504932015952886146304766135027");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[2] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[3] = new bigint_host<bls12_381_q_limbs_host>("1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[4] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[5] = new bigint_host<bls12_381_q_limbs_host>("877076961050607968509681729531255177986764537961432449499635504522207616027455086505066378536590128544573588734230");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[6] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[7] = new bigint_host<bls12_381_q_limbs_host>("3850754370037169011952147076051364057158807420970682438676050522613628423219637725072182697113062777891589506424760");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[8] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[9] = new bigint_host<bls12_381_q_limbs_host>("2973677408986561043442465346520108879172042883009249989176415018091420807192182638567116318576472649347015917690530");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[10] = new bigint_host<bls12_381_q_limbs_host>("0");
    bls12_381_fp12_params_q_host.Frobenius_coeffs_c1_c1[11] = new bigint_host<bls12_381_q_limbs_host>("3125332594171059424908108096204648978570118281977575435832422631601824034463382777937621250592425535493320683825557");


    //
    bls12_381_coeff_b_host = new bls12_381_Fq_host(&bls12_381_fp_params_q_host, "4");
    bls12_381_twist_host = new bls12_381_Fq2_host(&bls12_381_fp2_params_q_host, bls12_381_Fq_host(&bls12_381_fp_params_q_host, "1"), bls12_381_Fq_host(&bls12_381_fp_params_q_host, "1"));
    bls12_381_Fq2_host t = (*bls12_381_coeff_b_host) * (*bls12_381_twist_host);
    bls12_381_twist_coeff_b_host = new bls12_381_Fq2_host(&bls12_381_fp2_params_q_host, t.c0, t.c1);

    bls12_381_Fq_host q = (*bls12_381_coeff_b_host) * bls12_381_Fq_host(&bls12_381_fp_params_q_host, *bls12_381_fp2_params_q_host.non_residue);
    bls12_381_twist_mul_by_b_c0_host = new bls12_381_Fq_host(q);
    
    bls12_381_Fq_host m = (*bls12_381_coeff_b_host) * bls12_381_Fq_host(&bls12_381_fp_params_q_host, *bls12_381_fp2_params_q_host.non_residue);
    bls12_381_twist_mul_by_b_c1_host = new bls12_381_Fq_host(m);

    bls12_381_twist_mul_by_q_X_host = new bls12_381_Fq2_host(&bls12_381_fp2_params_q_host, 
                                                    bls12_381_Fq_host(&bls12_381_fp_params_q_host, "0"),
                                                    bls12_381_Fq_host(&bls12_381_fp_params_q_host, "4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437"));
    bls12_381_twist_mul_by_q_Y_host = new bls12_381_Fq2_host(&bls12_381_fp2_params_q_host, 
                                                    bls12_381_Fq_host(&bls12_381_fp_params_q_host, "2973677408986561043442465346520108879172042883009249989176415018091420807192182638567116318576472649347015917690530"),
                                                    bls12_381_Fq_host(&bls12_381_fp_params_q_host, "1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257"));



    // g1
    bls12_381_Fq_host fq_t(&bls12_381_fp_params_q_host);
    g1_params_host.fr_params = &bls12_381_fp_params_r_host;
    g1_params_host.fq_params = &bls12_381_fp_params_q_host;
    g1_params_host.G1_zero_X = new bls12_381_Fq_host(fq_t.zero());
    g1_params_host.G1_zero_Y = new bls12_381_Fq_host(fq_t.one());
    g1_params_host.G1_zero_Z = new bls12_381_Fq_host(fq_t.zero());
    g1_params_host.G1_one_X = new bls12_381_Fq_host(&bls12_381_fp_params_q_host, "3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507");
    g1_params_host.G1_one_Y = new bls12_381_Fq_host(&bls12_381_fp_params_q_host, "1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569");               
    g1_params_host.G1_one_Z = new bls12_381_Fq_host(fq_t.one());
                                    
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

    // g2
    bls12_381_Fq2_host fq2_t(&bls12_381_fp2_params_q_host);
    g2_params_host.fr_params = &bls12_381_fp_params_r_host;
    g2_params_host.fq_params = &bls12_381_fp_params_q_host;
    g2_params_host.fq2_params = &bls12_381_fp2_params_q_host;


    g2_params_host.G2_zero_X = new bls12_381_Fq2_host(fq2_t.zero());
    g2_params_host.G2_zero_Y = new bls12_381_Fq2_host(fq2_t.one());
    g2_params_host.G2_zero_Z = new bls12_381_Fq2_host(fq2_t.zero());


    g2_params_host.G2_one_X = new bls12_381_Fq2_host(&bls12_381_fp2_params_q_host,
                                            bls12_381_Fq_host(&bls12_381_fp_params_q_host, "352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160"),  
                                            bls12_381_Fq_host(&bls12_381_fp_params_q_host, "3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758"));
    g2_params_host.G2_one_Y = new bls12_381_Fq2_host(&bls12_381_fp2_params_q_host,
                                            bls12_381_Fq_host(&bls12_381_fp_params_q_host, "1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905"),
                                            bls12_381_Fq_host(&bls12_381_fp_params_q_host, "927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582"));
    g2_params_host.G2_one_Z = new bls12_381_Fq2_host(fq2_t.one());                                    

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
