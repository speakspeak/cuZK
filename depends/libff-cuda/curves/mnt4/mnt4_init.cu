#ifndef __MNT4_INIT_CU__
#define __MNT4_INIT_CU__

#include "mnt4_g1.cuh"
#include "mnt4_g2.cuh"

namespace libff {
 
__device__ Fp_params<mnt4_r_limbs> mnt4_fp_params_r;
__device__ Fp_params<mnt4_q_limbs> mnt4_fp_params_q;
__device__ Fp2_params<mnt4_q_limbs> mnt4_fp2_params_q;
__device__ Fp4_params<mnt4_q_limbs> mnt4_fp4_params_q;

__device__ mnt4_Fq2* mnt4_twist;
__device__ mnt4_Fq2* mnt4_twist_coeff_a;
__device__ mnt4_Fq2* mnt4_twist_coeff_b;
__device__ mnt4_Fq* mnt4_twist_mul_by_a_c0;
__device__ mnt4_Fq* mnt4_twist_mul_by_a_c1;
__device__ mnt4_Fq* mnt4_twist_mul_by_b_c0;
__device__ mnt4_Fq* mnt4_twist_mul_by_b_c1;
__device__ mnt4_Fq* mnt4_twist_mul_by_q_X;
__device__ mnt4_Fq* mnt4_twist_mul_by_q_Y;

struct mnt4_G1_params;
struct mnt4_G2_params;

__device__ mnt4_G1_params g1_params;
__device__ mnt4_G2_params g2_params;

__device__ bigint<mnt4_q_limbs>* mnt4_ate_loop_count;
__device__ bool mnt4_ate_is_loop_count_neg;
__device__ bigint<4*mnt4_q_limbs>* mnt4_final_exponent;
__device__ bigint<mnt4_q_limbs>* mnt4_final_exponent_last_chunk_abs_of_w0;
__device__ bool mnt4_final_exponent_last_chunk_is_w0_neg;
__device__ bigint<mnt4_q_limbs>* mnt4_final_exponent_last_chunk_w1;

__device__ void init_mnt4_params()
{
    typedef bigint<mnt4_r_limbs> bigint_r;
    typedef bigint<mnt4_q_limbs> bigint_q;

    assert(sizeof(mp_limb_t_) == 8 || sizeof(mp_limb_t_) == 4);

    // fr
    mnt4_fp_params_r.modulus = new bigint_r("475922286169261325753349249653048451545124878552823515553267735739164647307408490559963137");
    if (sizeof(mp_limb_t_) == 8)
    {
        mnt4_fp_params_r.Rsquared = new bigint_r("163983144722506446826715124368972380525894397127205577781234305496325861831001705438796139");
        mnt4_fp_params_r.Rcubed = new bigint_r("207236281459091063710247635236340312578688659363066707916716212805695955118593239854980171");
        mnt4_fp_params_r.inv = 0xbb4334a3ffffffff;
        
    }
    if (sizeof(mp_limb_t_) == 4)
    {
        mnt4_fp_params_r.Rsquared = new bigint_r("163983144722506446826715124368972380525894397127205577781234305496325861831001705438796139");
        mnt4_fp_params_r.Rcubed = new bigint_r("207236281459091063710247635236340312578688659363066707916716212805695955118593239854980171");
        mnt4_fp_params_r.inv = 0xffffffff;
    }

    mnt4_fp_params_r.num_bits = 298;
    mnt4_fp_params_r.euler = new bigint_r("237961143084630662876674624826524225772562439276411757776633867869582323653704245279981568");
    mnt4_fp_params_r.s = 34;
    mnt4_fp_params_r.t = new bigint_r("27702323054502562488973446286577291993024111641153199339359284829066871159442729");
    mnt4_fp_params_r.t_minus_1_over_2 = new bigint_r("13851161527251281244486723143288645996512055820576599669679642414533435579721364");
    mnt4_fp_params_r.multiplicative_generator = new bigint_r("10");
    mnt4_fp_params_r.root_of_unity = new bigint_r("120638817826913173458768829485690099845377008030891618010109772937363554409782252579816313");
    mnt4_fp_params_r.nqr = new bigint_r("5");
    mnt4_fp_params_r.nqr_to_t = new bigint_r("406220604243090401056429458730298145937262552508985450684842547562990900634752279902740880");
    mnt4_fp_params_r.zero = new mnt4_Fr(&mnt4_fp_params_r);
    mnt4_fp_params_r.zero->init_zero();
    mnt4_fp_params_r.one = new mnt4_Fr(&mnt4_fp_params_r);
    mnt4_fp_params_r.one->init_one();

    //fq
    mnt4_fp_params_q.modulus = new bigint_q("475922286169261325753349249653048451545124879242694725395555128576210262817955800483758081");
    if (sizeof(mp_limb_t_) == 8)
    {
        mnt4_fp_params_q.Rsquared = new bigint_q("273000478523237720910981655601160860640083126627235719712980612296263966512828033847775776");
        mnt4_fp_params_q.Rcubed = new bigint_q("427298980065529822574935274648041073124704261331681436071990730954930769758106792920349077");
        mnt4_fp_params_q.inv = 0xb071a1b67165ffff;
    }
    if (sizeof(mp_limb_t_) == 4)
    {
        mnt4_fp_params_q.Rsquared = new bigint_q("273000478523237720910981655601160860640083126627235719712980612296263966512828033847775776");
        mnt4_fp_params_q.Rcubed = new bigint_q("427298980065529822574935274648041073124704261331681436071990730954930769758106792920349077");
        mnt4_fp_params_q.inv = 0x7165ffff;
    }

    mnt4_fp_params_q.num_bits = 298;
    mnt4_fp_params_q.euler = new bigint_q("237961143084630662876674624826524225772562439621347362697777564288105131408977900241879040");
    mnt4_fp_params_q.s = 17;
    mnt4_fp_params_q.t = new bigint_q("3630998887399759870554727551674258816109656366292531779446068791017229177993437198515");
    mnt4_fp_params_q.t_minus_1_over_2 = new bigint_q("1815499443699879935277363775837129408054828183146265889723034395508614588996718599257");
    mnt4_fp_params_q.multiplicative_generator = new bigint_q("17");
    mnt4_fp_params_q.root_of_unity = new bigint_q("264706250571800080758069302369654305530125675521263976034054878017580902343339784464690243");
    mnt4_fp_params_q.nqr = new bigint_q("17");
    mnt4_fp_params_q.nqr_to_t = new bigint_q("264706250571800080758069302369654305530125675521263976034054878017580902343339784464690243");
    mnt4_fp_params_q.zero = new mnt4_Fr(&mnt4_fp_params_q);
    mnt4_fp_params_q.zero->init_zero();
    mnt4_fp_params_q.one = new mnt4_Fr(&mnt4_fp_params_q);
    mnt4_fp_params_q.one->init_one();

    //fq2
    mnt4_fp2_params_q.fp_params = &mnt4_fp_params_q;
    mnt4_fp2_params_q.euler = new bigint<2*mnt4_q_limbs>("113251011236288135098249345249154230895914381858788918106847214243419142422924133497460817468249854833067260038985710370091920860837014281886963086681184370139950267830740466401280");
    mnt4_fp2_params_q.s = 18;
    mnt4_fp2_params_q.t = new bigint<2*mnt4_q_limbs>("864036645784668999467844736092790457885088972921668381552484239528039111503022258739172496553419912972009735404859240494475714575477709059806542104196047745818712370534824115");
    mnt4_fp2_params_q.t_minus_1_over_2 = new bigint<2 * mnt4_q_limbs>("432018322892334499733922368046395228942544486460834190776242119764019555751511129369586248276709956486004867702429620247237857287738854529903271052098023872909356185267412057");
    mnt4_fp2_params_q.non_residue = new bigint_q("17");
    mnt4_fp2_params_q.nqr_c0 = new bigint_q("8");
    mnt4_fp2_params_q.nqr_c1 = new bigint_q("1");
    mnt4_fp2_params_q.nqr_to_t_c0 = new bigint_q("0");
    mnt4_fp2_params_q.nqr_to_t_c1 = new bigint_q("29402818985595053196743631544512156561638230562612542604956687802791427330205135130967658");
    mnt4_fp2_params_q.Frobenius_coeffs_c1[0] = new bigint_q("1");
    mnt4_fp2_params_q.Frobenius_coeffs_c1[1] = new bigint_q("475922286169261325753349249653048451545124879242694725395555128576210262817955800483758080");

    //fq4
    mnt4_fp4_params_q.fp2_params = &mnt4_fp2_params_q;
    mnt4_fp4_params_q.non_residue = new bigint_q("17");
    mnt4_fp4_params_q.Frobenius_coeffs_c1[0] = new bigint_q("1");
    mnt4_fp4_params_q.Frobenius_coeffs_c1[1] = new bigint_q("7684163245453501615621351552473337069301082060976805004625011694147890954040864167002308");
    mnt4_fp4_params_q.Frobenius_coeffs_c1[2] = new bigint_q("475922286169261325753349249653048451545124879242694725395555128576210262817955800483758080");
    mnt4_fp4_params_q.Frobenius_coeffs_c1[3] = new bigint_q("468238122923807824137727898100575114475823797181717920390930116882062371863914936316755773");

    //
    g1_params.coeff_a = new mnt4_Fq(&mnt4_fp_params_q, "2");
    g1_params.coeff_b = new mnt4_Fq(&mnt4_fp_params_q, "423894536526684178289416011533888240029318103673896002803341544124054745019340795360841685");

    mnt4_Fq non_residue = mnt4_Fq(&mnt4_fp_params_q, *mnt4_fp2_params_q.non_residue);

    mnt4_twist = new mnt4_Fq2(&mnt4_fp2_params_q, *mnt4_fp_params_q.zero, *mnt4_fp_params_q.one);
    mnt4_twist_coeff_a = new mnt4_Fq2(&mnt4_fp2_params_q, *g1_params.coeff_a * non_residue, *mnt4_fp_params_q.zero);
    mnt4_twist_coeff_b = new mnt4_Fq2(&mnt4_fp2_params_q, *mnt4_fp_params_q.zero, *g1_params.coeff_b * non_residue);

    g2_params.twist = new mnt4_Fq2(*mnt4_twist);
    g2_params.coeff_a = new mnt4_Fq2(*mnt4_twist_coeff_a);
    g2_params.coeff_b = new mnt4_Fq2(*mnt4_twist_coeff_b);
    mnt4_twist_mul_by_a_c0 = new mnt4_Fq(*g1_params.coeff_a * non_residue);
    mnt4_twist_mul_by_a_c1 = new mnt4_Fq(*g1_params.coeff_a * non_residue);
    mnt4_twist_mul_by_b_c0 = new mnt4_Fq(*g1_params.coeff_b * non_residue.squared());
    mnt4_twist_mul_by_b_c1 = new mnt4_Fq(*g1_params.coeff_b * non_residue);
    mnt4_twist_mul_by_q_X = new mnt4_Fq(&mnt4_fp_params_q, "475922286169261325753349249653048451545124879242694725395555128576210262817955800483758080");
    mnt4_twist_mul_by_q_Y = new mnt4_Fq(&mnt4_fp_params_q, "7684163245453501615621351552473337069301082060976805004625011694147890954040864167002308");


    // g1
    mnt4_Fq fq_t(&mnt4_fp_params_q);
    g1_params.fr_params = &mnt4_fp_params_r;
    g1_params.fq_params = &mnt4_fp_params_q;
    g1_params.G1_zero_X = new mnt4_Fq(fq_t.zero());
    g1_params.G1_zero_Y = new mnt4_Fq(fq_t.one());
    g1_params.G1_zero_Z = new mnt4_Fq(fq_t.zero());
    g1_params.G1_one_X = new mnt4_Fq(&mnt4_fp_params_q, "60760244141852568949126569781626075788424196370144486719385562369396875346601926534016838");
    g1_params.G1_one_Y = new mnt4_Fq(&mnt4_fp_params_q, "363732850702582978263902770815145784459747722357071843971107674179038674942891694705904306");
    g1_params.G1_one_Z = new mnt4_Fq(fq_t.one());

    g1_params.wnaf_window_table = new size_t[4];
    g1_params.wnaf_window_table_size = 4;
    g1_params.wnaf_window_table[0] = 11;
    g1_params.wnaf_window_table[1] = 24;
    g1_params.wnaf_window_table[2] = 60;
    g1_params.wnaf_window_table[3] = 127;

    g1_params.fixed_base_exp_window_table_length = new size_t[22];
    g1_params.fixed_base_exp_window_table_length_size = 22;
    g1_params.fixed_base_exp_window_table_length[0] = 1;
    g1_params.fixed_base_exp_window_table_length[1] = 5;
    g1_params.fixed_base_exp_window_table_length[2] = 10;
    g1_params.fixed_base_exp_window_table_length[3] = 25;
    g1_params.fixed_base_exp_window_table_length[4] = 60;
    g1_params.fixed_base_exp_window_table_length[5] = 144;
    g1_params.fixed_base_exp_window_table_length[6] = 345;
    g1_params.fixed_base_exp_window_table_length[7] = 855;
    g1_params.fixed_base_exp_window_table_length[8] = 1805;
    g1_params.fixed_base_exp_window_table_length[9] = 3912;
    g1_params.fixed_base_exp_window_table_length[10] = 11265;
    g1_params.fixed_base_exp_window_table_length[11] = 27898;
    g1_params.fixed_base_exp_window_table_length[12] = 57597;
    g1_params.fixed_base_exp_window_table_length[13] = 145299;
    g1_params.fixed_base_exp_window_table_length[14] = 157205;
    g1_params.fixed_base_exp_window_table_length[15] = 601601;
    g1_params.fixed_base_exp_window_table_length[16] = 1107377;
    g1_params.fixed_base_exp_window_table_length[17] = 1789647;
    g1_params.fixed_base_exp_window_table_length[18] = 4392627;
    g1_params.fixed_base_exp_window_table_length[19] = 8221211;
    g1_params.fixed_base_exp_window_table_length[20] = 0;
    g1_params.fixed_base_exp_window_table_length[21] = 42363731;

    // g2
    mnt4_Fq2 fq2_t(&mnt4_fp2_params_q);
    g2_params.fr_params = &mnt4_fp_params_r;
    g2_params.fq_params = &mnt4_fp_params_q;
    g2_params.fq2_params = &mnt4_fp2_params_q;

    g2_params.G2_zero_X = new mnt4_Fq2(fq2_t.zero());
    g2_params.G2_zero_Y = new mnt4_Fq2(fq2_t.one());
    g2_params.G2_zero_Z = new mnt4_Fq2(fq2_t.zero());

    g2_params.G2_one_X = new mnt4_Fq2(&mnt4_fp2_params_q, 
                                            mnt4_Fq(&mnt4_fp_params_q, "438374926219350099854919100077809681842783509163790991847867546339851681564223481322252708"),
                                            mnt4_Fq(&mnt4_fp_params_q, "37620953615500480110935514360923278605464476459712393277679280819942849043649216370485641"));
    g2_params.G2_one_Y = new mnt4_Fq2(&mnt4_fp2_params_q, 
                                            mnt4_Fq(&mnt4_fp_params_q, "37437409008528968268352521034936931842973546441370663118543015118291998305624025037512482"),
                                            mnt4_Fq(&mnt4_fp_params_q, "424621479598893882672393190337420680597584695892317197646113820787463109735345923009077489"));
    g2_params.G2_one_Z = new mnt4_Fq2(fq2_t.one());

    g2_params.wnaf_window_table = new size_t[4];
    g2_params.wnaf_window_table_size = 4;
    g2_params.wnaf_window_table[0] = 5;
    g2_params.wnaf_window_table[1] = 15;
    g2_params.wnaf_window_table[2] = 39;
    g2_params.wnaf_window_table[3] = 109;

    g2_params.fixed_base_exp_window_table_length = new size_t[22];
    g2_params.fixed_base_exp_window_table_length_size = 22;
    g2_params.fixed_base_exp_window_table_length[0] = 1;
    g2_params.fixed_base_exp_window_table_length[1] = 4;
    g2_params.fixed_base_exp_window_table_length[2] = 10;
    g2_params.fixed_base_exp_window_table_length[3] = 25;
    g2_params.fixed_base_exp_window_table_length[4] = 60;
    g2_params.fixed_base_exp_window_table_length[5] = 143;
    g2_params.fixed_base_exp_window_table_length[6] = 345;
    g2_params.fixed_base_exp_window_table_length[7] = 821;
    g2_params.fixed_base_exp_window_table_length[8] = 1794;
    g2_params.fixed_base_exp_window_table_length[9] = 3920;
    g2_params.fixed_base_exp_window_table_length[10] = 11301;
    g2_params.fixed_base_exp_window_table_length[11] = 18960;
    g2_params.fixed_base_exp_window_table_length[12] = 44199;
    g2_params.fixed_base_exp_window_table_length[13] = 0;
    g2_params.fixed_base_exp_window_table_length[14] = 150800;
    g2_params.fixed_base_exp_window_table_length[15] = 548695;
    g2_params.fixed_base_exp_window_table_length[16] = 1051769;
    g2_params.fixed_base_exp_window_table_length[17] = 2023926;
    g2_params.fixed_base_exp_window_table_length[18] = 3787109;
    g2_params.fixed_base_exp_window_table_length[19] = 7107480;
    g2_params.fixed_base_exp_window_table_length[20] = 0;
    g2_params.fixed_base_exp_window_table_length[21] = 38760027;

    // pairing paramters
    mnt4_ate_loop_count = new bigint_q("689871209842287392837045615510547309923794944");
    mnt4_ate_is_loop_count_neg = false;
    mnt4_final_exponent = new bigint<4*mnt4_q_limbs>("107797360357109903430794490309592072278927783803031854357910908121903439838772861497177116410825586743089760869945394610511917274977971559062689561855016270594656570874331111995170645233717143416875749097203441437192367065467706065411650403684877366879441766585988546560");
    mnt4_final_exponent_last_chunk_abs_of_w0 = new bigint_q("689871209842287392837045615510547309923794945");
    mnt4_final_exponent_last_chunk_is_w0_neg = false;
    mnt4_final_exponent_last_chunk_w1 = new bigint_q("1");
}

}


#endif