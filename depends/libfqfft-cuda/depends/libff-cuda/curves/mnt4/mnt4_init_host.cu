#include "mnt4_init_host.cuh"
#include "mnt4_g1_host.cuh"
#include "mnt4_g2_host.cuh"

#include <gmp.h>

namespace libff {
 
 Fp_params_host<mnt4_r_limbs_host> mnt4_fp_params_r_host;
 Fp_params_host<mnt4_q_limbs_host> mnt4_fp_params_q_host;
 Fp2_params_host<mnt4_q_limbs_host> mnt4_fp2_params_q_host;
 Fp4_params_host<mnt4_q_limbs_host> mnt4_fp4_params_q_host;

 mnt4_Fq2_host* mnt4_twist_host;
 mnt4_Fq2_host* mnt4_twist_coeff_a_host;
 mnt4_Fq2_host* mnt4_twist_coeff_b_host;
 mnt4_Fq_host* mnt4_twist_mul_by_a_c0_host;
 mnt4_Fq_host* mnt4_twist_mul_by_a_c1_host;
 mnt4_Fq_host* mnt4_twist_mul_by_b_c0_host;
 mnt4_Fq_host* mnt4_twist_mul_by_b_c1_host;
 mnt4_Fq_host* mnt4_twist_mul_by_q_X_host;
 mnt4_Fq_host* mnt4_twist_mul_by_q_Y_host;

struct mnt4_G1_params_host;
struct mnt4_G2_params_host;

mnt4_G1_params_host g1_params_host;
mnt4_G2_params_host g2_params_host;


void init_mnt4_params_host()
{
    typedef bigint_host<mnt4_r_limbs_host> bigint_r;
    typedef bigint_host<mnt4_q_limbs_host> bigint_q;

    // fr
    mnt4_fp_params_r_host.modulus = new bigint_r("475922286169261325753349249653048451545124878552823515553267735739164647307408490559963137");
    if (sizeof(mp_limb_t) == 8)
    {
        mnt4_fp_params_r_host.Rsquared = new bigint_r("163983144722506446826715124368972380525894397127205577781234305496325861831001705438796139");
        mnt4_fp_params_r_host.Rcubed = new bigint_r("207236281459091063710247635236340312578688659363066707916716212805695955118593239854980171");
        mnt4_fp_params_r_host.inv = 0xbb4334a3ffffffff;
        
    }
    if (sizeof(mp_limb_t) == 4)
    {
        mnt4_fp_params_r_host.Rsquared = new bigint_r("163983144722506446826715124368972380525894397127205577781234305496325861831001705438796139");
        mnt4_fp_params_r_host.Rcubed = new bigint_r("207236281459091063710247635236340312578688659363066707916716212805695955118593239854980171");
        mnt4_fp_params_r_host.inv = 0xffffffff;
    }

    mnt4_fp_params_r_host.num_bits = 298;
    mnt4_fp_params_r_host.euler = new bigint_r("237961143084630662876674624826524225772562439276411757776633867869582323653704245279981568");
    mnt4_fp_params_r_host.s = 34;
    mnt4_fp_params_r_host.t = new bigint_r("27702323054502562488973446286577291993024111641153199339359284829066871159442729");
    mnt4_fp_params_r_host.t_minus_1_over_2 = new bigint_r("13851161527251281244486723143288645996512055820576599669679642414533435579721364");
    mnt4_fp_params_r_host.multiplicative_generator = new bigint_r("10");
    mnt4_fp_params_r_host.root_of_unity = new bigint_r("120638817826913173458768829485690099845377008030891618010109772937363554409782252579816313");
    mnt4_fp_params_r_host.nqr = new bigint_r("5");
    mnt4_fp_params_r_host.nqr_to_t = new bigint_r("406220604243090401056429458730298145937262552508985450684842547562990900634752279902740880");
    mnt4_fp_params_r_host.zero = new mnt4_Fr_host(&mnt4_fp_params_r_host);
    mnt4_fp_params_r_host.zero->init_zero();
    mnt4_fp_params_r_host.one = new mnt4_Fr_host(&mnt4_fp_params_r_host);
    mnt4_fp_params_r_host.one->init_one();

    //fq
    mnt4_fp_params_q_host.modulus = new bigint_q("475922286169261325753349249653048451545124879242694725395555128576210262817955800483758081");
    if (sizeof(mp_limb_t) == 8)
    {
        mnt4_fp_params_q_host.Rsquared = new bigint_q("273000478523237720910981655601160860640083126627235719712980612296263966512828033847775776");
        mnt4_fp_params_q_host.Rcubed = new bigint_q("427298980065529822574935274648041073124704261331681436071990730954930769758106792920349077");
        mnt4_fp_params_q_host.inv = 0xb071a1b67165ffff;
    }
    if (sizeof(mp_limb_t) == 4)
    {
        mnt4_fp_params_q_host.Rsquared = new bigint_q("273000478523237720910981655601160860640083126627235719712980612296263966512828033847775776");
        mnt4_fp_params_q_host.Rcubed = new bigint_q("427298980065529822574935274648041073124704261331681436071990730954930769758106792920349077");
        mnt4_fp_params_q_host.inv = 0x7165ffff;
    }

    mnt4_fp_params_q_host.num_bits = 298;
    mnt4_fp_params_q_host.euler = new bigint_q("237961143084630662876674624826524225772562439621347362697777564288105131408977900241879040");
    mnt4_fp_params_q_host.s = 17;
    mnt4_fp_params_q_host.t = new bigint_q("3630998887399759870554727551674258816109656366292531779446068791017229177993437198515");
    mnt4_fp_params_q_host.t_minus_1_over_2 = new bigint_q("1815499443699879935277363775837129408054828183146265889723034395508614588996718599257");
    mnt4_fp_params_q_host.multiplicative_generator = new bigint_q("17");
    mnt4_fp_params_q_host.root_of_unity = new bigint_q("264706250571800080758069302369654305530125675521263976034054878017580902343339784464690243");
    mnt4_fp_params_q_host.nqr = new bigint_q("17");
    mnt4_fp_params_q_host.nqr_to_t = new bigint_q("264706250571800080758069302369654305530125675521263976034054878017580902343339784464690243");
    mnt4_fp_params_q_host.zero = new mnt4_Fr_host(&mnt4_fp_params_q_host);
    mnt4_fp_params_q_host.zero->init_zero();
    mnt4_fp_params_q_host.one = new mnt4_Fr_host(&mnt4_fp_params_q_host);
    mnt4_fp_params_q_host.one->init_one();

    //fq2
    mnt4_fp2_params_q_host.fp_params = &mnt4_fp_params_q_host;
    mnt4_fp2_params_q_host.euler = new bigint_host<2*mnt4_q_limbs_host>("113251011236288135098249345249154230895914381858788918106847214243419142422924133497460817468249854833067260038985710370091920860837014281886963086681184370139950267830740466401280");
    mnt4_fp2_params_q_host.s = 18;
    mnt4_fp2_params_q_host.t = new bigint_host<2*mnt4_q_limbs_host>("864036645784668999467844736092790457885088972921668381552484239528039111503022258739172496553419912972009735404859240494475714575477709059806542104196047745818712370534824115");
    mnt4_fp2_params_q_host.t_minus_1_over_2 = new bigint_host<2 * mnt4_q_limbs_host>("432018322892334499733922368046395228942544486460834190776242119764019555751511129369586248276709956486004867702429620247237857287738854529903271052098023872909356185267412057");
    mnt4_fp2_params_q_host.non_residue = new bigint_q("17");
    mnt4_fp2_params_q_host.nqr_c0 = new bigint_q("8");
    mnt4_fp2_params_q_host.nqr_c1 = new bigint_q("1");
    mnt4_fp2_params_q_host.nqr_to_t_c0 = new bigint_q("0");
    mnt4_fp2_params_q_host.nqr_to_t_c1 = new bigint_q("29402818985595053196743631544512156561638230562612542604956687802791427330205135130967658");
    mnt4_fp2_params_q_host.Frobenius_coeffs_c1[0] = new bigint_q("1");
    mnt4_fp2_params_q_host.Frobenius_coeffs_c1[1] = new bigint_q("475922286169261325753349249653048451545124879242694725395555128576210262817955800483758080");

    //fq4
    mnt4_fp4_params_q_host.fp2_params = &mnt4_fp2_params_q_host;
    mnt4_fp4_params_q_host.non_residue = new bigint_q("17");
    mnt4_fp4_params_q_host.Frobenius_coeffs_c1[0] = new bigint_q("1");
    mnt4_fp4_params_q_host.Frobenius_coeffs_c1[1] = new bigint_q("7684163245453501615621351552473337069301082060976805004625011694147890954040864167002308");
    mnt4_fp4_params_q_host.Frobenius_coeffs_c1[2] = new bigint_q("475922286169261325753349249653048451545124879242694725395555128576210262817955800483758080");
    mnt4_fp4_params_q_host.Frobenius_coeffs_c1[3] = new bigint_q("468238122923807824137727898100575114475823797181717920390930116882062371863914936316755773");

    //
    g1_params_host.coeff_a = new mnt4_Fq_host(&mnt4_fp_params_q_host, "2");
    g1_params_host.coeff_b = new mnt4_Fq_host(&mnt4_fp_params_q_host, "423894536526684178289416011533888240029318103673896002803341544124054745019340795360841685");

    mnt4_Fq_host non_residue = mnt4_Fq_host(&mnt4_fp_params_q_host, *mnt4_fp2_params_q_host.non_residue);

    mnt4_twist_host = new mnt4_Fq2_host(&mnt4_fp2_params_q_host, *mnt4_fp_params_q_host.zero, *mnt4_fp_params_q_host.one);
    mnt4_twist_coeff_a_host = new mnt4_Fq2_host(&mnt4_fp2_params_q_host, *g1_params_host.coeff_a * non_residue, *mnt4_fp_params_q_host.zero);
    mnt4_twist_coeff_b_host = new mnt4_Fq2_host(&mnt4_fp2_params_q_host, *mnt4_fp_params_q_host.zero, *g1_params_host.coeff_b * non_residue);

    g2_params_host.twist = new mnt4_Fq2_host(*mnt4_twist_host);
    g2_params_host.coeff_a = new mnt4_Fq2_host(*mnt4_twist_coeff_a_host);
    g2_params_host.coeff_b = new mnt4_Fq2_host(*mnt4_twist_coeff_b_host);
    mnt4_twist_mul_by_a_c0_host = new mnt4_Fq_host(*g1_params_host.coeff_a * non_residue);
    mnt4_twist_mul_by_a_c1_host = new mnt4_Fq_host(*g1_params_host.coeff_a * non_residue);
    mnt4_twist_mul_by_b_c0_host = new mnt4_Fq_host(*g1_params_host.coeff_b * non_residue.squared());
    mnt4_twist_mul_by_b_c1_host = new mnt4_Fq_host(*g1_params_host.coeff_b * non_residue);
    mnt4_twist_mul_by_q_X_host = new mnt4_Fq_host(&mnt4_fp_params_q_host, "475922286169261325753349249653048451545124879242694725395555128576210262817955800483758080");
    mnt4_twist_mul_by_q_Y_host = new mnt4_Fq_host(&mnt4_fp_params_q_host, "7684163245453501615621351552473337069301082060976805004625011694147890954040864167002308");


    // g1
    mnt4_Fq_host fq_t(&mnt4_fp_params_q_host);
    g1_params_host.fr_params = &mnt4_fp_params_r_host;
    g1_params_host.fq_params = &mnt4_fp_params_q_host;
    g1_params_host.G1_zero_X = new mnt4_Fq_host(fq_t.zero());
    g1_params_host.G1_zero_Y = new mnt4_Fq_host(fq_t.one());
    g1_params_host.G1_zero_Z = new mnt4_Fq_host(fq_t.zero());
    g1_params_host.G1_one_X = new mnt4_Fq_host(&mnt4_fp_params_q_host, "60760244141852568949126569781626075788424196370144486719385562369396875346601926534016838");
    g1_params_host.G1_one_Y = new mnt4_Fq_host(&mnt4_fp_params_q_host, "363732850702582978263902770815145784459747722357071843971107674179038674942891694705904306");
    g1_params_host.G1_one_Z = new mnt4_Fq_host(fq_t.one());

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
    g1_params_host.fixed_base_exp_window_table_length[2] = 10;
    g1_params_host.fixed_base_exp_window_table_length[3] = 25;
    g1_params_host.fixed_base_exp_window_table_length[4] = 60;
    g1_params_host.fixed_base_exp_window_table_length[5] = 144;
    g1_params_host.fixed_base_exp_window_table_length[6] = 345;
    g1_params_host.fixed_base_exp_window_table_length[7] = 855;
    g1_params_host.fixed_base_exp_window_table_length[8] = 1805;
    g1_params_host.fixed_base_exp_window_table_length[9] = 3912;
    g1_params_host.fixed_base_exp_window_table_length[10] = 11265;
    g1_params_host.fixed_base_exp_window_table_length[11] = 27898;
    g1_params_host.fixed_base_exp_window_table_length[12] = 57597;
    g1_params_host.fixed_base_exp_window_table_length[13] = 145299;
    g1_params_host.fixed_base_exp_window_table_length[14] = 157205;
    g1_params_host.fixed_base_exp_window_table_length[15] = 601601;
    g1_params_host.fixed_base_exp_window_table_length[16] = 1107377;
    g1_params_host.fixed_base_exp_window_table_length[17] = 1789647;
    g1_params_host.fixed_base_exp_window_table_length[18] = 4392627;
    g1_params_host.fixed_base_exp_window_table_length[19] = 8221211;
    g1_params_host.fixed_base_exp_window_table_length[20] = 0;
    g1_params_host.fixed_base_exp_window_table_length[21] = 42363731;

    // g2
    mnt4_Fq2_host fq2_t(&mnt4_fp2_params_q_host);
    g2_params_host.fr_params = &mnt4_fp_params_r_host;
    g2_params_host.fq_params = &mnt4_fp_params_q_host;
    g2_params_host.fq2_params = &mnt4_fp2_params_q_host;

    g2_params_host.G2_zero_X = new mnt4_Fq2_host(fq2_t.zero());
    g2_params_host.G2_zero_Y = new mnt4_Fq2_host(fq2_t.one());
    g2_params_host.G2_zero_Z = new mnt4_Fq2_host(fq2_t.zero());

    g2_params_host.G2_one_X = new mnt4_Fq2_host(&mnt4_fp2_params_q_host, 
                                            mnt4_Fq_host(&mnt4_fp_params_q_host, "438374926219350099854919100077809681842783509163790991847867546339851681564223481322252708"),
                                            mnt4_Fq_host(&mnt4_fp_params_q_host, "37620953615500480110935514360923278605464476459712393277679280819942849043649216370485641"));
    g2_params_host.G2_one_Y = new mnt4_Fq2_host(&mnt4_fp2_params_q_host, 
                                            mnt4_Fq_host(&mnt4_fp_params_q_host, "37437409008528968268352521034936931842973546441370663118543015118291998305624025037512482"),
                                            mnt4_Fq_host(&mnt4_fp_params_q_host, "424621479598893882672393190337420680597584695892317197646113820787463109735345923009077489"));
    g2_params_host.G2_one_Z = new mnt4_Fq2_host(fq2_t.one());

    g2_params_host.wnaf_window_table = new size_t[4];
    g2_params_host.wnaf_window_table_size = 4;
    g2_params_host.wnaf_window_table[0] = 5;
    g2_params_host.wnaf_window_table[1] = 15;
    g2_params_host.wnaf_window_table[2] = 39;
    g2_params_host.wnaf_window_table[3] = 109;

    g2_params_host.fixed_base_exp_window_table_length = new size_t[22];
    g2_params_host.fixed_base_exp_window_table_length_size = 22;
    g2_params_host.fixed_base_exp_window_table_length[0] = 1;
    g2_params_host.fixed_base_exp_window_table_length[1] = 4;
    g2_params_host.fixed_base_exp_window_table_length[2] = 10;
    g2_params_host.fixed_base_exp_window_table_length[3] = 25;
    g2_params_host.fixed_base_exp_window_table_length[4] = 60;
    g2_params_host.fixed_base_exp_window_table_length[5] = 143;
    g2_params_host.fixed_base_exp_window_table_length[6] = 345;
    g2_params_host.fixed_base_exp_window_table_length[7] = 821;
    g2_params_host.fixed_base_exp_window_table_length[8] = 1794;
    g2_params_host.fixed_base_exp_window_table_length[9] = 3920;
    g2_params_host.fixed_base_exp_window_table_length[10] = 11301;
    g2_params_host.fixed_base_exp_window_table_length[11] = 18960;
    g2_params_host.fixed_base_exp_window_table_length[12] = 44199;
    g2_params_host.fixed_base_exp_window_table_length[13] = 0;
    g2_params_host.fixed_base_exp_window_table_length[14] = 150800;
    g2_params_host.fixed_base_exp_window_table_length[15] = 548695;
    g2_params_host.fixed_base_exp_window_table_length[16] = 1051769;
    g2_params_host.fixed_base_exp_window_table_length[17] = 2023926;
    g2_params_host.fixed_base_exp_window_table_length[18] = 3787109;
    g2_params_host.fixed_base_exp_window_table_length[19] = 7107480;
    g2_params_host.fixed_base_exp_window_table_length[20] = 0;
    g2_params_host.fixed_base_exp_window_table_length[21] = 38760027;
}

}

