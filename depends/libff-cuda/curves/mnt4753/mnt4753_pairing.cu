#ifndef __MNT4753_PAIRING_CU__
#define __MNT4753_PAIRING_CU__

#include "../../scalar_multiplication/wnaf.cuh"

namespace libff {


__device__ bool mnt4753_ate_G1_precomp::operator==(const mnt4753_ate_G1_precomp &other) const
{
    return (this->PX == other.PX &&
            this->PY == other.PY &&
            this->PX_twist == other.PX_twist &&
            this->PY_twist == other.PY_twist);
}


__device__ bool mnt4753_ate_dbl_coeffs::operator==(const mnt4753_ate_dbl_coeffs &other) const
{
    return (this->c_H == other.c_H &&
            this->c_4C == other.c_4C &&
            this->c_J == other.c_J &&
            this->c_L == other.c_L);
}

__device__ bool mnt4753_ate_add_coeffs::operator==(const mnt4753_ate_add_coeffs &other) const
{
    return (this->c_L1 == other.c_L1 &&
            this->c_RZ == other.c_RZ);
}

__device__ bool mnt4753_ate_G2_precomp::operator==(const mnt4753_ate_G2_precomp &other) const
{
    bool same = (this->QX == other.QX &&
                 this->QY == other.QY &&
                 this->QY2 == other.QY2 &&
                 this->QX_over_twist == other.QX_over_twist &&
                 this->QY_over_twist == other.QY_over_twist &&
                 this->dbl_coeffs_size == other.dbl_coeffs_size &&
                 this->add_coeffs_size == other.add_coeffs_size);

    if (!same)
        return false;

    for (size_t i = 0; i < dbl_coeffs_size; i++)
        if (!(this->dbl_coeffs[i] == other.dbl_coeffs[i]))
            return false;

    
    for (size_t i = 0; i < add_coeffs_size; i++)
        if (!(this->add_coeffs[i] == other.add_coeffs[i]))
            return false;
 
    return true;
}

__device__ mnt4753_Fq4 mnt4753_final_exponentiation_last_chunk(const mnt4753_Fq4 &elt, const mnt4753_Fq4 &elt_inv)
{
    const mnt4753_Fq4 elt_q = elt.Frobenius_map(1);
    mnt4753_Fq4 w1_part = elt_q.cyclotomic_exp(*mnt4753_final_exponent_last_chunk_w1);
    mnt4753_Fq4 w0_part;

    if (mnt4753_final_exponent_last_chunk_is_w0_neg)
    	w0_part = elt_inv.cyclotomic_exp(*mnt4753_final_exponent_last_chunk_abs_of_w0);
    else 
    	w0_part = elt.cyclotomic_exp(*mnt4753_final_exponent_last_chunk_abs_of_w0);

    mnt4753_Fq4 result = w1_part * w0_part;
    return result;
}

__device__ mnt4753_Fq4 mnt4753_final_exponentiation_first_chunk(const mnt4753_Fq4 &elt, const mnt4753_Fq4 &elt_inv)
{
    /* (q^2-1) */

    /* elt_q2 = elt^(q^2) */
    const mnt4753_Fq4 elt_q2 = elt.Frobenius_map(2);
    /* elt_q3_over_elt = elt^(q^2-1) */
    const mnt4753_Fq4 elt_q2_over_elt = elt_q2 * elt_inv;

    return elt_q2_over_elt;
}

__device__ mnt4753_GT mnt4753_final_exponentiation(const mnt4753_Fq4 &elt)
{
    const mnt4753_Fq4 elt_inv = elt.inverse();
    const mnt4753_Fq4 elt_to_first_chunk = mnt4753_final_exponentiation_first_chunk(elt, elt_inv);
    const mnt4753_Fq4 elt_inv_to_first_chunk = mnt4753_final_exponentiation_first_chunk(elt_inv, elt);
    
    mnt4753_GT result = mnt4753_final_exponentiation_last_chunk(elt_to_first_chunk, elt_inv_to_first_chunk);
    return result;
}

__device__ mnt4753_affine_ate_G1_precomputation mnt4753_affine_ate_precompute_G1(const mnt4753_G1& P)
{
    mnt4753_G1 Pcopy = P;
    Pcopy.to_affine_coordinates();

    mnt4753_affine_ate_G1_precomputation result;
    result.PX = Pcopy.X;
    result.PY = Pcopy.Y;
    result.PY_twist_squared = Pcopy.Y * mnt4753_twist->squared();

    return result;
}

__device__ mnt4753_affine_ate_G2_precomputation mnt4753_affine_ate_precompute_G2(const mnt4753_G2& Q)
{
    mnt4753_G2 Qcopy(Q);
    Qcopy.to_affine_coordinates();

    mnt4753_affine_ate_G2_precomputation result;
    result.QX = Qcopy.X;
    result.QY = Qcopy.Y;

    mnt4753_Fq2 RX = Qcopy.X;
    mnt4753_Fq2 RY = Qcopy.Y;

    long* NAF;
    size_t NAF_size;
    find_wnaf(&NAF, &NAF_size, 1, *mnt4753_ate_loop_count);

    bool found_nonzero = false;

    for (long i = NAF_size - 1; i >= 0; --i)
    {
        if (!found_nonzero)
        {
            /* this skips the MSB itself */
            found_nonzero |= (NAF[i] != 0);
            continue;
        }

        ++result.coeffs_size;

        if (NAF[i] != 0)
            ++result.coeffs_size;
    }

    result.coeffs = new mnt4753_affine_ate_coeffs[result.coeffs_size];

    size_t idx = 0;
    found_nonzero = false;
    
    for (long i = NAF_size - 1; i >= 0; --i)
    {
        if (!found_nonzero)
        {
            /* this skips the MSB itself */
            found_nonzero |= (NAF[i] != 0);
            continue;
        }

        mnt4753_affine_ate_coeffs c;
        c.old_RX = RX;
        c.old_RY = RY;
        mnt4753_Fq2 old_RX_2 = c.old_RX.squared();
        c.gamma = (old_RX_2 + old_RX_2 + old_RX_2 + *mnt4753_twist_coeff_a) * (c.old_RY + c.old_RY).inverse();
        c.gamma_twist = c.gamma * *mnt4753_twist;
        c.gamma_X = c.gamma * c.old_RX;
        result.coeffs[idx++] = c;

        RX = c.gamma.squared() - (c.old_RX+c.old_RX);
        RY = c.gamma * (c.old_RX - RX) - c.old_RY;

        if (NAF[i] != 0)
        {
            mnt4753_affine_ate_coeffs c;
            c.old_RX = RX;
            c.old_RY = RY;
            if (NAF[i] > 0)
                c.gamma = (c.old_RY - result.QY) * (c.old_RX - result.QX).inverse();
            else
                c.gamma = (c.old_RY + result.QY) * (c.old_RX - result.QX).inverse();

            c.gamma_twist = c.gamma * *mnt4753_twist;
            c.gamma_X = c.gamma * result.QX;
            result.coeffs[idx++] = c;

            RX = c.gamma.squared() - (c.old_RX + result.QX);
            RY = c.gamma * (c.old_RX - RX) - c.old_RY;
        }
    }

    return result;
}

__device__ mnt4753_Fq4 mnt4753_affine_ate_miller_loop(const mnt4753_affine_ate_G1_precomputation &prec_P,
                                                const mnt4753_affine_ate_G2_precomputation &prec_Q)
{
    mnt4753_Fq4 t(&mnt4753_fp4_params_q);
    mnt4753_Fq4 f = t.one();

    bool found_nonzero = false;
    size_t idx = 0;
   
    long* NAF;
    size_t NAF_size;
    find_wnaf(&NAF, &NAF_size, 1, *mnt4753_ate_loop_count);

    for (long i = NAF_size - 1; i >= 0; --i)
    {
        if (!found_nonzero)
        {
            /* this skips the MSB itself */
            found_nonzero |= (NAF[i] != 0);
            continue;
        }

        /* code below gets executed for all bits (EXCEPT the MSB itself) of
           mnt4753_param_p (skipping leading zeros) in MSB to LSB
           order */
        mnt4753_affine_ate_coeffs c = prec_Q.coeffs[idx++];

        mnt4753_Fq4 g_RR_at_P = mnt4753_Fq4(&mnt4753_fp4_params_q, prec_P.PY_twist_squared, - prec_P.PX * c.gamma_twist + c.gamma_X - c.old_RY);
        f = f.squared().mul_by_023(g_RR_at_P);

        if (NAF[i] != 0)
        {
            mnt4753_affine_ate_coeffs c = prec_Q.coeffs[idx++];
            mnt4753_Fq4 g_RQ_at_P;

            if (NAF[i] > 0)
                g_RQ_at_P = mnt4753_Fq4(&mnt4753_fp4_params_q, prec_P.PY_twist_squared, - prec_P.PX * c.gamma_twist + c.gamma_X - prec_Q.QY);
            else
                g_RQ_at_P = mnt4753_Fq4(&mnt4753_fp4_params_q, prec_P.PY_twist_squared, - prec_P.PX * c.gamma_twist + c.gamma_X + prec_Q.QY);

            f = f.mul_by_023(g_RQ_at_P);
        }
    }

    return f;
}

/* ate pairing */

struct extended_mnt4753_G2_projective {
    mnt4753_Fq2 X;
    mnt4753_Fq2 Y;
    mnt4753_Fq2 Z;
    mnt4753_Fq2 T;
};

__device__ void doubling_step_for_flipped_miller_loop(extended_mnt4753_G2_projective &current, mnt4753_ate_dbl_coeffs &dc)
{
    const mnt4753_Fq2 X = current.X, Y = current.Y, Z = current.Z, T = current.T;

    const mnt4753_Fq2 A = T.squared(); // A = T1^2
    const mnt4753_Fq2 B = X.squared(); // B = X1^2
    const mnt4753_Fq2 C = Y.squared(); // C = Y1^2
    const mnt4753_Fq2 D = C.squared(); // D = C^2
    const mnt4753_Fq2 E = (X + C).squared() - B - D;               // E = (X1+C)^2-B-D
    const mnt4753_Fq2 F = (B.dbl() + B) + *mnt4753_twist_coeff_a * A;  // F = 3*B +  a  *A
    const mnt4753_Fq2 G = F.squared(); // G = F^2

    current.X = -(E.dbl().dbl()) + G; // X3 = -4*E+G
    current.Y = -mnt4753_Fq(&mnt4753_fp_params_q, "8") * D + F * (E.dbl() - current.X); // Y3 = -8*D+F*(2*E-X3)
    current.Z = (Y + Z).squared() - C - Z.squared();           // Z3 = (Y1+Z1)^2-C-Z1^2
    current.T = current.Z.squared();  // T3 = Z3^2

    dc.c_H = (current.Z + T).squared() - current.T - A; // H = (Z3+T1)^2-T3-A
    dc.c_4C = C.dbl().dbl();            // fourC = 4*C
    dc.c_J = (F + T).squared() - G - A; // J = (F+T1)^2-G-A
    dc.c_L = (F + X).squared() - G - B; // L = (F+X1)^2-G-B
}

__device__ void mixed_addition_step_for_flipped_miller_loop(const mnt4753_Fq2 base_X, const mnt4753_Fq2 base_Y, const mnt4753_Fq2 base_Y_squared,
                                                            extended_mnt4753_G2_projective &current,
                                                            mnt4753_ate_add_coeffs &ac)
{
    const mnt4753_Fq2 X1 = current.X, Y1 = current.Y, Z1 = current.Z, T1 = current.T;
    const mnt4753_Fq2 &x2 = base_X,   &y2 =  base_Y,  &y2_squared = base_Y_squared;

    const mnt4753_Fq2 B = x2 * T1; // B = x2 * T1
    const mnt4753_Fq2 D = ((y2 + Z1).squared() - y2_squared - T1) * T1; // D = ((y2 + Z1)^2 - y2squared - T1) * T1
    const mnt4753_Fq2 H = B - X1;        // H = B - X1
    const mnt4753_Fq2 I = H.squared();   // I = H^2
    const mnt4753_Fq2 E = I.dbl().dbl(); // E = 4*I
    const mnt4753_Fq2 J = H * E;         // J = H * E
    const mnt4753_Fq2 V = X1 * E;        // V = X1 * E
    const mnt4753_Fq2 L1 = D - Y1.dbl(); // L1 = D - 2 * Y1

    current.X = L1.squared() - J - V.dbl();          // X3 = L1^2 - J - 2*V
    current.Y = L1 * (V - current.X) - Y1.dbl() * J; // Y3 = L1 * (V-X3) - 2*Y1 * J
    current.Z = (Z1 + H).squared() - T1 - I;         // Z3 = (Z1 + H)^2 - T1 - I
    current.T = current.Z.squared();                 // T3 = Z3^2

    ac.c_L1 = L1;
    ac.c_RZ = current.Z;

}

__device__ mnt4753_ate_G1_precomp mnt4753_ate_precompute_G1(const mnt4753_G1& P)
{
    mnt4753_G1 Pcopy = P;
    Pcopy.to_affine_coordinates();

    mnt4753_ate_G1_precomp result;
    result.PX = Pcopy.X;
    result.PY = Pcopy.Y;
    result.PX_twist = Pcopy.X * *mnt4753_twist;
    result.PY_twist = Pcopy.Y * *mnt4753_twist;

    return result;
}

__device__ mnt4753_ate_G2_precomp mnt4753_ate_precompute_G2(const mnt4753_G2& Q)
{
    mnt4753_Fq2 t(&mnt4753_fp2_params_q);

    mnt4753_G2 Qcopy(Q);
    Qcopy.to_affine_coordinates();

    mnt4753_ate_G2_precomp result;
    result.QX = Qcopy.X;
    result.QY = Qcopy.Y;
    result.QY2 = Qcopy.Y.squared();

    result.QX_over_twist = Qcopy.X * mnt4753_twist->inverse();
    result.QY_over_twist = Qcopy.Y * mnt4753_twist->inverse();

    extended_mnt4753_G2_projective R;
    R.X = Qcopy.X;
    R.Y = Qcopy.Y;
    R.Z = t.one();
    R.T = t.one();

    const bigint<mnt4753_r_limbs> &loop_count = *mnt4753_ate_loop_count;
    bool found_one = false;

    for (long i = loop_count.max_bits() - 1; i >= 0; --i)
    {
        const bool bit = loop_count.test_bit(i);
        if (!found_one)
        {
            /* this skips the MSB itself */
            found_one |= bit;
            continue;
        }

        ++result.dbl_coeffs_size;
        if (bit)
            ++result.add_coeffs_size;
    }

    if (mnt4753_ate_is_loop_count_neg)
        ++result.add_coeffs_size;

    result.dbl_coeffs = new mnt4753_ate_dbl_coeffs[result.dbl_coeffs_size];
    result.add_coeffs = new mnt4753_ate_add_coeffs[result.add_coeffs_size];

    found_one = false;

    size_t dbl_idx = 0;
    size_t add_idx = 0;

    for (long i = loop_count.max_bits() - 1; i >= 0; --i)
    {
        const bool bit = loop_count.test_bit(i);
        if (!found_one)
        {
            /* this skips the MSB itself */
            found_one |= bit;
            continue;
        }

        mnt4753_ate_dbl_coeffs dc;
        doubling_step_for_flipped_miller_loop(R, dc);
        result.dbl_coeffs[dbl_idx++] = dc;

        if (bit)
        {
            mnt4753_ate_add_coeffs ac;
            mixed_addition_step_for_flipped_miller_loop(result.QX, result.QY, result.QY2, R, ac);
            result.add_coeffs[add_idx++] = ac;
        }
    }

    if (mnt4753_ate_is_loop_count_neg)
    {
    	mnt4753_Fq2 RZ_inv = R.Z.inverse();
    	mnt4753_Fq2 RZ2_inv = RZ_inv.squared();
    	mnt4753_Fq2 RZ3_inv = RZ2_inv * RZ_inv;
    	mnt4753_Fq2 minus_R_affine_X = R.X * RZ2_inv;
    	mnt4753_Fq2 minus_R_affine_Y = - R.Y * RZ3_inv;
    	mnt4753_Fq2 minus_R_affine_Y2 = minus_R_affine_Y.squared();
    	mnt4753_ate_add_coeffs ac;
        mixed_addition_step_for_flipped_miller_loop(minus_R_affine_X, minus_R_affine_Y, minus_R_affine_Y2, R, ac);
        result.add_coeffs[add_idx++] = ac;
    }

    return result;
}

__device__ mnt4753_Fq4 mnt4753_ate_miller_loop(const mnt4753_ate_G1_precomp &prec_P, const mnt4753_ate_G2_precomp &prec_Q)
{
    mnt4753_Fq  t(&mnt4753_fp_params_q);
    mnt4753_Fq4 t4(&mnt4753_fp4_params_q);

    mnt4753_Fq2 L1_coeff = mnt4753_Fq2(&mnt4753_fp2_params_q, prec_P.PX, t.zero()) - prec_Q.QX_over_twist;

    mnt4753_Fq4 f = t4.one();

    bool found_one = false;
    size_t dbl_idx = 0;
    size_t add_idx = 0;

    const bigint<mnt4753_r_limbs> &loop_count = *mnt4753_ate_loop_count;
    for (long i = loop_count.max_bits() - 1; i >= 0; --i)
    {
        const bool bit = loop_count.test_bit(i);

        if (!found_one)
        {
            /* this skips the MSB itself */
            found_one |= bit;
            continue;
        }

        /* code below gets executed for all bits (EXCEPT the MSB itself) of
           mnt4753_param_p (skipping leading zeros) in MSB to LSB
           order */
        mnt4753_ate_dbl_coeffs dc = prec_Q.dbl_coeffs[dbl_idx++];

        mnt4753_Fq4 g_RR_at_P = mnt4753_Fq4(&mnt4753_fp4_params_q, - dc.c_4C - dc.c_J * prec_P.PX_twist + dc.c_L,  dc.c_H * prec_P.PY_twist);
        f = f.squared() * g_RR_at_P;

        if (bit)
        {
            mnt4753_ate_add_coeffs ac = prec_Q.add_coeffs[add_idx++];
            mnt4753_Fq4 g_RQ_at_P = mnt4753_Fq4(&mnt4753_fp4_params_q, ac.c_RZ * prec_P.PY_twist, -(prec_Q.QY_over_twist * ac.c_RZ + L1_coeff * ac.c_L1));
            f = f * g_RQ_at_P;
        }
    }

    if (mnt4753_ate_is_loop_count_neg)
    {
    	mnt4753_ate_add_coeffs ac = prec_Q.add_coeffs[add_idx++];
    	mnt4753_Fq4 g_RnegR_at_P = mnt4753_Fq4(&mnt4753_fp4_params_q, ac.c_RZ * prec_P.PY_twist, -(prec_Q.QY_over_twist * ac.c_RZ + L1_coeff * ac.c_L1));
    	f = (f * g_RnegR_at_P).inverse();
    }

    return f;
}

__device__ mnt4753_Fq4 mnt4753_ate_double_miller_loop(const mnt4753_ate_G1_precomp &prec_P1,
                                                const mnt4753_ate_G2_precomp &prec_Q1,
                                                const mnt4753_ate_G1_precomp &prec_P2,
                                                const mnt4753_ate_G2_precomp &prec_Q2)
{
    mnt4753_Fq  t(&mnt4753_fp_params_q);
    mnt4753_Fq4 t4(&mnt4753_fp4_params_q);

    mnt4753_Fq2 L1_coeff1 = mnt4753_Fq2(&mnt4753_fp2_params_q, prec_P1.PX, t.zero()) - prec_Q1.QX_over_twist;
    mnt4753_Fq2 L1_coeff2 = mnt4753_Fq2(&mnt4753_fp2_params_q, prec_P2.PX, t.zero()) - prec_Q2.QX_over_twist;

    mnt4753_Fq4 f = t4.one();

    bool found_one = false;
    size_t dbl_idx = 0;
    size_t add_idx = 0;

    const bigint<mnt4753_r_limbs> &loop_count = *mnt4753_ate_loop_count;
    for (long i = loop_count.max_bits() - 1; i >= 0; --i)
    {
        const bool bit = loop_count.test_bit(i);

        if (!found_one)
        {
            /* this skips the MSB itself */
            found_one |= bit;
            continue;
        }

        /* code below gets executed for all bits (EXCEPT the MSB itself) of
           mnt4753_param_p (skipping leading zeros) in MSB to LSB
           order */
        mnt4753_ate_dbl_coeffs dc1 = prec_Q1.dbl_coeffs[dbl_idx];
        mnt4753_ate_dbl_coeffs dc2 = prec_Q2.dbl_coeffs[dbl_idx];
        ++dbl_idx;

        mnt4753_Fq4 g_RR_at_P1 = mnt4753_Fq4(&mnt4753_fp4_params_q, - dc1.c_4C - dc1.c_J * prec_P1.PX_twist + dc1.c_L, dc1.c_H * prec_P1.PY_twist);
        mnt4753_Fq4 g_RR_at_P2 = mnt4753_Fq4(&mnt4753_fp4_params_q, - dc2.c_4C - dc2.c_J * prec_P2.PX_twist + dc2.c_L, dc2.c_H * prec_P2.PY_twist);

        f = f.squared() * g_RR_at_P1 * g_RR_at_P2;

        if (bit)
        {
            mnt4753_ate_add_coeffs ac1 = prec_Q1.add_coeffs[add_idx];
            mnt4753_ate_add_coeffs ac2 = prec_Q2.add_coeffs[add_idx];
            ++add_idx;

            mnt4753_Fq4 g_RQ_at_P1 = mnt4753_Fq4(&mnt4753_fp4_params_q, ac1.c_RZ * prec_P1.PY_twist, -(prec_Q1.QY_over_twist * ac1.c_RZ + L1_coeff1 * ac1.c_L1));
            mnt4753_Fq4 g_RQ_at_P2 = mnt4753_Fq4(&mnt4753_fp4_params_q, ac2.c_RZ * prec_P2.PY_twist, -(prec_Q2.QY_over_twist * ac2.c_RZ + L1_coeff2 * ac2.c_L1));

            f = f * g_RQ_at_P1 * g_RQ_at_P2;
        }
    }

    if (mnt4753_ate_is_loop_count_neg)
    {
    	mnt4753_ate_add_coeffs ac1 = prec_Q1.add_coeffs[add_idx];
        mnt4753_ate_add_coeffs ac2 = prec_Q2.add_coeffs[add_idx];
    	++add_idx;
    	mnt4753_Fq4 g_RnegR_at_P1 = mnt4753_Fq4(&mnt4753_fp4_params_q, ac1.c_RZ * prec_P1.PY_twist, -(prec_Q1.QY_over_twist * ac1.c_RZ + L1_coeff1 * ac1.c_L1));
    	mnt4753_Fq4 g_RnegR_at_P2 = mnt4753_Fq4(&mnt4753_fp4_params_q, ac2.c_RZ * prec_P2.PY_twist, -(prec_Q2.QY_over_twist * ac2.c_RZ + L1_coeff2 * ac2.c_L1));

    	f = (f * g_RnegR_at_P1 * g_RnegR_at_P2).inverse();
    }

    return f;
}

__device__ mnt4753_Fq4 mnt4753_ate_pairing(const mnt4753_G1& P, const mnt4753_G2 &Q)
{
    mnt4753_ate_G1_precomp prec_P = mnt4753_ate_precompute_G1(P);
    mnt4753_ate_G2_precomp prec_Q = mnt4753_ate_precompute_G2(Q);
    mnt4753_Fq4 result = mnt4753_ate_miller_loop(prec_P, prec_Q);

    return result;
}

__device__ mnt4753_GT mnt4753_ate_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q)
{
    const mnt4753_Fq4 f = mnt4753_ate_pairing(P, Q);
    const mnt4753_GT result = mnt4753_final_exponentiation(f);
    return result;
}

__device__ mnt4753_G1_precomp mnt4753_precompute_G1(const mnt4753_G1& P)
{
    return mnt4753_ate_precompute_G1(P);
}

__device__ mnt4753_G2_precomp mnt4753_precompute_G2(const mnt4753_G2& Q)
{
    return mnt4753_ate_precompute_G2(Q);
}

__device__ mnt4753_Fq4 mnt4753_miller_loop(const mnt4753_G1_precomp &prec_P, const mnt4753_G2_precomp &prec_Q)
{
    return mnt4753_ate_miller_loop(prec_P, prec_Q);
}

__device__ mnt4753_Fq4 mnt4753_double_miller_loop(const mnt4753_G1_precomp &prec_P1,
                                            const mnt4753_G2_precomp &prec_Q1,
                                            const mnt4753_G1_precomp &prec_P2,
                                            const mnt4753_G2_precomp &prec_Q2)
{
    return mnt4753_ate_double_miller_loop(prec_P1, prec_Q1, prec_P2, prec_Q2);
}

__device__ mnt4753_Fq4 mnt4753_pairing(const mnt4753_G1& P, const mnt4753_G2 &Q)
{
    return mnt4753_ate_pairing(P, Q);
}

__device__ mnt4753_GT mnt4753_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q)
{
    return mnt4753_ate_reduced_pairing(P, Q);
}

__device__ mnt4753_GT mnt4753_affine_reduced_pairing(const mnt4753_G1 &P, const mnt4753_G2 &Q)
{
    const mnt4753_affine_ate_G1_precomputation prec_P = mnt4753_affine_ate_precompute_G1(P);
    const mnt4753_affine_ate_G2_precomputation prec_Q = mnt4753_affine_ate_precompute_G2(Q);
    const mnt4753_Fq4 f = mnt4753_affine_ate_miller_loop(prec_P, prec_Q);
    const mnt4753_GT result = mnt4753_final_exponentiation(f);
    return result;
}



}


#endif