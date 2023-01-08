#include "mnt4753_g1_host.cuh"

namespace libff {

mnt4753_G1_host::mnt4753_G1_host(mnt4753_G1_params_host* params) : params(params), X(params->fq_params), Y(params->fq_params), Z(params->fq_params)
{
    this->X = *params->G1_zero_X;
    this->Y = *params->G1_zero_Y;
    this->Z = *params->G1_zero_Z;
}

void mnt4753_G1_host::to_affine_coordinates()
{
    mnt4753_Fq_host t(params->fq_params);

    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        mnt4753_Fq_host Z_inv = Z.inverse();
        this->X = this->X * Z_inv;
        this->Y = this->Y * Z_inv;
        this->Z = t.one();
    }
}

 void mnt4753_G1_host::to_special()
{
    this->to_affine_coordinates();
}

 bool mnt4753_G1_host::is_special() const
{
    return this->is_zero() || this->Z == this->Z.one();
}

 bool mnt4753_G1_host::is_zero() const
{
    return (this->X.is_zero() && this->Z.is_zero());
}

 bool mnt4753_G1_host::operator==(const mnt4753_G1_host &other) const
{
    if (this->is_zero())
        return other.is_zero();

    if (other.is_zero())
        return false;

    if ((this->X * other.Z) != (other.X * this->Z))
        return false;

    return !((this->Y * other.Z) != (other.Y * this->Z));
}

 bool mnt4753_G1_host::operator!=(const mnt4753_G1_host& other) const
{
    return !(operator==(other));
}

 mnt4753_G1_host mnt4753_G1_host::operator+(const mnt4753_G1_host &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const mnt4753_Fq_host X1Z2 = (this->X) * (other.Z);        // X1Z2 = X1 * Z2
    const mnt4753_Fq_host X2Z1 = (this->Z) * (other.X);        // X2Z1 = X2 * Z1

    // (used both in add and double checks)

    const mnt4753_Fq_host Y1Z2 = (this->Y) * (other.Z);        // Y1Z2 = Y1 * Z2
    const mnt4753_Fq_host Y2Z1 = (this->Z) * (other.Y);        // Y2Z1 = Y2 * Z1

    if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
        return this->dbl(); 

    const mnt4753_Fq_host Z1Z2 = (this->Z) * (other.Z);        // Z1Z2 = Z1 * Z2
    const mnt4753_Fq_host u    = Y2Z1 - Y1Z2;                  // u    = Y2 * Z1 - Y1Z2
    const mnt4753_Fq_host uu   = u.squared();                  // uu   = u^2
    const mnt4753_Fq_host v    = X2Z1 - X1Z2;                  // v    = X2 * Z1 - X1Z2
    const mnt4753_Fq_host vv   = v.squared();                  // vv   = v^2
    const mnt4753_Fq_host vvv  = v * vv;                       // vvv  = v * vv
    const mnt4753_Fq_host R    = vv * X1Z2;                    // R    = vv * X1Z2
    const mnt4753_Fq_host A    = uu * Z1Z2 - (vvv + R.dbl());  // A    = uu * Z1Z2 - vvv - 2 * R

    const mnt4753_Fq_host X3   = v * A;                        // X3   = v * A
    const mnt4753_Fq_host Y3   = u * (R - A) - vvv * Y1Z2;     // Y3   = u * (R -  A) - vvv * Y1Z2
    const mnt4753_Fq_host Z3   = vvv * Z1Z2;                   // Z3   = vvv * Z1Z2


    return mnt4753_G1_host(params, X3, Y3, Z3);
} 

 mnt4753_G1_host mnt4753_G1_host::operator-() const
{
    return mnt4753_G1_host(params, this->X, -(this->Y), this->Z);
}

 mnt4753_G1_host mnt4753_G1_host::operator-(const mnt4753_G1_host &other) const
{
    return (*this) + (-other);
}

 mnt4753_G1_host mnt4753_G1_host::operator*(const unsigned long lhs) const
{
    return scalar_mul_host<mnt4753_G1_host>(*this, lhs);
}

 mnt4753_G1_host mnt4753_G1_host::dbl() const
{
    if (this->is_zero())
        return *this;

    const mnt4753_Fq_host XX   = (this->X).squared();                     // XX  = X1^2
    const mnt4753_Fq_host ZZ   = (this->Z).squared();                     // ZZ  = Z1^2
    const mnt4753_Fq_host w    = *params->coeff_a * ZZ + (XX.dbl() + XX); // w   = a * ZZ + 3 * XX
    const mnt4753_Fq_host Y1Z1 = (this->Y) * (this->Z);
    const mnt4753_Fq_host s    = Y1Z1.dbl();                              // s   = 2 * Y1 * Z1
    const mnt4753_Fq_host ss   = s.squared();                             // ss  = s^2
    const mnt4753_Fq_host sss  = s * ss;                                  // sss = s * ss
    const mnt4753_Fq_host R    = (this->Y) * s;                           // R   = Y1 *  s
    const mnt4753_Fq_host RR   = R.squared();                             // RR  = R^2
    const mnt4753_Fq_host B    = ((this->X) + R).squared()- XX - RR;      // B   = (X1 + R)^2 - XX - RR
    const mnt4753_Fq_host h    = w.squared() - B.dbl();                   // h   = w^2 - 2 * B
    const mnt4753_Fq_host X3   = h * s;                                   // X3  = h * s
    const mnt4753_Fq_host Y3   = w * (B - h) - RR.dbl();                  // Y3  = w * (B - h) - 2 * RR
    const mnt4753_Fq_host Z3   = sss;                                     // Z3  = sss

    return mnt4753_G1_host(params, X3, Y3, Z3);
}

 mnt4753_G1_host mnt4753_G1_host::add(const mnt4753_G1_host &other) const
{
    return (*this) + other;
}

 mnt4753_G1_host mnt4753_G1_host::mixed_add(const mnt4753_G1_host &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;
    
    const mnt4753_Fq_host X2Z1 = (this->Z) * (other.X);             // X2Z1 = X2 * Z1
    const mnt4753_Fq_host Y2Z1 = (this->Z) * (other.Y);             // Y2Z1 = Y 2* Z1

    if (this->X == X2Z1 && this->Y == Y2Z1)
        return this->dbl();

    const mnt4753_Fq_host u = Y2Z1 - this->Y;                       // u = Y2 * Z1 - Y1
    const mnt4753_Fq_host uu = u.squared();                         // uu = u2
    const mnt4753_Fq_host v = X2Z1 - this->X;                       // v = X2 * Z1 - X1
    const mnt4753_Fq_host vv = v.squared();                         // vv = v2
    const mnt4753_Fq_host vvv = v * vv;                             // vvv = v * vv
    const mnt4753_Fq_host R = vv * this->X;                         // R = vv * X1
    const mnt4753_Fq_host A = uu * this->Z - vvv - R.dbl();         // A = uu * Z1 - vv v- 2 * R

    const mnt4753_Fq_host X3 = v * A;                               // X3 = v * A
    const mnt4753_Fq_host Y3 = u * (R - A) - vvv * this->Y;         // Y3 = u * (R - A) - vvv * Y1
    const mnt4753_Fq_host Z3 = vvv * this->Z ;                      // Z3 = vvv * Z1

    return mnt4753_G1_host(params, X3, Y3, Z3);
}

 bool mnt4753_G1_host::is_well_formed() const
{
    if (this->is_zero())
        return true;
    
    mnt4753_Fq_host X2 = this->X.squared();
    mnt4753_Fq_host Y2 = this->Y.squared();
    mnt4753_Fq_host Z2 = this->Z.squared();

    return (this->Z * (Y2 - *params->coeff_b * Z2) == this->X * (X2 + *params->coeff_a * Z2));
}

 mnt4753_G1_host mnt4753_G1_host::zero() const
{
    return mnt4753_G1_host(params, *params->G1_zero_X, *params->G1_zero_Y, *params->G1_zero_Z);
}

 mnt4753_G1_host mnt4753_G1_host::one() const
{
    return mnt4753_G1_host(params, *params->G1_one_X, *params->G1_one_Y, *params->G1_one_Z);
}

 mnt4753_G1_host mnt4753_G1_host::random_element() const
{
    scalar_field t(params->fr_params);
    return (t.random_element().as_bigint()) * this->one();
}

 size_t mnt4753_G1_host::size_in_bits()
{
    base_field t(params->fq_params);
    return t.size_in_bits() + 1;
}

bigint_host<mnt4753_q_limbs_host> mnt4753_G1_host::base_field_char()
{
    base_field t(params->fq_params);
    return t.field_char();
}

bigint_host<mnt4753_r_limbs_host> mnt4753_G1_host::order()
{
    scalar_field t(params->fr_params);
    return t.field_char();
}

void mnt4753_G1_host::set_params(mnt4753_G1_params_host* params)
{
    this->params = params;
    this->X.set_params(params->fq_params);
    this->Y.set_params(params->fq_params);
    this->Z.set_params(params->fq_params);
}


}
