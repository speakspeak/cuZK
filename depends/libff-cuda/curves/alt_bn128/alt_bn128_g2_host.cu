#include "alt_bn128_g2_host.cuh"


namespace libff {

alt_bn128_G2_host::alt_bn128_G2_host(alt_bn128_G2_params_host* params) : params(params), X(params->fq2_params), Y(params->fq2_params), Z(params->fq2_params)
{
    this->X = *params->G2_zero_X;
    this->Y = *params->G2_zero_Y;
    this->Z = *params->G2_zero_Z;
}

alt_bn128_Fq2_host alt_bn128_G2_host::mul_by_b(const alt_bn128_Fq2_host &elt)
{
    return alt_bn128_Fq2_host(params->fq2_params, *alt_bn128_twist_mul_by_b_c0_host * elt.c0, *alt_bn128_twist_mul_by_b_c1_host * elt.c1);
}


void alt_bn128_G2_host::to_affine_coordinates()
{    
    alt_bn128_Fq2_host t(params->fq2_params);

    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        alt_bn128_Fq2_host Z_inv = Z.inverse();
        alt_bn128_Fq2_host Z2_inv = Z_inv.squared();
        alt_bn128_Fq2_host Z3_inv = Z2_inv * Z_inv;
        this->X = this->X * Z2_inv;
        this->Y = this->Y * Z3_inv;
        this->Z = t.one();
    }
}

void alt_bn128_G2_host::to_special()
{
    this->to_affine_coordinates();
}

bool alt_bn128_G2_host::is_special() const
{
    return (this->is_zero() || this->Z == this->Z.one());
}

 bool alt_bn128_G2_host::is_zero() const
{
    return (this->Z.is_zero());
}

 bool alt_bn128_G2_host::operator==(const alt_bn128_G2_host &other) const
{
    if (this->is_zero())
        return other.is_zero();

    if (other.is_zero())
        return false;

    alt_bn128_Fq2_host Z1_squared = this->Z.squared();
    alt_bn128_Fq2_host Z2_squared = other.Z.squared();

    if ((this->X * Z2_squared) != (other.X * Z1_squared))
        return false;

    alt_bn128_Fq2_host Z1_cubed = this->Z * Z1_squared;
    alt_bn128_Fq2_host Z2_cubed = other.Z * Z2_squared;

    return !((this->Y * Z2_cubed) != (other.Y * Z1_cubed));
}

 bool alt_bn128_G2_host::operator!=(const alt_bn128_G2_host& other) const
{
    return !(operator==(other));
}

 alt_bn128_G2_host alt_bn128_G2_host::operator+(const alt_bn128_G2_host &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    alt_bn128_Fq2_host Z1Z1 = this->Z.squared();
    alt_bn128_Fq2_host Z2Z2 = other.Z.squared();

    alt_bn128_Fq2_host U1 = this->X * Z2Z2;
    alt_bn128_Fq2_host U2 = other.X * Z1Z1;

    alt_bn128_Fq2_host Z1_cubed = this->Z * Z1Z1;
    alt_bn128_Fq2_host Z2_cubed = other.Z * Z2Z2;

    alt_bn128_Fq2_host S1 = this->Y * Z2_cubed;      // S1 = Y1 * Z2 * Z2Z2
    alt_bn128_Fq2_host S2 = other.Y * Z1_cubed;      // S2 = Y2 * Z1 * Z1Z1

    if (U1 == U2 && S1 == S2)
        return this->dbl();    // dbl case; nothing of above can be reused

    // rest of add case
    alt_bn128_Fq2_host H = U2 - U1;                            // H = U2-U1
    alt_bn128_Fq2_host I = H.dbl().squared();                  // I = (2 * H)^2
    alt_bn128_Fq2_host J = H * I;                              // J = H * I
    alt_bn128_Fq2_host r = (S2 - S1).dbl();                    // r = 2 * (S2 - S1)
    alt_bn128_Fq2_host V = U1 * I;                             // V = U1 * I
    
    alt_bn128_Fq2_host X3 = r.squared() - J - V.dbl();                           // X3 = r^2 - J - 2 * V
    alt_bn128_Fq2_host Y3 = r * (V - X3) - (S1 * J).dbl();                       // Y3 = r * (V-X3)-2 S1 J
    alt_bn128_Fq2_host Z3 = ((this->Z + other.Z).squared() - Z1Z1 - Z2Z2) * H;   // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2) * H

    return alt_bn128_G2_host(params, X3, Y3, Z3);
}

 alt_bn128_G2_host alt_bn128_G2_host::operator-() const
{
    return alt_bn128_G2_host(params, this->X, -(this->Y), this->Z);
}

 alt_bn128_G2_host alt_bn128_G2_host::operator-(const alt_bn128_G2_host &other) const
{
    return (*this) + (-other);
}



 alt_bn128_G2_host alt_bn128_G2_host::operator*(const unsigned long lhs) const
{
    return scalar_mul_host<alt_bn128_G2_host>(*this, lhs);
}

 alt_bn128_G2_host alt_bn128_G2_host::dbl() const
{
    if (this->is_zero())
        return *this;

    const alt_bn128_Fq2_host A = this->X.squared();    // A = X1^2
    const alt_bn128_Fq2_host B = this->Y.squared();    // B = Y1^2
    const alt_bn128_Fq2_host C = B.squared();          // C = B^2

    const alt_bn128_Fq2_host D = ((this->X + B).squared() - A - C).dbl();   // D = 2 * ((X1 + B)^2 - A - C)

    const alt_bn128_Fq2_host E = A + A.dbl();   // E = 3 * A
    const alt_bn128_Fq2_host F = E.squared();   // F = E^2

    const alt_bn128_Fq2_host X3 = F - D.dbl();                         // X3 = F - 2 D
    const alt_bn128_Fq2_host Y3 = E * (D - X3) - C.dbl().dbl().dbl();  // Y3 = E * (D - X3) - 8 * C
    const alt_bn128_Fq2_host Z3 = (this->Y * this->Z).dbl();           // Z3 = 2 * Y1 * Z1

    return alt_bn128_G2_host(params, X3, Y3, Z3);
}

 alt_bn128_G2_host alt_bn128_G2_host::add(const alt_bn128_G2_host &other) const
{
    return (*this) + other;
}

 alt_bn128_G2_host alt_bn128_G2_host::mixed_add(const alt_bn128_G2_host &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const alt_bn128_Fq2_host Z1Z1 = this->Z.squared();

    const alt_bn128_Fq2_host U2 = other.X * Z1Z1;

     const alt_bn128_Fq2_host S2 = this->Z * other.Y * Z1Z1;   // S2 = Y2 * Z1 * Z1Z1

    if (this->X == U2 && this->Y == S2)
        return this->dbl(); 

    const alt_bn128_Fq2_host H = U2 - this->X;               // H = U2-X1
    const alt_bn128_Fq2_host HH = H.squared();               // HH = H^2
    const alt_bn128_Fq2_host I = HH.dbl().dbl();             // I = 4*HH
    const alt_bn128_Fq2_host J = H * I;                      // J = H*I
    const alt_bn128_Fq2_host r = (S2 - this->Y).dbl();       // r = 2*(S2-Y1)
    const alt_bn128_Fq2_host V = this->X * I;                // V = X1*I
    
    const alt_bn128_Fq2_host X3 = r.squared() - J - V.dbl();           // X3 = r^2-J-2*V
    const alt_bn128_Fq2_host Y3 = r * (V - X3) - (this->Y * J).dbl();  // Y3 = r*(V-X3)-2*Y1*J
    const alt_bn128_Fq2_host Z3 = (this->Z + H).squared() - Z1Z1 - HH; // Z3 = (Z1+H)^2-Z1Z1-HH

    return alt_bn128_G2_host(params, X3, Y3, Z3);
}

 alt_bn128_G2_host alt_bn128_G2_host::mul_by_q() const
{
    return alt_bn128_G2_host(params,
                        *alt_bn128_twist_mul_by_q_X_host * (this->X).Frobenius_map(1),
                        *alt_bn128_twist_mul_by_q_Y_host * (this->Y).Frobenius_map(1),
                        (this->Z).Frobenius_map(1));
}

 bool alt_bn128_G2_host::is_well_formed() const
{
    if (this->is_zero())
        return true;

    alt_bn128_Fq2_host X2 = this->X.squared();
    alt_bn128_Fq2_host Y2 = this->Y.squared();
    alt_bn128_Fq2_host Z2 = this->Z.squared();

    alt_bn128_Fq2_host X3 = this->X * X2;
    alt_bn128_Fq2_host Z3 = this->Z * Z2;
    alt_bn128_Fq2_host Z6 = Z3.squared();

    return (Y2 == X3 + *alt_bn128_twist_coeff_b_host * Z6);
}

 alt_bn128_G2_host alt_bn128_G2_host::zero() const
{
    return alt_bn128_G2_host(params, *params->G2_zero_X, *params->G2_zero_Y, *params->G2_zero_Z);
}

 alt_bn128_G2_host alt_bn128_G2_host::one() const
{
    return alt_bn128_G2_host(params, *params->G2_one_X, *params->G2_one_Y, *params->G2_one_Z);
}

 alt_bn128_G2_host alt_bn128_G2_host::random_element() const
{
    scalar_field t(params->fr_params);
    return (t.random_element().as_bigint()) * this->one();
}

 size_t alt_bn128_G2_host::size_in_bits()
{
    base_field t(params->fq_params);
    return t.size_in_bits() + 1;
}

 bigint_host<alt_bn128_q_limbs_host> alt_bn128_G2_host::base_field_char()
{
    base_field t(params->fq_params);
    return t.field_char();
}

 bigint_host<alt_bn128_r_limbs_host> alt_bn128_G2_host::order()
{
    scalar_field t(params->fr_params);
    return t.field_char();
}

void alt_bn128_G2_host::set_params(alt_bn128_G2_params_host* params)
{
    this->params = params;
    this->X.set_params(params->fq2_params);
    this->Y.set_params(params->fq2_params);
    this->Z.set_params(params->fq2_params);
}



}