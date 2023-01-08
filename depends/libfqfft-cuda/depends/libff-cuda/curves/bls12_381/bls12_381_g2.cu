#ifndef __BLS12_381_G2_CU__
#define __BLS12_381_G2_CU__

namespace libff {

__device__ bls12_381_G2::bls12_381_G2(bls12_381_G2_params* params) : params(params), X(params->fq2_params), Y(params->fq2_params), Z(params->fq2_params)
{
    this->X = *params->G2_zero_X;
    this->Y = *params->G2_zero_Y;
    this->Z = *params->G2_zero_Z;
}

__device__ bls12_381_Fq2 bls12_381_G2::mul_by_b(const bls12_381_Fq2 &elt)
{
    return bls12_381_Fq2(params->fq2_params, *bls12_381_twist_mul_by_b_c0 * elt.c0, *bls12_381_twist_mul_by_b_c1 * elt.c1);
}

__device__ void bls12_381_G2::to_affine_coordinates()
{    
    bls12_381_Fq2 t(params->fq2_params);

    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        bls12_381_Fq2 Z_inv = Z.inverse();
        bls12_381_Fq2 Z2_inv = Z_inv.squared();
        bls12_381_Fq2 Z3_inv = Z2_inv * Z_inv;
        this->X = this->X * Z2_inv;
        this->Y = this->Y * Z3_inv;
        this->Z = t.one();
    }
}

__device__ void bls12_381_G2::to_special()
{
    this->to_affine_coordinates();
}

__device__ bool bls12_381_G2::is_special() const
{
    return (this->is_zero() || this->Z == this->Z.one());
}

__device__ bool bls12_381_G2::is_zero() const
{
    return (this->Z.is_zero());
}

__device__ bool bls12_381_G2::operator==(const bls12_381_G2 &other) const
{
    if (this->is_zero())
        return other.is_zero();

    if (other.is_zero())
        return false;

    bls12_381_Fq2 Z1_squared = this->Z.squared();
    bls12_381_Fq2 Z2_squared = other.Z.squared();

    if ((this->X * Z2_squared) != (other.X * Z1_squared))
        return false;

    bls12_381_Fq2 Z1_cubed = this->Z * Z1_squared;
    bls12_381_Fq2 Z2_cubed = other.Z * Z2_squared;

    return !((this->Y * Z2_cubed) != (other.Y * Z1_cubed));
}

__device__ bool bls12_381_G2::operator!=(const bls12_381_G2& other) const
{
    return !(operator==(other));
}

__device__ bls12_381_G2 bls12_381_G2::operator+(const bls12_381_G2 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    bls12_381_Fq2 Z1Z1 = this->Z.squared();
    bls12_381_Fq2 Z2Z2 = other.Z.squared();

    bls12_381_Fq2 U1 = this->X * Z2Z2;
    bls12_381_Fq2 U2 = other.X * Z1Z1;

    bls12_381_Fq2 Z1_cubed = this->Z * Z1Z1;
    bls12_381_Fq2 Z2_cubed = other.Z * Z2Z2;

    bls12_381_Fq2 S1 = this->Y * Z2_cubed;      // S1 = Y1 * Z2 * Z2Z2
    bls12_381_Fq2 S2 = other.Y * Z1_cubed;      // S2 = Y2 * Z1 * Z1Z1

    if (U1 == U2 && S1 == S2)
        return this->dbl();    // dbl case; nothing of above can be reused

    // rest of add case
    bls12_381_Fq2 H = U2 - U1;                            // H = U2-U1
    bls12_381_Fq2 I = H.dbl().squared();                  // I = (2 * H)^2
    bls12_381_Fq2 J = H * I;                              // J = H * I
    bls12_381_Fq2 r = (S2 - S1).dbl();                    // r = 2 * (S2 - S1)
    bls12_381_Fq2 V = U1 * I;                             // V = U1 * I
    
    bls12_381_Fq2 X3 = r.squared() - J - V.dbl();                           // X3 = r^2 - J - 2 * V
    bls12_381_Fq2 Y3 = r * (V - X3) - (S1 * J).dbl();                       // Y3 = r * (V-X3)-2 S1 J
    bls12_381_Fq2 Z3 = ((this->Z + other.Z).squared() - Z1Z1 - Z2Z2) * H;   // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2) * H

    return bls12_381_G2(params, X3, Y3, Z3);
}

__device__ bls12_381_G2 bls12_381_G2::operator-() const
{
    return bls12_381_G2(params, this->X, -(this->Y), this->Z);
}

__device__ bls12_381_G2 bls12_381_G2::operator-(const bls12_381_G2 &other) const
{
    return (*this) + (-other);
}

__device__ bls12_381_G2 bls12_381_G2::operator*(const unsigned long lhs) const
{
    return scalar_mul<bls12_381_G2>(*this, lhs);
}

__device__ bls12_381_G2 bls12_381_G2::dbl() const
{
    if (this->is_zero())
        return *this;

    const bls12_381_Fq2 A = this->X.squared();    // A = X1^2
    const bls12_381_Fq2 B = this->Y.squared();    // B = Y1^2
    const bls12_381_Fq2 C = B.squared();          // C = B^2

    const bls12_381_Fq2 D = ((this->X + B).squared() - A - C).dbl();   // D = 2 * ((X1 + B)^2 - A - C)

    const bls12_381_Fq2 E = A + A.dbl();   // E = 3 * A
    const bls12_381_Fq2 F = E.squared();   // F = E^2

    const bls12_381_Fq2 X3 = F - D.dbl();                         // X3 = F - 2 D
    const bls12_381_Fq2 Y3 = E * (D - X3) - C.dbl().dbl().dbl();  // Y3 = E * (D - X3) - 8 * C
    const bls12_381_Fq2 Z3 = (this->Y * this->Z).dbl();           // Z3 = 2 * Y1 * Z1

    return bls12_381_G2(params, X3, Y3, Z3);
}

__device__ bls12_381_G2 bls12_381_G2::add(const bls12_381_G2 &other) const
{
    return (*this) + other;
}

__device__ bls12_381_G2 bls12_381_G2::mixed_add(const bls12_381_G2 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const bls12_381_Fq2 Z1Z1 = this->Z.squared();

    const bls12_381_Fq2 U2 = other.X * Z1Z1;

     const bls12_381_Fq2 S2 = this->Z * other.Y * Z1Z1;   // S2 = Y2 * Z1 * Z1Z1

    if (this->X == U2 && this->Y == S2)
        return this->dbl(); 

    const bls12_381_Fq2 H = U2 - this->X;               // H = U2-X1
    const bls12_381_Fq2 HH = H.squared();               // HH = H^2
    const bls12_381_Fq2 I = HH.dbl().dbl();             // I = 4*HH
    bls12_381_Fq2 J = H * I;                      // J = H*I
    const bls12_381_Fq2 r = (S2 - this->Y).dbl();       // r = 2*(S2-Y1)
    const bls12_381_Fq2 V = this->X * I;                // V = X1*I
    
    const bls12_381_Fq2 X3 = r.squared() - J - V.dbl();           // X3 = r^2-J-2*V

    J = (this->Y * J).dbl();
    const bls12_381_Fq2 Y3 = r * (V - X3) - J;  // Y3 = r*(V-X3)-2*Y1*J

    bls12_381_Fq2 Z3 = this->Z + H;
    Z3 = Z3.squared() - Z1Z1 - HH; // Z3 = (Z1+H)^2-Z1Z1-HH

    return bls12_381_G2(params, X3, Y3, Z3);
}

__device__ bls12_381_G2 bls12_381_G2::mul_by_q() const
{
    return bls12_381_G2(params,
                        *bls12_381_twist_mul_by_q_X * (this->X).Frobenius_map(1),
                        *bls12_381_twist_mul_by_q_Y * (this->Y).Frobenius_map(1),
                        (this->Z).Frobenius_map(1));
}

__device__ bool bls12_381_G2::is_well_formed() const
{
    if (this->is_zero())
        return true;

    bls12_381_Fq2 X2 = this->X.squared();
    bls12_381_Fq2 Y2 = this->Y.squared();
    bls12_381_Fq2 Z2 = this->Z.squared();

    bls12_381_Fq2 X3 = this->X * X2;
    bls12_381_Fq2 Z3 = this->Z * Z2;
    bls12_381_Fq2 Z6 = Z3.squared();

    return (Y2 == X3 + *bls12_381_twist_coeff_b * Z6);
}

__device__ bls12_381_G2 bls12_381_G2::zero() const
{
    return bls12_381_G2(params, *params->G2_zero_X, *params->G2_zero_Y, *params->G2_zero_Z);
}

__host__ bls12_381_G2* bls12_381_G2::zero_host()
{
    bls12_381_G2* zero = libstl::create_host<bls12_381_G2>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *zero = bls12_381_G2(params, *params->G2_zero_X, *params->G2_zero_Y, *params->G2_zero_Z);
        }
    );
    cudaStreamSynchronize(0);
    return zero;
}

__device__ bls12_381_G2 bls12_381_G2::one() const
{
    return bls12_381_G2(params, *params->G2_one_X, *params->G2_one_Y, *params->G2_one_Z);
}

__host__ bls12_381_G2* bls12_381_G2::one_host()
{
    bls12_381_G2* one = libstl::create_host<bls12_381_G2>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *one = bls12_381_G2(params, *params->G2_one_X, *params->G2_one_Y, *params->G2_one_Z);
        }
    );
    cudaStreamSynchronize(0);
    return one;
}

__device__ bls12_381_G2 bls12_381_G2::random_element() const
{
    scalar_field t(params->fr_params);
    return (t.random_element().as_bigint()) * this->one();
}

__device__ size_t bls12_381_G2::size_in_bits()
{
    base_field t(params->fq_params);
    return t.size_in_bits() + 1;
}

__device__ bigint<bls12_381_q_limbs> bls12_381_G2::base_field_char()
{
    base_field t(params->fq_params);
    return t.field_char();
}

__device__ bigint<bls12_381_r_limbs> bls12_381_G2::order()
{
    scalar_field t(params->fr_params);
    return t.field_char();
}

__device__ void bls12_381_G2::set_params(bls12_381_G2_params* params)
{
    this->params = params;
    this->X.set_params(params->fq2_params);
    this->Y.set_params(params->fq2_params);
    this->Z.set_params(params->fq2_params);
}


__device__ static void batch_to_special_invert(libstl::vector<bls12_381_G2> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const bls12_381_Fq2& instance)
{
    libstl::vector<bls12_381_Fq2> prod;
    prod.resize(non_zero_length);

    bls12_381_Fq2 acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    bls12_381_Fq2 acc_inverse = acc.inverse();

    for (size_t i = non_zero_length-1; i <= non_zero_length; --i)
    {
        size_t idx = zero_idx[i];
        const bls12_381_Fq2 old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}

__device__ void bls12_381_G2::batch_to_special(libstl::vector<bls12_381_G2> &vec)
{
    bls12_381_Fq2 instance(params->fq2_params);
    const bls12_381_Fq2 one = instance.one();
    const bls12_381_Fq2 zero = instance.zero();
    const bls12_381_G2 g2_zero = this->zero();

    libstl::vector<size_t> zero_idx;
    zero_idx.resize(vec.size());
    size_t zero_length = vec.size() - 1;
    size_t non_zero_length = 0;
    for(size_t i=0; i<vec.size(); i++)
    {
        if(vec[i] == g2_zero)
        {
            zero_idx[zero_length--] = i;
        }
        else
        {
            zero_idx[non_zero_length++] = i;
        }
    }

    batch_to_special_invert(vec, zero_idx, non_zero_length, instance);

    for (size_t i = 0; i < zero_idx.size(); ++i)
    {
        if(vec[i] == g2_zero)
        {
            vec[i].X = zero;
            vec[i].Y = one;
            vec[i].Z = zero;
        }
        else{
            bls12_381_Fq2 Z2 = vec[i].Z.squared();
            bls12_381_Fq2 Z3 = vec[i].Z * Z2;

            vec[i].X = vec[i].X * Z2;
            vec[i].Y = vec[i].Y * Z3;
            vec[i].Z = one;
        }
    }
}


static __device__ libstl::ParallalAllocator* _g2_batch_to_special_pAllocator;

__device__ __inline__ static void p_batch_to_special_invert(libstl::vector<bls12_381_G2> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const bls12_381_Fq2& instance)
{
    libstl::vector<bls12_381_Fq2> prod;
    prod.set_parallel_allocator(_g2_batch_to_special_pAllocator);
    prod.resize(zero_idx.size());

    bls12_381_Fq2 acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    bls12_381_Fq2 acc_inverse = acc.inverse();

    for (size_t i = non_zero_length - 1; i <= non_zero_length; --i)
    { 
        size_t idx = zero_idx[i];
        const bls12_381_Fq2 old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}



__device__ void bls12_381_G2::p_batch_to_special(libstl::vector<bls12_381_G2> &vec, size_t gridSize, size_t blockSize)
{
    size_t total = vec.size();
    size_t tnum = gridSize * blockSize;
    size_t alloc_size = (total + tnum - 1) / tnum * (sizeof(size_t) + sizeof(bls12_381_Fq2));

    size_t lockMem;
    libstl::lock(lockMem);

    _g2_batch_to_special_pAllocator = libstl::allocate(gridSize, blockSize, 4124 + alloc_size);
    gmp_set_parallel_allocator_(_g2_batch_to_special_pAllocator);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &vec]
        __device__ ()
        {
            bls12_381_Fq2 instance(this->params->fq2_params);
            const bls12_381_Fq2 one = instance.one();
            const bls12_381_Fq2 zero = instance.zero();
            const bls12_381_G2 g1_zero = this->zero();

            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = gridDim.x * blockDim.x;
            size_t total = vec.size();
            size_t range_s = (total + tnum - 1) / tnum * tid;
            size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
            libstl::vector<size_t> zero_idx;
            zero_idx.set_parallel_allocator(_g2_batch_to_special_pAllocator);
            zero_idx.resize(range_e - range_s);
            size_t zero_length = range_e - range_s - 1;
            size_t non_zero_length = 0;

            for(size_t i=range_s; i < range_e && i < total; i++)
            {
                if(vec[i] == g1_zero)
                {
                    zero_idx[zero_length--] = i;
                }
                else
                {
                    zero_idx[non_zero_length++] = i;
                }
            }

            p_batch_to_special_invert(vec, zero_idx, non_zero_length, instance);

            for (size_t i = range_s; i < range_e && i < total; ++i)
            {
                if(vec[i] == g1_zero)
                {
                    vec[i].X = zero;
                    vec[i].Y = one;
                    vec[i].Z = zero;
                }
                else
                {
                    bls12_381_Fq2 Z2 = vec[i].Z.squared();
                    bls12_381_Fq2 Z3 = vec[i].Z * Z2;

                    vec[i].X = vec[i].X * Z2;
                    vec[i].Y = vec[i].Y * Z3;
                    vec[i].Z = one;
                }
            }
        }
    );
    cudaDeviceSynchronize();
    gmp_set_serial_allocator_();
    libstl::resetlock(lockMem);
}

}

#endif
