#ifndef __MNT4_G2_CU__
#define __MNT4_G2_CU__

namespace libff {

__device__ mnt4_G2::mnt4_G2(mnt4_G2_params* params) : params(params), X(params->fq2_params), Y(params->fq2_params), Z(params->fq2_params)
{
    this->X = *params->G2_zero_X;
    this->Y = *params->G2_zero_Y;
    this->Z = *params->G2_zero_Z;
}

__device__ mnt4_Fq2 mnt4_G2::mul_by_a(const mnt4_Fq2 &elt) const
{
    return mnt4_Fq2(params->fq2_params, *mnt4_twist_mul_by_a_c0 * elt.c0, *mnt4_twist_mul_by_a_c1 * elt.c1);
}

__device__ mnt4_Fq2 mnt4_G2::mul_by_b(const mnt4_Fq2 &elt) const
{
    return mnt4_Fq2(params->fq2_params, *mnt4_twist_mul_by_b_c0 * elt.c1, *mnt4_twist_mul_by_b_c1 * elt.c0);
}

__device__ void mnt4_G2::to_affine_coordinates()
{    
    mnt4_Fq2 t(params->fq2_params);

    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        mnt4_Fq2 Z_inv = Z.inverse();
        this->X = this->X * Z_inv;
        this->Y = this->Y * Z_inv;
        this->Z = t.one();
    }
}

__device__ void mnt4_G2::to_special()
{
    this->to_affine_coordinates();
}

__device__ bool mnt4_G2::is_special() const
{
    return (this->is_zero() || this->Z == this->Z.one());
}

__device__ bool mnt4_G2::is_zero() const
{
    return (this->X.is_zero() && this->Z.is_zero());
}

__device__ bool mnt4_G2::operator==(const mnt4_G2 &other) const
{
    if (this->is_zero())
        return other.is_zero();

    if (other.is_zero())
        return false;

    if ((this->X * other.Z) != (other.X * this->Z))
        return false;

    return !((this->Y * other.Z) != (other.Y * this->Z));
}

__device__ bool mnt4_G2::operator!=(const mnt4_G2& other) const
{
    return !(operator==(other));
}

__device__ mnt4_G2 mnt4_G2::operator+(const mnt4_G2 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const mnt4_Fq2 X1Z2 = (this->X) * (other.Z);        // X1Z2 = X1 * Z2
    const mnt4_Fq2 X2Z1 = (this->Z) * (other.X);        // X2Z1 = X2 * Z1

    // (used both in add and double checks)

    const mnt4_Fq2 Y1Z2 = (this->Y) * (other.Z);        // Y1Z2 = Y1 * Z2
    const mnt4_Fq2 Y2Z1 = (this->Z) * (other.Y);        // Y2Z1 = Y2 * Z1

    if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
        return this->dbl(); 

    const mnt4_Fq2 Z1Z2 = (this->Z) * (other.Z);        // Z1Z2 = Z1 * Z2
    const mnt4_Fq2 u    = Y2Z1 - Y1Z2;                  // u    = Y2 * Z1 - Y1Z2
    const mnt4_Fq2 uu   = u.squared();                  // uu   = u^2
    const mnt4_Fq2 v    = X2Z1 - X1Z2;                  // v    = X2 * Z1 - X1Z2
    const mnt4_Fq2 vv   = v.squared();                  // vv   = v^2
    const mnt4_Fq2 vvv  = v * vv;                       // vvv  = v * vv
    const mnt4_Fq2 R    = vv * X1Z2;                    // R    = vv * X1Z2
    const mnt4_Fq2 A    = uu * Z1Z2 - (vvv + R.dbl());  // A    = uu * Z1Z2 - vvv - 2 * R

    const mnt4_Fq2 X3   = v * A;                        // X3   = v * A
    const mnt4_Fq2 Y3   = u * (R - A) - vvv * Y1Z2;     // Y3   = u * (R -  A) - vvv * Y1Z2
    const mnt4_Fq2 Z3   = vvv * Z1Z2;                   // Z3   = vvv * Z1Z2


    return mnt4_G2(params, X3, Y3, Z3);
}

__device__ mnt4_G2 mnt4_G2::operator-() const
{
    return mnt4_G2(params, this->X, -(this->Y), this->Z);
}

__device__ mnt4_G2 mnt4_G2::operator-(const mnt4_G2 &other) const
{
    return (*this) + (-other);
}

__device__ mnt4_G2 mnt4_G2::operator*(const unsigned long lhs) const
{
    return scalar_mul<mnt4_G2>(*this, lhs);
}

__device__ mnt4_G2 mnt4_G2::dbl() const
{
    if (this->is_zero())
        return *this;

    const mnt4_Fq2 XX   = (this->X).squared();                     // XX  = X1^2
    const mnt4_Fq2 ZZ   = (this->Z).squared();                     // ZZ  = Z1^2mnt4_Fq2
    const mnt4_Fq2 w    = mul_by_a(ZZ) + (XX.dbl() + XX);          // w   = a * ZZ + 3 * XX
    const mnt4_Fq2 Y1Z1 = (this->Y) * (this->Z);
    const mnt4_Fq2 s    = Y1Z1.dbl();                              // s   = 2 * Y1 * Z1
    const mnt4_Fq2 ss   = s.squared();                             // ss  = s^2
    const mnt4_Fq2 sss  = s * ss;                                  // sss = s * ss
    const mnt4_Fq2 R    = (this->Y) * s;                           // R   = Y1 *  s
    const mnt4_Fq2 RR   = R.squared();                             // RR  = R^2
    const mnt4_Fq2 B    = ((this->X) + R).squared()- XX - RR;      // B   = (X1 + R)^2 - XX - RR
    const mnt4_Fq2 h    = w.squared() - B.dbl();                   // h   = w^2 - 2 * B
    const mnt4_Fq2 X3   = h * s;                                   // X3  = h * s
    const mnt4_Fq2 Y3   = w * (B - h) - RR.dbl();                  // Y3  = w * (B - h) - 2 * RR
    const mnt4_Fq2 Z3   = sss;                                     // Z3  = sss

    return mnt4_G2(params, X3, Y3, Z3);
}

__device__ mnt4_G2 mnt4_G2::add(const mnt4_G2 &other) const
{
    return (*this) + other;
}

__device__ mnt4_G2 mnt4_G2::mixed_add(const mnt4_G2 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;
    
    const mnt4_Fq2 X2Z1 = (this->Z) * (other.X);             // X2Z1 = X2 * Z1
    const mnt4_Fq2 Y2Z1 = (this->Z) * (other.Y);             // Y2Z1 = Y 2* Z1

    if (this->X == X2Z1 && this->Y == Y2Z1)
        return this->dbl();

    const mnt4_Fq2 u = Y2Z1 - this->Y;                       // u = Y2 * Z1 - Y1
    const mnt4_Fq2 uu = u.squared();                         // uu = u2
    const mnt4_Fq2 v = X2Z1 - this->X;                       // v = X2 * Z1 - X1
    const mnt4_Fq2 vv = v.squared();                         // vv = v2
    const mnt4_Fq2 vvv = v * vv;                             // vvv = v * vv
    const mnt4_Fq2 R = vv * this->X;                         // R = vv * X1
    const mnt4_Fq2 A = uu * this->Z - vvv - R.dbl();         // A = uu * Z1 - vv v- 2 * R

    const mnt4_Fq2 X3 = v * A;                               // X3 = v * A
    const mnt4_Fq2 Y3 = u * (R - A) - vvv * this->Y;         // Y3 = u * (R - A) - vvv * Y1
    const mnt4_Fq2 Z3 = vvv * this->Z ;                      // Z3 = vvv * Z1

    return mnt4_G2(params, X3, Y3, Z3);
}

__device__ mnt4_G2 mnt4_G2::mul_by_q() const
{
    return mnt4_G2(params,
                        *mnt4_twist_mul_by_q_X * (this->X).Frobenius_map(1),
                        *mnt4_twist_mul_by_q_Y * (this->Y).Frobenius_map(1),
                        (this->Z).Frobenius_map(1));
}

__device__ bool mnt4_G2::is_well_formed() const
{
    if (this->is_zero())
        return true;

    mnt4_Fq2 X2 = this->X.squared();
    mnt4_Fq2 Y2 = this->Y.squared();
    mnt4_Fq2 Z2 = this->Z.squared();
    mnt4_Fq2 aZ2 = *mnt4_twist_coeff_a * Z2;

    return (this->Z * (Y2 - *mnt4_twist_coeff_b * Z2) == this->X * (X2 + aZ2));
}

__device__ mnt4_G2 mnt4_G2::zero() const
{
    return mnt4_G2(params, *params->G2_zero_X, *params->G2_zero_Y, *params->G2_zero_Z);
}

__host__ mnt4_G2* mnt4_G2::zero_host()
{
    mnt4_G2* zero = libstl::create_host<mnt4_G2>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *zero = mnt4_G2(params, *params->G2_zero_X, *params->G2_zero_Y, *params->G2_zero_Z);
        }
    );
    cudaStreamSynchronize(0);
    return zero;
}

__device__ mnt4_G2 mnt4_G2::one() const
{
    return mnt4_G2(params, *params->G2_one_X, *params->G2_one_Y, *params->G2_one_Z);
}

__host__ mnt4_G2* mnt4_G2::one_host()
{
    mnt4_G2* one = libstl::create_host<mnt4_G2>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *one = mnt4_G2(params, *params->G2_one_X, *params->G2_one_Y, *params->G2_one_Z);
        }
    );
    cudaStreamSynchronize(0);
    return one;
}

__device__ mnt4_G2 mnt4_G2::random_element() const
{
    scalar_field t(params->fr_params);
    return (t.random_element().as_bigint()) * this->one();
}

__device__ size_t mnt4_G2::size_in_bits()
{
    base_field t(params->fq_params);
    return t.size_in_bits() + 1;
}

__device__ bigint<mnt4_q_limbs> mnt4_G2::base_field_char()
{
    base_field t(params->fq_params);
    return t.field_char();
}

__device__ bigint<mnt4_r_limbs> mnt4_G2::order()
{
    scalar_field t(params->fr_params);
    return t.field_char();
}

__device__ void mnt4_G2::set_params(mnt4_G2_params* params)
{
    this->params = params;
    this->X.set_params(params->fq2_params);
    this->Y.set_params(params->fq2_params);
    this->Z.set_params(params->fq2_params);
}


__device__ static void batch_to_special_invert(libstl::vector<mnt4_G2> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const mnt4_Fq2& instance)
{
    libstl::vector<mnt4_Fq2> prod;
    prod.resize(non_zero_length);

    mnt4_Fq2 acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    mnt4_Fq2 acc_inverse = acc.inverse();

    for (size_t i = non_zero_length-1; i <= non_zero_length; --i)
    {
        size_t idx = zero_idx[i];
        const mnt4_Fq2 old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}

__device__ void mnt4_G2::batch_to_special(libstl::vector<mnt4_G2> &vec)
{
    mnt4_Fq2 instance(params->fq2_params);
    const mnt4_Fq2 one = instance.one();
    const mnt4_Fq2 zero = instance.zero();
    const mnt4_G2 g2_zero = this->zero();

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
            vec[i].X = vec[i].X * vec[i].Z;
            vec[i].Y = vec[i].Y * vec[i].Z;
            vec[i].Z = one;
        }
    }
}


static __device__ libstl::ParallalAllocator* _g2_batch_to_special_pAllocator;

__device__ __inline__ static void p_batch_to_special_invert(libstl::vector<mnt4_G2> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const mnt4_Fq2& instance)
{
    libstl::vector<mnt4_Fq2> prod;
    prod.set_parallel_allocator(_g2_batch_to_special_pAllocator);
    prod.resize(zero_idx.size());

    mnt4_Fq2 acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    mnt4_Fq2 acc_inverse = acc.inverse();

    for (size_t i = non_zero_length - 1; i <= non_zero_length; --i)
    { 
        size_t idx = zero_idx[i];
        const mnt4_Fq2 old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}



__device__ void mnt4_G2::p_batch_to_special(libstl::vector<mnt4_G2> &vec, size_t gridSize, size_t blockSize)
{
    size_t total = vec.size();
    size_t tnum = gridSize * blockSize;
    size_t alloc_size = (total + tnum - 1) / tnum * (sizeof(size_t) + sizeof(mnt4_Fq2));
    
    size_t lockMem;
    libstl::lock(lockMem);

    _g2_batch_to_special_pAllocator = libstl::allocate(gridSize, blockSize, 4124 + alloc_size);
    gmp_set_parallel_allocator_(_g2_batch_to_special_pAllocator);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &vec]
        __device__ ()
        {
            mnt4_Fq2 instance(this->params->fq2_params);
            const mnt4_Fq2 one = instance.one();
            const mnt4_Fq2 zero = instance.zero();
            const mnt4_G2 g1_zero = this->zero();

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
                    vec[i].X = vec[i].X * vec[i].Z;
                    vec[i].Y = vec[i].Y * vec[i].Z;
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