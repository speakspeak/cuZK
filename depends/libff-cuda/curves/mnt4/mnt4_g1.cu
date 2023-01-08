#ifndef __MNT4_G1_CU__
#define __MNT4_G1_CU__

namespace libff {

__device__ mnt4_G1::mnt4_G1(mnt4_G1_params* params) : params(params), X(params->fq_params), Y(params->fq_params), Z(params->fq_params)
{
    this->X = *params->G1_zero_X;
    this->Y = *params->G1_zero_Y;
    this->Z = *params->G1_zero_Z;
}

__device__ void mnt4_G1::to_affine_coordinates()
{
    mnt4_Fq t(params->fq_params);

    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        mnt4_Fq Z_inv = Z.inverse();
        this->X = this->X * Z_inv;
        this->Y = this->Y * Z_inv;
        this->Z = t.one();
    }
}

__device__ void mnt4_G1::to_special()
{
    this->to_affine_coordinates();
}

__device__ bool mnt4_G1::is_special() const
{
    return this->is_zero() || this->Z == this->Z.one();
}

__device__ bool mnt4_G1::is_zero() const
{
    return (this->X.is_zero() && this->Z.is_zero());
}

__device__ bool mnt4_G1::operator==(const mnt4_G1 &other) const
{
    if (this->is_zero())
        return other.is_zero();

    if (other.is_zero())
        return false;

    if ((this->X * other.Z) != (other.X * this->Z))
        return false;

    return !((this->Y * other.Z) != (other.Y * this->Z));
}

__device__ bool mnt4_G1::operator!=(const mnt4_G1& other) const
{
    return !(operator==(other));
}

__device__ mnt4_G1 mnt4_G1::operator+(const mnt4_G1 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const mnt4_Fq X1Z2 = (this->X) * (other.Z);        // X1Z2 = X1 * Z2
    const mnt4_Fq X2Z1 = (this->Z) * (other.X);        // X2Z1 = X2 * Z1

    // (used both in add and double checks)

    const mnt4_Fq Y1Z2 = (this->Y) * (other.Z);        // Y1Z2 = Y1 * Z2
    const mnt4_Fq Y2Z1 = (this->Z) * (other.Y);        // Y2Z1 = Y2 * Z1

    if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
        return this->dbl(); 

    const mnt4_Fq Z1Z2 = (this->Z) * (other.Z);        // Z1Z2 = Z1 * Z2
    const mnt4_Fq u    = Y2Z1 - Y1Z2;                  // u    = Y2 * Z1 - Y1Z2
    const mnt4_Fq uu   = u.squared();                  // uu   = u^2
    const mnt4_Fq v    = X2Z1 - X1Z2;                  // v    = X2 * Z1 - X1Z2
    const mnt4_Fq vv   = v.squared();                  // vv   = v^2
    const mnt4_Fq vvv  = v * vv;                       // vvv  = v * vv
    const mnt4_Fq R    = vv * X1Z2;                    // R    = vv * X1Z2
    const mnt4_Fq A    = uu * Z1Z2 - (vvv + R.dbl());  // A    = uu * Z1Z2 - vvv - 2 * R

    const mnt4_Fq X3   = v * A;                        // X3   = v * A
    const mnt4_Fq Y3   = u * (R - A) - vvv * Y1Z2;     // Y3   = u * (R -  A) - vvv * Y1Z2
    const mnt4_Fq Z3   = vvv * Z1Z2;                   // Z3   = vvv * Z1Z2


    return mnt4_G1(params, X3, Y3, Z3);
} 

__device__ mnt4_G1 mnt4_G1::operator-() const
{
    return mnt4_G1(params, this->X, -(this->Y), this->Z);
}

__device__ mnt4_G1 mnt4_G1::operator-(const mnt4_G1 &other) const
{
    return (*this) + (-other);
}

__device__ mnt4_G1 mnt4_G1::operator*(const unsigned long lhs) const
{
    return scalar_mul<mnt4_G1>(*this, lhs);
}

__device__ mnt4_G1 mnt4_G1::dbl() const
{
    if (this->is_zero())
        return *this;

    const mnt4_Fq XX   = (this->X).squared();                     // XX  = X1^2
    const mnt4_Fq ZZ   = (this->Z).squared();                     // ZZ  = Z1^2
    const mnt4_Fq w    = *params->coeff_a * ZZ + (XX.dbl() + XX); // w   = a * ZZ + 3 * XX
    const mnt4_Fq Y1Z1 = (this->Y) * (this->Z);
    const mnt4_Fq s    = Y1Z1.dbl();                              // s   = 2 * Y1 * Z1
    const mnt4_Fq ss   = s.squared();                             // ss  = s^2
    const mnt4_Fq sss  = s * ss;                                  // sss = s * ss
    const mnt4_Fq R    = (this->Y) * s;                           // R   = Y1 *  s
    const mnt4_Fq RR   = R.squared();                             // RR  = R^2
    const mnt4_Fq B    = ((this->X) + R).squared()- XX - RR;      // B   = (X1 + R)^2 - XX - RR
    const mnt4_Fq h    = w.squared() - B.dbl();                   // h   = w^2 - 2 * B
    const mnt4_Fq X3   = h * s;                                   // X3  = h * s
    const mnt4_Fq Y3   = w * (B - h) - RR.dbl();                  // Y3  = w * (B - h) - 2 * RR
    const mnt4_Fq Z3   = sss;                                     // Z3  = sss

    return mnt4_G1(params, X3, Y3, Z3);
}

__device__ mnt4_G1 mnt4_G1::add(const mnt4_G1 &other) const
{
    return (*this) + other;
}

__device__ mnt4_G1 mnt4_G1::mixed_add(const mnt4_G1 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;
    
    const mnt4_Fq X2Z1 = (this->Z) * (other.X);             // X2Z1 = X2 * Z1
    const mnt4_Fq Y2Z1 = (this->Z) * (other.Y);             // Y2Z1 = Y 2* Z1

    if (this->X == X2Z1 && this->Y == Y2Z1)
        return this->dbl();

    const mnt4_Fq u = Y2Z1 - this->Y;                       // u = Y2 * Z1 - Y1
    const mnt4_Fq uu = u.squared();                         // uu = u2
    const mnt4_Fq v = X2Z1 - this->X;                       // v = X2 * Z1 - X1
    const mnt4_Fq vv = v.squared();                         // vv = v2
    const mnt4_Fq vvv = v * vv;                             // vvv = v * vv
    const mnt4_Fq R = vv * this->X;                         // R = vv * X1
    const mnt4_Fq A = uu * this->Z - vvv - R.dbl();         // A = uu * Z1 - vv v- 2 * R

    const mnt4_Fq X3 = v * A;                               // X3 = v * A
    const mnt4_Fq Y3 = u * (R - A) - vvv * this->Y;         // Y3 = u * (R - A) - vvv * Y1
    const mnt4_Fq Z3 = vvv * this->Z ;                      // Z3 = vvv * Z1

    return mnt4_G1(params, X3, Y3, Z3);
}

__device__ bool mnt4_G1::is_well_formed() const
{
    if (this->is_zero())
        return true;
    
    mnt4_Fq X2 = this->X.squared();
    mnt4_Fq Y2 = this->Y.squared();
    mnt4_Fq Z2 = this->Z.squared();

    return (this->Z * (Y2 - *params->coeff_b * Z2) == this->X * (X2 + *params->coeff_a * Z2));
}

__host__ mnt4_G1* mnt4_G1::zero_host()
{
    mnt4_G1* zero = libstl::create_host<mnt4_G1>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *zero = mnt4_G1(params, *params->G1_zero_X, *params->G1_zero_Y, *params->G1_zero_Z);
        }
    );
    cudaStreamSynchronize(0);

    return zero;
}

__device__ mnt4_G1 mnt4_G1::zero() const
{
    return mnt4_G1(params, *params->G1_zero_X, *params->G1_zero_Y, *params->G1_zero_Z);
}

__host__ mnt4_G1* mnt4_G1::one_host()
{
    mnt4_G1* one = libstl::create_host<mnt4_G1>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *one = mnt4_G1(params, *params->G1_one_X, *params->G1_one_Y, *params->G1_one_Z);
        }
    );
    cudaStreamSynchronize(0);
    return one;
}

__device__ mnt4_G1 mnt4_G1::one() const
{
    return mnt4_G1(params, *params->G1_one_X, *params->G1_one_Y, *params->G1_one_Z);
}

__device__ mnt4_G1 mnt4_G1::random_element() const
{
    scalar_field t(params->fr_params);
    return (t.random_element().as_bigint()) * this->one();
}

__device__ size_t mnt4_G1::size_in_bits()
{
    base_field t(params->fq_params);
    return t.size_in_bits() + 1;
}

__device__ bigint<mnt4_q_limbs> mnt4_G1::base_field_char()
{
    base_field t(params->fq_params);
    return t.field_char();
}

__device__ bigint<mnt4_r_limbs> mnt4_G1::order()
{
    scalar_field t(params->fr_params);
    return t.field_char();
}


__device__ void mnt4_G1::set_params(mnt4_G1_params* params)
{
    this->params = params;
    this->X.set_params(params->fq_params);
    this->Y.set_params(params->fq_params);
    this->Z.set_params(params->fq_params);
}


__device__ static void batch_to_special_invert(libstl::vector<mnt4_G1> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const mnt4_Fq& instance)
{
    libstl::vector<mnt4_Fq> prod;
    prod.resize(non_zero_length);

    mnt4_Fq acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    mnt4_Fq acc_inverse = acc.inverse();

    for (size_t i = non_zero_length-1; i <= non_zero_length; --i)
    {
        size_t idx = zero_idx[i];
        const mnt4_Fq old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}


__device__  void mnt4_G1::batch_to_special(libstl::vector<mnt4_G1> &vec)
{
    mnt4_Fq instance(params->fq_params);
    const mnt4_Fq one = instance.one();
    const mnt4_Fq zero = instance.zero();
    const mnt4_G1 g1_zero = this->zero();

    libstl::vector<size_t> zero_idx;
    zero_idx.resize(vec.size());
    size_t zero_length = vec.size() - 1;
    size_t non_zero_length = 0;
    for(size_t i=0; i<vec.size(); i++)
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

    batch_to_special_invert(vec, zero_idx, non_zero_length, instance);

    for (size_t i = 0; i < zero_idx.size(); ++i)
    {
        if(vec[i] == g1_zero)
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

static __device__ libstl::ParallalAllocator* _batch_to_special_pAllocator;

__device__ __inline__ static void p_batch_to_special_invert(libstl::vector<mnt4_G1> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const mnt4_Fq& instance)
{
    libstl::vector<mnt4_Fq> prod;
    prod.set_parallel_allocator(_batch_to_special_pAllocator);
    prod.resize(zero_idx.size());

    mnt4_Fq acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    mnt4_Fq acc_inverse = acc.inverse();

    for (size_t i = non_zero_length - 1; i <= non_zero_length; --i)
    { 
        size_t idx = zero_idx[i];
        const mnt4_Fq old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}

__device__ void mnt4_G1::p_batch_to_special(libstl::vector<mnt4_G1> &vec, size_t gridSize, size_t blockSize)
{
    size_t total = vec.size();
    size_t tnum = gridSize * blockSize;
    size_t alloc_size = (total + tnum - 1) / tnum * (sizeof(size_t) + sizeof(mnt4_Fq));

    size_t lockMem;
    libstl::lock(lockMem);

    _batch_to_special_pAllocator = libstl::allocate(gridSize, blockSize, 2124 + alloc_size);
    gmp_set_parallel_allocator_(_batch_to_special_pAllocator);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &vec]
        __device__ ()
        {
            mnt4_Fq instance(this->params->fq_params);
            const mnt4_Fq one = instance.one();
            const mnt4_Fq zero = instance.zero();
            const mnt4_G1 g1_zero = this->zero();

            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            size_t tnum = gridDim.x * blockDim.x;
            size_t total = vec.size();
            size_t range_s = (total + tnum - 1) / tnum * tid;
            size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
            libstl::vector<size_t> zero_idx;
            zero_idx.set_parallel_allocator(_batch_to_special_pAllocator);
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