#ifndef __BLS12_381_G1_CU__
#define __BLS12_381_G1_CU__

namespace libff {

__device__ bls12_381_G1::bls12_381_G1(bls12_381_G1_params* params) : params(params), X(params->fq_params), Y(params->fq_params), Z(params->fq_params)
{
    this->X = *params->G1_zero_X;
    this->Y = *params->G1_zero_Y;
    this->Z = *params->G1_zero_Z;
}

__device__ void bls12_381_G1::to_affine_coordinates()
{
    bls12_381_Fq t(params->fq_params);

    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        bls12_381_Fq Z_inv = Z.inverse();
        bls12_381_Fq Z2_inv = Z_inv.squared();
        bls12_381_Fq Z3_inv = Z2_inv * Z_inv;
        this->X = this->X * Z2_inv;
        this->Y = this->Y * Z3_inv;
        this->Z = t.one();
    }
}

__device__ void bls12_381_G1::to_special()
{
    this->to_affine_coordinates();
}

__device__ bool bls12_381_G1::is_special() const
{
    return this->is_zero() || this->Z == this->Z.one();
}

__device__ bool bls12_381_G1::is_zero() const
{
    return (this->Z.is_zero());
}

__device__ bool bls12_381_G1::operator==(const bls12_381_G1 &other) const
{
    if (this->is_zero())
        return other.is_zero();

    if (other.is_zero())
        return false;

    bls12_381_Fq Z1_squared = this->Z.squared();
    bls12_381_Fq Z2_squared = other.Z.squared();

    if ((this->X * Z2_squared) != (other.X * Z1_squared))
        return false;

    bls12_381_Fq Z1_cubed = this->Z * Z1_squared;
    bls12_381_Fq Z2_cubed = other.Z * Z2_squared;

    return !((this->Y * Z2_cubed) != (other.Y * Z1_cubed));
}

__device__ bool bls12_381_G1::operator!=(const bls12_381_G1& other) const
{
    return !(operator==(other));
}

__device__ bls12_381_G1 bls12_381_G1::operator+(const bls12_381_G1 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const bls12_381_Fq Z1Z1 = this->Z.squared();
    const bls12_381_Fq Z2Z2 = other.Z.squared();

    const bls12_381_Fq U1 = this->X * Z2Z2; 
    const bls12_381_Fq U2 = other.X * Z1Z1; 

    const bls12_381_Fq S1 = this->Y * other.Z * Z2Z2;  // S1 = Y1 * Z2 * Z2Z2
    const bls12_381_Fq S2 = this->Z * other.Y * Z1Z1;  // S2 = Y2 * Z1 * Z1Z1

    if (U1 == U2 && S1 == S2)
        return this->dbl(); 

    const bls12_381_Fq H = U2 - U1;             // H = U2 - U1
    const bls12_381_Fq I = H.dbl().squared();   // I = (2 * H)^2
    const bls12_381_Fq J = H * I;               // J = H * I
    const bls12_381_Fq r = (S2 - S1).dbl();     // r = 2 * (S2 - S1)
    const bls12_381_Fq V = U1 * I;              // V = U1 * I

    const bls12_381_Fq X3 = r.squared() - J - V.dbl();                           // X3 = r^2 - J - 2 * V
    const bls12_381_Fq Y3 = r * (V - X3)  - (S1 * J).dbl();                      // Y3 = r * (V-X3)-2 * S1 * J
    const bls12_381_Fq Z3 = ((this->Z + other.Z).squared() - Z1Z1 - Z2Z2) * H;   // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2) * H

    return bls12_381_G1(params, X3, Y3, Z3);
}

__device__ bls12_381_G1 bls12_381_G1::operator-() const
{
    return bls12_381_G1(params, this->X, -(this->Y), this->Z);
}

__device__ bls12_381_G1 bls12_381_G1::operator-(const bls12_381_G1 &other) const
{
    return (*this) + (-other);
}

__device__ bls12_381_G1 bls12_381_G1::operator*(const unsigned long lhs) const
{
    return scalar_mul<bls12_381_G1>(*this, lhs);
}

__device__ bls12_381_G1 bls12_381_G1::dbl() const
{
    if (this->is_zero())
        return *this;

    const bls12_381_Fq A = this->X.squared();    // A = X1^2
    const bls12_381_Fq B = this->Y.squared();    // B = Y1^2
    const bls12_381_Fq C = B.squared();          // C = B^2

    const bls12_381_Fq D = ((this->X + B).squared() - A - C).dbl();   // D = 2 * ((X1 + B)^2 - A - C)

    const bls12_381_Fq E = A + A.dbl();   // E = 3 * A
    const bls12_381_Fq F = E.squared();   // F = E^2

    const bls12_381_Fq X3 = F - D.dbl();                         // X3 = F - 2 D
    const bls12_381_Fq Y3 = E * (D - X3) - C.dbl().dbl().dbl();  // Y3 = E * (D - X3) - 8 * C
    const bls12_381_Fq Z3 = (this->Y * this->Z).dbl();           // Z3 = 2 * Y1 * Z1

    return bls12_381_G1(params, X3, Y3, Z3);
}

__device__ bls12_381_G1 bls12_381_G1::add(const bls12_381_G1 &other) const
{
    return (*this) + other;
}

__device__ bls12_381_G1 bls12_381_G1::mixed_add(const bls12_381_G1 &other) const
{
    if (this->is_zero())
        return other;

    if (other.is_zero())
        return *this;

    const bls12_381_Fq Z1Z1 = this->Z.squared();

    const bls12_381_Fq U2 = other.X * Z1Z1;

     const bls12_381_Fq S2 = this->Z * other.Y * Z1Z1;   // S2 = Y2 * Z1 * Z1Z1

    if (this->X == U2 && this->Y == S2)
        return this->dbl(); 

    const bls12_381_Fq H = U2 - this->X;               // H = U2-X1
    const bls12_381_Fq HH = H.squared();               // HH = H^2
    const bls12_381_Fq I = HH.dbl().dbl();             // I = 4*HH
    const bls12_381_Fq J = H * I;                      // J = H*I
    const bls12_381_Fq r = (S2 - this->Y).dbl();       // r = 2*(S2-Y1)
    const bls12_381_Fq V = this->X * I;                // V = X1*I
    
    const bls12_381_Fq X3 = r.squared() - J - V.dbl();           // X3 = r^2-J-2*V
    const bls12_381_Fq Y3 = r * (V - X3) - (this->Y * J).dbl();  // Y3 = r*(V-X3)-2*Y1*J
    const bls12_381_Fq Z3 = (this->Z + H).squared() - Z1Z1 - HH; // Z3 = (Z1+H)^2-Z1Z1-HH

    return bls12_381_G1(params, X3, Y3, Z3);
}

__device__ bool bls12_381_G1::is_well_formed() const
{
    if (this->is_zero())
        return true;
    
    bls12_381_Fq X2 = this->X.squared();
    bls12_381_Fq Y2 = this->Y.squared();
    bls12_381_Fq Z2 = this->Z.squared();

    bls12_381_Fq X3 = this->X * X2;
    bls12_381_Fq Z3 = this->Z * Z2;
    bls12_381_Fq Z6 = Z3.squared();

    return (Y2 == X3 + *bls12_381_coeff_b * Z6);
}

__device__ bls12_381_G1 bls12_381_G1::zero() const
{
    return bls12_381_G1(params, *params->G1_zero_X, *params->G1_zero_Y, *params->G1_zero_Z);
}

__host__ bls12_381_G1* bls12_381_G1::zero_host()
{
    bls12_381_G1* zero = libstl::create_host<bls12_381_G1>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *zero = bls12_381_G1(params, *params->G1_zero_X, *params->G1_zero_Y, *params->G1_zero_Z);
        }
    );
    cudaStreamSynchronize(0);

    return zero;
}

__device__ bls12_381_G1 bls12_381_G1::one() const
{
    return bls12_381_G1(params, *params->G1_one_X, *params->G1_one_Y, *params->G1_one_Z);
}

__host__ bls12_381_G1* bls12_381_G1::one_host()
{
    bls12_381_G1* one = libstl::create_host<bls12_381_G1>();
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ ()
        {
            *one = bls12_381_G1(params, *params->G1_one_X, *params->G1_one_Y, *params->G1_one_Z);
        }
    );
    cudaStreamSynchronize(0);
    return one;
}

__device__ bls12_381_G1 bls12_381_G1::random_element() const
{
    scalar_field t(params->fr_params);
    return (t.random_element().as_bigint()) * this->one();
}

__device__ size_t bls12_381_G1::size_in_bits()
{
    base_field t(params->fq_params);
    return t.size_in_bits() + 1;
}

__device__ bigint<bls12_381_q_limbs> bls12_381_G1::base_field_char()
{
    base_field t(params->fq_params);
    return t.field_char();
}

__device__ bigint<bls12_381_r_limbs> bls12_381_G1::order()
{
    scalar_field t(params->fr_params);
    return t.field_char();
}

__device__ void bls12_381_G1::set_params(bls12_381_G1_params* params)
{
    this->params = params;
    this->X.set_params(params->fq_params);
    this->Y.set_params(params->fq_params);
    this->Z.set_params(params->fq_params);
}


__device__ static void batch_to_special_invert(libstl::vector<bls12_381_G1> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const bls12_381_Fq& instance)
{
    libstl::vector<bls12_381_Fq> prod;
    prod.resize(non_zero_length);

    bls12_381_Fq acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    bls12_381_Fq acc_inverse = acc.inverse();

    for (size_t i = non_zero_length-1; i <= non_zero_length; --i)
    {
        size_t idx = zero_idx[i];
        const bls12_381_Fq old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}


__device__  void bls12_381_G1::batch_to_special(libstl::vector<bls12_381_G1> &vec)
{
    bls12_381_Fq instance(params->fq_params);
    const bls12_381_Fq one = instance.one();
    const bls12_381_Fq zero = instance.zero();
    const bls12_381_G1 g1_zero = this->zero();

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
            bls12_381_Fq Z2 = vec[i].Z.squared();
            bls12_381_Fq Z3 = vec[i].Z * Z2;

            vec[i].X = vec[i].X * Z2;
            vec[i].Y = vec[i].Y * Z3;
            vec[i].Z = one;
        }
    }
}

static __device__ libstl::ParallalAllocator* _batch_to_special_pAllocator;

__device__ __inline__ static void p_batch_to_special_invert(libstl::vector<bls12_381_G1> &vec, libstl::vector<size_t> &zero_idx, size_t non_zero_length, const bls12_381_Fq& instance)
{
    libstl::vector<bls12_381_Fq> prod;
    prod.set_parallel_allocator(_batch_to_special_pAllocator);
    prod.resize(zero_idx.size());

    bls12_381_Fq acc = instance.one();

    for (size_t i=0; i<non_zero_length; i++)
    {
        size_t idx = zero_idx[i];
        prod[i] = acc;
        acc = acc * vec[idx].Z;
    }

    bls12_381_Fq acc_inverse = acc.inverse();

    for (size_t i = non_zero_length - 1; i <= non_zero_length; --i)
    { 
        size_t idx = zero_idx[i];
        const bls12_381_Fq old_el = vec[idx].Z;
        vec[idx].Z = acc_inverse * prod[i];
        acc_inverse = acc_inverse * old_el;
    }
}

__device__ void bls12_381_G1::p_batch_to_special(libstl::vector<bls12_381_G1> &vec, size_t gridSize, size_t blockSize)
{
    size_t total = vec.size();
    size_t tnum = gridSize * blockSize;
    size_t alloc_size = (total + tnum - 1) / tnum * (sizeof(size_t) + sizeof(bls12_381_Fq));

    size_t lockMem;
    libstl::lock(lockMem);

    _batch_to_special_pAllocator = libstl::allocate(gridSize, blockSize, 2124 + alloc_size);
    gmp_set_parallel_allocator_(_batch_to_special_pAllocator);

    libstl::launch<<<gridSize, blockSize>>>
    (
        [=, &vec]
        __device__ ()
        {
            bls12_381_Fq instance(this->params->fq_params);
            const bls12_381_Fq one = instance.one();
            const bls12_381_Fq zero = instance.zero();
            const bls12_381_G1 g1_zero = this->zero();

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

            // for(size_t i=range_s; i < range_e && i < total; i++)
            // {
            //     vec[i].Y = vec[i].Z;
            // }
                
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
                    bls12_381_Fq Z2 = vec[i].Z.squared();
                    bls12_381_Fq Z3 = vec[i].Z * Z2;

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
