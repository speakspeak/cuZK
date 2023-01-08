#ifndef __KRONECKER_SUBSTITUTION_CU__
#define __KRONECKER_SUBSTITUTION_CU__

#include "../../depends/libstl-cuda/algorithm.cuh"
//#include <cmath>
#include "../../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cuh"
#include "../../depends/libff-cuda/common/utils.cuh"

namespace libfqfft {

template<typename FieldT>
__device__ void kronecker_substitution(libstl::vector<FieldT>& v3, const libstl::vector<FieldT>& v1, const libstl::vector<FieldT>& v2)
{
    FieldT instance = v1[0];

    /* Initialize */
    bool square = (v1 == v2) ? 1 : 0;

    /* Polynomial length */
    size_t n1 = v1.size();
    size_t n2 = v2.size();
    size_t n3 = n1 + n2 - 1;

    /* Determine number of bits needed */
    FieldT v1_max = *libstl::max_element(v1.begin(), v1.end());
    FieldT v2_max = *libstl::max_element(v2.begin(), v2.end());
    size_t b = 2 * (v1_max * v2_max).as_bigint().num_bits() + 1;

    /* Number of limbs needed in total */
    size_t k1 = libff::div_ceil (n1 * b, GMP_NUMB_BITS_);
    size_t k2 = libff::div_ceil (n2 * b, GMP_NUMB_BITS_);

    /* Output polynomial */
    v3.resize(n3, instance.zero());

    /*
     * Allocate all mp_limb_t_ space once and store the reference pointer M1
     * to free memory afterwards. P1, P2, and P3 will remain fixed pointers
     * to the start of their respective polynomials as reference.
     */
    mp_limb_t_* m1 = (mp_limb_t_*)malloc(sizeof(mp_limb_t_) * 2 * (k1 + k2));
    mp_limb_t_* p1 = m1;
    mp_limb_t_* p2 = p1 + k1;
    mp_limb_t_* p3 = p2 + k2;

    /* Helper variables */
    mp_limb_t_* ref;
    mp_limb_t_ limb;
    unsigned long val;
    unsigned long mask;
    unsigned long limb_b;
    unsigned long delta;
    unsigned long delta_b;

    /* Construct P1 limb */
    ref = p1;
    limb = 0;
    limb_b = 0;
    for (size_t i = 0; i < n1; i++)
    {
        val = v1[i].as_ulong();
        limb += (val << limb_b);

        /*
         * If the next iteration of LIMB_B is >= to the GMP_LIMB_BITS, then
         * write it out to mp_limb_t_* and reset LIMB. If VAL has remaining
         * bits due to GMP_LIMB_BITS boundary, set it in LIMB and proceed.
         */
        if (limb_b + b >= GMP_LIMB_BITS)
        {
            *ref++ = limb;
            limb = limb_b ? (val >> (GMP_LIMB_BITS - limb_b)) : 0;
            limb_b -= GMP_LIMB_BITS;
        }
        limb_b += b;
    }

    if (limb_b) 
        *ref++ = limb;

    /* Construct P2 limb. If V2 == V1, then P2 = P1 - square case. */
    if (square) 
        p2 = p1;
    else
    {
        ref = p2;
        limb = 0;
        limb_b = 0;
        for (size_t i = 0; i < n2; i++)
        {
            val = v2[i].as_ulong();
            limb += (val << limb_b);

            /*
             * If the next iteration of LIMB_B is >= to the GMP_LIMB_BITS, then
             * write it out to mp_limb_t_* and reset LIMB. If VAL has remaining
             * bits due to GMP_LIMB_BITS boundary, set it in LIMB and proceed.
             */
            if (limb_b + b >= GMP_LIMB_BITS)
            {
                *ref++ = limb;
                limb = limb_b ? (val >> (GMP_LIMB_BITS - limb_b)) : 0;
                limb_b -= GMP_LIMB_BITS;
            }
            limb_b += b;
        }
        if (limb_b) 
            *ref++ = limb;
    }

    /* Multiply P1 and P2 limbs and store result in P3 limb. */
    mpn_mul_(p3, p1, k1, p2, k2);

    /* Perfect alignment case: bits B is equivalent to GMP_LIMB_BITS */
    if (b == GMP_LIMB_BITS) 
        for (size_t i = 0; i < n3; i++) 
            v3[i] = FieldT(instance.params, *p3++); 
    /* Non-alignment case */
    else
    {
        /* Mask of 2^b - 1 */
        mask = (1UL << b) - 1;

        limb = 0;
        limb_b = 0;
        for (size_t i = 0; i < n3; i++)
        {
            /*
             * If the coefficient's bit length is contained in LIMB, then
             * write the masked value out to vector V3 and decrement LIMB
             * by B bits.
             */
            if (b <= limb_b)
            {
                v3[i] = FieldT(instance.params, limb & mask);

                delta = b;
                delta_b = limb_b - delta;
            }
            /*
             * If the remaining coefficient is across two LIMBs, then write
             * to vector V3 the current limb's value and add upper bits from
             * the second part. Lastly, decrement LIMB by the coefficient's
             * upper portion bit length.
             */
            else
            {
                v3[i] = FieldT(instance.params, limb);
                v3[i] += FieldT(instance.params, ((limb = *p3++) << limb_b) & mask);

                delta = b - limb_b;
                delta_b = GMP_LIMB_BITS - delta;
            }

            limb >>= delta;
            limb_b = delta_b;
        }
    }

    /* Free memory */
    free (m1);

    _condense(v3);
}

} // libfqfft

#endif
