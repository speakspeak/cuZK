#ifndef __MINI_MP_CUDA_CUH__
#define __MINI_MP_CUDA_CUH__

#include "../depends/libstl-cuda/memory.cuh"

typedef unsigned long mp_limb_t_;
typedef long mp_size_t_;
typedef unsigned long mp_bitcnt_t_;

typedef mp_limb_t_ *mp_ptr_;
typedef const mp_limb_t_ *mp_srcptr_;

typedef struct
{
    int _mp_alloc;		/* Number of *limbs* allocated and pointed
            to by the _mp_d field.  */
    int _mp_size;			/* abs(_mp_size) is the number of limbs the
            last field points to.  If _mp_size is
            negative this is a negative number.  */
    mp_limb_t_ *_mp_d;	 	/* Pointer to the limbs.  */
} __mpz_struct_;

typedef __mpz_struct_ mpz_t_[1];     //   __mpz_struct_[1]   = mpz_t_

typedef __mpz_struct_ *mpz_ptr_;
typedef const __mpz_struct_ *mpz_srcptr_;

__device__ void gmp_init_allocator_();

__device__ void gmp_set_serial_allocator_(libstl::SerialAllocator* allocator = libstl::mainAllocator);
__device__ void gmp_set_parallel_allocator_(libstl::ParallalAllocator* allocator);

__device__ void mpn_copyi_(mp_ptr_, mp_srcptr_, mp_size_t_);
__device__ void mpn_copyd_(mp_ptr_, mp_srcptr_, mp_size_t_);

__device__ int mpn_cmp_(mp_srcptr_, mp_srcptr_, mp_size_t_);

__device__ mp_limb_t_ mpn_add__1_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_limb_t_);
__device__ mp_limb_t_ mpn_add__n_(mp_ptr_, mp_srcptr_, mp_srcptr_, mp_size_t_);
__device__ mp_limb_t_ mpn_add_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_srcptr_, mp_size_t_);

__device__ mp_limb_t_ mpn_sub__1_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_limb_t_);
__device__ mp_limb_t_ mpn_sub__n_(mp_ptr_, mp_srcptr_, mp_srcptr_, mp_size_t_);
__device__ mp_limb_t_ mpn_sub_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_srcptr_, mp_size_t_);

__device__ mp_limb_t_ mpn_mul__1_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_limb_t_);
__device__ mp_limb_t_ mpn_add_mul_1_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_limb_t_);
__device__ mp_limb_t_ mpn_sub_mul_1_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_limb_t_);

__device__ mp_limb_t_ mpn_mul_(mp_ptr_, mp_srcptr_, mp_size_t_, mp_srcptr_, mp_size_t_);
__device__ void mpn_mul__n_(mp_ptr_, mp_srcptr_, mp_srcptr_, mp_size_t_);
__device__ void mpn_sqr_(mp_ptr_, mp_srcptr_, mp_size_t_);

__device__ mp_limb_t_ mpn_lshift_(mp_ptr_, mp_srcptr_, mp_size_t_, unsigned int);
__device__ mp_limb_t_ mpn_rshift_(mp_ptr_, mp_srcptr_, mp_size_t_, unsigned int);


__device__ mp_limb_t_ mpn_invert_3by2_(mp_limb_t_, mp_limb_t_);              
#define mpn_invert_limb_(x) mpn_invert_3by2_((x), 0)

__device__ size_t mpn_get_str_(unsigned char *, int, mp_ptr_, mp_size_t_);
__device__ mp_size_t_ mpn_set_str_(mp_ptr_, const unsigned char *, size_t, int);


// mpz

__device__ void mpz_init_(mpz_t_);
__device__ void mpz_init_(mpz_t_, mp_size_t_);
__device__ void mpz_init_2_(mpz_t_, mp_bitcnt_t_);
__device__ void mpz_clear_(mpz_t_);

#define mpz_odd_p_(z)   (((z)->_mp_size != 0) & (int) (z)->_mp_d[0])
#define mpz_even_p_(z)  (! mpz_odd_p_(z))


__device__ int mpz_sgn_(const mpz_t_);
__device__ int mpz_cmp__si_(const mpz_t_, long);
__device__ int mpz_cmp__ui_(const mpz_t_, unsigned long);
__device__ int mpz_cmp_(const mpz_t_, const mpz_t_);
__device__ int mpz_cmp_abs__ui_(const mpz_t_, unsigned long);
__device__ int mpz_cmp_abs_(const mpz_t_, const mpz_t_);

__device__ void mpz_abs_(mpz_t_, const mpz_t_);
__device__ void mpz_neg_(mpz_t_, const mpz_t_);
__device__ void mpz_swap_(mpz_t_, mpz_t_);

__device__ void mpz_add__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ void mpz_add_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_sub__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ void mpz_ui_sub_(mpz_t_, unsigned long, const mpz_t_);
__device__ void mpz_sub_(mpz_t_, const mpz_t_, const mpz_t_);

__device__ void mpz_mul__si_(mpz_t_, const mpz_t_, long int);
__device__ void mpz_mul__ui_(mpz_t_, const mpz_t_, unsigned long int);
__device__ void mpz_mul_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_mul__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);

__device__ void mpz_cdiv_q_r_(mpz_t_, mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_fdiv_q_r_(mpz_t_, mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_t_div_q_r_(mpz_t_, mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_cdiv_q_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_fdiv_q_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_t_div_q_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_cdiv_r_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_fdiv_r_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_t_div_r_(mpz_t_, const mpz_t_, const mpz_t_);

__device__ void mpz_cdiv_q__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);
__device__ void mpz_fdiv_q__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);
__device__ void mpz_t_div_q__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);
__device__ void mpz_cdiv_r__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);
__device__ void mpz_fdiv_r__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);
__device__ void mpz_t_div_r__2exp_(mpz_t_, const mpz_t_, mp_bitcnt_t_);

__device__ void mpz_mod_(mpz_t_, const mpz_t_, const mpz_t_);

__device__ void mpz_divexact_(mpz_t_, const mpz_t_, const mpz_t_);

__device__ int mpz_divisible_p_(const mpz_t_, const mpz_t_);

__device__ unsigned long mpz_cdiv_q_r__ui_(mpz_t_, mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_fdiv_q_r__ui_(mpz_t_, mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_t_div_q_r__ui_(mpz_t_, mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_cdiv_q__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_fdiv_q__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_t_div_q__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_cdiv_r__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_fdiv_r__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_t_div_r__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ unsigned long mpz_cdiv_ui_(const mpz_t_, unsigned long);
__device__ unsigned long mpz_fdiv_ui_(const mpz_t_, unsigned long);
__device__ unsigned long mpz_t_div_ui_(const mpz_t_, unsigned long);

__device__ unsigned long mpz_mod__ui_(mpz_t_, const mpz_t_, unsigned long);

__device__ void mpz_divexact__ui_(mpz_t_, const mpz_t_, unsigned long);

__device__ int mpz_divisible_ui_p_(const mpz_t_, unsigned long);

__device__ unsigned long mpz_gcd__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ void mpz_gcd_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_gcd_ext(mpz_t_, mpz_t_, mpz_t_, const mpz_t_, const mpz_t_);


/////////////////////////////// unfinished //////////////////////////////////////
__device__ void mpz_lcm__ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ void mpz_lcm_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ int mpz_invert_(mpz_t_, const mpz_t_, const mpz_t_);

__device__ void mpz_sqrt_rem_(mpz_t_, mpz_t_, const mpz_t_);
__device__ void mpz_sqrt_(mpz_t_, const mpz_t_);

__device__ void mpz_pow_ui_(mpz_t_, const mpz_t_, unsigned long);
__device__ void mpz_ui_pow_ui_(mpz_t_, unsigned long, unsigned long);
__device__ void mpz_powm_(mpz_t_, const mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_powm__ui_(mpz_t_, const mpz_t_, unsigned long, const mpz_t_);

__device__ void mpz_root_rem_(mpz_t_, mpz_t_, const mpz_t_, unsigned long);
__device__ int mpz_root_(mpz_t_, const mpz_t_, unsigned long);

__device__ void mpz_fac_ui_(mpz_t_, unsigned long);
__device__ void mpz_bin_uiui_(mpz_t_, unsigned long, unsigned long);


__device__ int mpz_t_stbit_(const mpz_t_, mp_bitcnt_t_);
__device__ void mpz_set_bit_(mpz_t_, mp_bitcnt_t_);
__device__ void mpz_clrbit_(mpz_t_, mp_bitcnt_t_);
__device__ void mpz_com_bit_(mpz_t_, mp_bitcnt_t_);   // 

__device__ void mpz_com_(mpz_t_, const mpz_t_);
__device__ void mpz_and_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_ior_(mpz_t_, const mpz_t_, const mpz_t_);
__device__ void mpz_xor_(mpz_t_, const mpz_t_, const mpz_t_);

__device__ mp_bitcnt_t_ mpz_popcount_(const mpz_t_);
__device__ mp_bitcnt_t_ mpz_hamdis_ (const mpz_t_, const mpz_t_);
__device__ mp_bitcnt_t_ mpz_scan0_(const mpz_t_, mp_bitcnt_t_);
__device__ mp_bitcnt_t_ mpz_scan1_(const mpz_t_, mp_bitcnt_t_);

//////////////////////////////// unfinished /////////////////////////////////////


__device__ int mpz_fits_slong_p_(const mpz_t_);
__device__ int mpz_fits_ulong_p_(const mpz_t_);
__device__ long int mpz_get_si_(const mpz_t_);
__device__ unsigned long int mpz_get_ui_(const mpz_t_);
__device__ size_t mpz_size_(const mpz_t_);
__device__ mp_limb_t_ mpz_getlimbn_(const mpz_t_, mp_size_t_);

__device__ void mpz_set__si_(mpz_t_, signed long int);
__device__ void mpz_set__ui_(mpz_t_, unsigned long int);
__device__ void mpz_set_(mpz_t_, const mpz_t_);

__device__ void mpz_init__set__si_(mpz_t_, signed long int);
__device__ void mpz_init__set__ui_(mpz_t_, unsigned long int);
__device__ void mpz_init__set_(mpz_t_, const mpz_t_);

__device__ size_t mpz_size_inbase_(const mpz_t_, int);
__device__ char *mpz_get_str_(char *, int, const mpz_t_);
__device__ int mpz_set__str_(mpz_t_, const char *, int);
__device__ int mpz_init__set__str_(mpz_t_, const char *, int);


//


#endif