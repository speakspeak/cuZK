#include "mini-mp-cuda.cuh"
#include <assert.h>
#include <stdio.h>

/* Macros */
#define GMP_LIMB_BITS (sizeof(mp_limb_t_) * CHAR_BIT)

#define GMP_LIMB_MAX (~ (mp_limb_t_) 0)
#define GMP_LIMB_HIGHBIT ((mp_limb_t_) 1 << (GMP_LIMB_BITS - 1))

#define GMP_HLIMB_BIT ((mp_limb_t_) 1 << (GMP_LIMB_BITS / 2))
#define GMP_LLIMB_MASK (GMP_HLIMB_BIT - 1)

#define GMP_ULONG_BITS (sizeof(unsigned long) * CHAR_BIT)
#define GMP_ULONG_HIGHBIT ((unsigned long) 1 << (GMP_ULONG_BITS - 1))

#define GMP_ABS(x) ((x) >= 0 ? (x) : -(x))
#define GMP_NEG_CAST(T,x) (-((T)((x) + 1) - 1))

#define GMP_MIN(a, b) ((a) < (b) ? (a) : (b))
#define GMP_MAX(a, b) ((a) > (b) ? (a) : (b))


#define gmp_assert_nocarry(x) do { \
    mp_limb_t_ __cy = x;		   \
  } while (0)

#define gmp_clz(count, x) do {						\
    mp_limb_t_ __clz_x = (x);						\
    unsigned __clz_c;							\
    for (__clz_c = 0;							\
	 (__clz_x & ((mp_limb_t_) 0xff << (GMP_LIMB_BITS - 8))) == 0;	\
	 __clz_c += 8)							\
      __clz_x <<= 8;							\
    for (; (__clz_x & GMP_LIMB_HIGHBIT) == 0; __clz_c++)		\
      __clz_x <<= 1;							\
    (count) = __clz_c;							\
  } while (0)

#define gmp_ctz(count, x) do {						\
    mp_limb_t_ __ctz_x = (x);						\
    unsigned __ctz_c = 0;						\
    gmp_clz(__ctz_c, __ctz_x & - __ctz_x);				\
    (count) = GMP_LIMB_BITS - 1 - __ctz_c;				\
  } while (0)

#define gmp_add_ssaaaa(sh, sl, ah, al, bh, bl) \
  do {									\
    mp_limb_t_ __x;							\
    __x = (al) + (bl);							\
    (sh) = (ah) + (bh) + (__x < (al));					\
    (sl) = __x;								\
  } while (0)

#define gmp_sub_ddmmss(sh, sl, ah, al, bh, bl) \
  do {									\
    mp_limb_t_ __x;							\
    __x = (al) - (bl);							\
    (sh) = (ah) - (bh) - ((al) < (bl));					\
    (sl) = __x;								\
  } while (0)

#define gmp_umul_ppmm(w1, w0, u, v)					\
  do {									\
    mp_limb_t_ __x0, __x1, __x2, __x3;					\
    unsigned __ul, __vl, __uh, __vh;					\
    mp_limb_t_ __u = (u), __v = (v);					\
									\
    __ul = __u & GMP_LLIMB_MASK;					\
    __uh = __u >> (GMP_LIMB_BITS / 2);					\
    __vl = __v & GMP_LLIMB_MASK;					\
    __vh = __v >> (GMP_LIMB_BITS / 2);					\
									\
    __x0 = (mp_limb_t_) __ul * __vl;					\
    __x1 = (mp_limb_t_) __ul * __vh;					\
    __x2 = (mp_limb_t_) __uh * __vl;					\
    __x3 = (mp_limb_t_) __uh * __vh;					\
									\
    __x1 += __x0 >> (GMP_LIMB_BITS / 2);/* this can't give carry */	\
    __x1 += __x2;		/* but this indeed can */		\
    if (__x1 < __x2)		/* did we get it? */			\
      __x3 += GMP_HLIMB_BIT;	/* yes, add it in the proper pos. */	\
									\
    (w1) = __x3 + (__x1 >> (GMP_LIMB_BITS / 2));			\
    (w0) = (__x1 << (GMP_LIMB_BITS / 2)) + (__x0 & GMP_LLIMB_MASK);	\
  } while (0)

#define gmp_udiv_qrnnd_preinv(q, r, nh, nl, d, di)			\
  do {									\
    mp_limb_t_ _qh, _ql, _r, _mask;					\
    gmp_umul_ppmm(_qh, _ql, (nh), (di));				\
    gmp_add_ssaaaa(_qh, _ql, _qh, _ql, (nh) + 1, (nl));		\
    _r = (nl) - _qh * (d);						\
    _mask = -(mp_limb_t_) (_r > _ql); /* both > and >= are OK */		\
    _qh += _mask;							\
    _r += _mask & (d);							\
    if (_r >= (d))							\
      {									\
	_r -= (d);							\
	_qh++;								\
      }									\
									\
    (r) = _r;								\
    (q) = _qh;								\
  } while (0)

#define gmp_udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)		\
  do {									\
    mp_limb_t_ _q0, _t1, _t0, _mask;					\
    gmp_umul_ppmm((q), _q0, (n2), (dinv));				\
    gmp_add_ssaaaa((q), _q0, (q), _q0, (n2), (n1));			\
									\
    /* Compute the two most significant limbs of n - q'd */		\
    (r1) = (n1) - (d1) * (q);						\
    gmp_sub_ddmmss((r1), (r0), (r1), (n0), (d1), (d0));		\
    gmp_umul_ppmm(_t1, _t0, (d0), (q));				\
    gmp_sub_ddmmss((r1), (r0), (r1), (r0), _t1, _t0);			\
    (q)++;								\
									\
    /* Conditionally adjust q and the remainders */			\
    _mask = - (mp_limb_t_) ((r1) >= _q0);				\
    (q) += _mask;							\
    gmp_add_ssaaaa((r1), (r0), (r1), (r0), _mask & (d1), _mask & (d0)); \
    if ((r1) >= (d1))							\
      {									\
	if ((r1) > (d1) || (r0) >= (d0))				\
	  {								\
	    (q)++;							\
	    gmp_sub_ddmmss((r1), (r0), (r1), (r0), (d1), (d0));	\
	  }								\
      }									\
  } while (0)


/* Swap macros. */
#define mp_limb_t__SWAP(x, y)						\
  do {									\
    mp_limb_t_ __mp_limb_t__swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_limb_t__swap__tmp;					\
  } while (0)
#define mp_size_t__SWAP(x, y)						\
  do {									\
    mp_size_t_ __mp_size_t__swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_size_t__swap__tmp;					\
  } while (0)
#define mp_bitcnt_t__SWAP(x,y)			\
  do {						\
    mp_bitcnt_t_ __mp_bitcnt_t__swap__tmp = (x);	\
    (x) = (y);					\
    (y) = __mp_bitcnt_t__swap__tmp;		\
  } while (0)
#define mp_ptr__SWAP(x, y)						\
  do {									\
    mp_ptr_ __mp_ptr__swap__tmp = (x);					\
    (x) = (y);								\
    (y) = __mp_ptr__swap__tmp;						\
  } while (0)
#define mp_srcptr__SWAP(x, y)						\
  do {									\
    mp_srcptr_ __mp_srcptr__swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_srcptr__swap__tmp;					\
  } while (0)

#define MPN_PTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    mp_ptr__SWAP(xp, yp);						\
    mp_size_t__SWAP(xs, ys);						\
  } while(0)
#define MPN_SRCPTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    mp_srcptr__SWAP(xp, yp);						\
    mp_size_t__SWAP(xs, ys);						\
  } while(0)

#define mpz_ptr__SWAP(x, y)						\
  do {									\
    mpz_ptr_ __mpz_ptr__swap__tmp = (x);					\
    (x) = (y);								\
    (y) = __mpz_ptr__swap__tmp;						\
  } while (0)
#define mpz_srcptr__SWAP(x, y)						\
  do {									\
    mpz_srcptr_ __mpz_srcptr__swap__tmp = (x);			\
    (x) = (y);								\
    (y) = __mpz_srcptr__swap__tmp;					\
  } while (0)

__device__ static bool disspace(unsigned char c)
{
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
}

__device__ static size_t dstrlen(const char *str)
{
    const char *cp =  str;
    while (*cp++);
    return (cp - str - 1);
}

__device__ static libstl::AllocatorManager* gmp_alloc_manager;

__device__ void gmp_init_allocator_()
{
    gmp_alloc_manager = (libstl::AllocatorManager*)libstl::allocate(sizeof(libstl::AllocatorManager));
    construct(gmp_alloc_manager);
}

__device__ void gmp_set_serial_allocator_(libstl::SerialAllocator* allocator)
{
    gmp_alloc_manager->setSerialAllocator(allocator);
}

__device__ void gmp_set_parallel_allocator_(libstl::ParallalAllocator* allocator)
{
    gmp_alloc_manager->setParallelAllocator(allocator);
}

__device__ static void gmp_die(const char *msg)
{
    assert(1 == 0);            
}

__device__ static inline void* gmp_default_alloc(size_t size)
{
    return gmp_alloc_manager->allocate(size);
}

__device__ static inline void* gmp_default_realloc(void* old, size_t old_size, size_t new_size)
{
    if (old_size == new_size)
        return (mp_ptr_)old;

    mp_ptr_ p = (mp_ptr_)gmp_default_alloc(new_size);

    int size = old_size < new_size ? old_size : new_size;

    for (int i = 0; i < size; i++)
        p[i] = ((mp_ptr_)old)[i];

    return p;
}

__device__ static void gmp_default_free(void *p, size_t size)
{
    
}

__device__ static void * (*gmp_allocate_func) (size_t) = gmp_default_alloc;
__device__ static void * (*gmp_reallocate_func) (void *, size_t, size_t) = gmp_default_realloc;
__device__ static void (*gmp_free_func) (void *, size_t) = gmp_default_free;


#define gmp_xalloc(size) ((*gmp_allocate_func)((size)))
#define gmp_free(p) ((*gmp_free_func) ((p), 0))

__device__ static mp_ptr_ gmp_xalloc_limbs(mp_size_t_ size)
{
    return (mp_ptr_)gmp_xalloc(size * sizeof (mp_limb_t_));
}

__device__ static mp_ptr_ gmp_xrealloc_limbs(mp_ptr_ old, mp_size_t_ old_size, mp_size_t_ size)
{
    // assert (size > 0);
    return (mp_ptr_)(*gmp_reallocate_func)(old, old_size, size * sizeof (mp_limb_t_));
}

__device__ void mpn_copyi_(mp_ptr_ d, mp_srcptr_ s, mp_size_t_ n)
{
    mp_size_t_ i;
    for (i = 0; i < n; i++)
        d[i] = s[i];
}

__device__ void mpn_copyd_(mp_ptr_ d, mp_srcptr_ s, mp_size_t_ n)
{
    while (n-- > 0)
        d[n] = s[n];
}

__device__ int mpn_cmp_(mp_srcptr_ ap, mp_srcptr_ bp, mp_size_t_ n)
{
    for (; n > 0; n--)
    {
        if (ap[n-1] < bp[n-1])
	          return -1;
        else if (ap[n-1] > bp[n-1])
	          return 1;
    }
    return 0;
}

__device__ static int mpn_cmp_4(mp_srcptr_ ap, mp_size_t_ an, mp_srcptr_ bp, mp_size_t_ bn)
{
    if (an > bn)
        return 1;
    else if (an < bn)
        return -1;
    else
        return mpn_cmp_(ap, bp, an);
}

__device__ static mp_size_t_ mpn_normalized_size(mp_srcptr_ xp, mp_size_t_ n)
{
  for (; n > 0 && xp[n-1] == 0; n--);

  return n;
}

#define mpn_zero_p(xp, n) (mpn_normalized_size((xp), (n)) == 0)

__device__ mp_limb_t_ mpn_add__1_(mp_ptr_ rp, mp_srcptr_ ap, mp_size_t_ n, mp_limb_t_ b)
{
    mp_size_t_ i;

    // assert (n > 0);

    for (i = 0; i < n; i++)
    {
        mp_limb_t_ r = ap[i] + b;
        /* Carry out */
        b = (r < b);
        rp[i] = r;
    }
    return b;
}

__device__ mp_limb_t_ mpn_add__n_(mp_ptr_ rp, mp_srcptr_ ap, mp_srcptr_ bp, mp_size_t_ n)
{
    mp_size_t_ i;
    mp_limb_t_ cy;

    for (i = 0, cy = 0; i < n; i++)
    {
        mp_limb_t_ a, b, r;
        a = ap[i]; 
        b = bp[i];
        r = a + cy;
        cy = (r < cy);
        r += b;
        cy += (r < b);
        rp[i] = r;
    }
    return cy;
}

__device__ mp_limb_t_ mpn_add_(mp_ptr_ rp, mp_srcptr_ ap, mp_size_t_ an, mp_srcptr_ bp, mp_size_t_ bn)
{
    mp_limb_t_ cy;

    // assert (an >= bn);

    cy = mpn_add__n_(rp, ap, bp, bn);
    if (an > bn)
        cy = mpn_add__1_(rp + bn, ap + bn, an - bn, cy);
    return cy;
}


__device__ mp_limb_t_ mpn_sub__1_ (mp_ptr_ rp, mp_srcptr_ ap, mp_size_t_ n, mp_limb_t_ b)
{
    mp_size_t_ i;

    // assert (n > 0);

    for (i = 0; i < n; i++)
    {
        mp_limb_t_ a = ap[i];
        /* Carry out */
        mp_limb_t_ cy = a < b;
        rp[i] = a - b;
        b = cy;
    }
    return b;
}

__device__ mp_limb_t_ mpn_sub__n_(mp_ptr_ rp, mp_srcptr_ ap, mp_srcptr_ bp, mp_size_t_ n)
{
    mp_size_t_ i;
    mp_limb_t_ cy;

    for (i = 0, cy = 0; i < n; i++)
    {
        mp_limb_t_ a, b;
        a = ap[i]; b = bp[i];
        b += cy;
        cy = (b < cy);
        cy += (a < b);
        rp[i] = a - b;
    }
    return cy;
}

__device__ mp_limb_t_ mpn_sub_(mp_ptr_ rp, mp_srcptr_ ap, mp_size_t_ an, mp_srcptr_ bp, mp_size_t_ bn)
{
    mp_limb_t_ cy;

    // assert (an >= bn);

    cy = mpn_sub__n_(rp, ap, bp, bn);
    if (an > bn)
        cy = mpn_sub__1_(rp + bn, ap + bn, an - bn, cy);
    return cy;
}

// rp = up * vl
__device__ mp_limb_t_ mpn_mul__1_(mp_ptr_ rp, mp_srcptr_ up, mp_size_t_ n, mp_limb_t_ vl)
{
    mp_limb_t_ ul, cl, hpl, lpl;

    // assert (n >= 1);

    cl = 0;
    do
    {
        ul = *up++;
        gmp_umul_ppmm(hpl, lpl, ul, vl);

        lpl += cl;
        cl = (lpl < cl) + hpl;

        *rp++ = lpl;
    }
    while (--n != 0);

    return cl;
}

// rp = rp + up * vl
__device__ mp_limb_t_ mpn_add_mul_1_(mp_ptr_ rp, mp_srcptr_ up, mp_size_t_ n, mp_limb_t_ vl)
{
    mp_limb_t_ ul, cl, hpl, lpl, rl;

    // assert (n >= 1);

    cl = 0;
    do
    {
        ul = *up++;
        gmp_umul_ppmm(hpl, lpl, ul, vl);

        lpl += cl;
        cl = (lpl < cl) + hpl;

        rl = *rp;
        lpl = rl + lpl;
        cl += lpl < rl;
        *rp++ = lpl;
    }
    while (--n != 0);

    return cl;
}

// rp = rp - up * vl
__device__ mp_limb_t_ mpn_sub_mul_1_(mp_ptr_ rp, mp_srcptr_ up, mp_size_t_ n, mp_limb_t_ vl)
{
    mp_limb_t_ ul, cl, hpl, lpl, rl;

    // assert (n >= 1);

    cl = 0;
    do
      {
        ul = *up++;
        gmp_umul_ppmm(hpl, lpl, ul, vl);

        lpl += cl;
        cl = (lpl < cl) + hpl;

        rl = *rp;
        lpl = rl - lpl;
        cl += lpl > rl;
        *rp++ = lpl;
      }
    while (--n != 0);

    return cl;
}


__device__ mp_limb_t_ mpn_mul_(mp_ptr_ rp, mp_srcptr_ up, mp_size_t_ un, mp_srcptr_ vp, mp_size_t_ vn)
{
    // assert (un >= vn);
    // assert (vn >= 1);

    /* We first multiply by the low order limb. This result can be
      stored, not added, to rp. We also avoid a loop for zeroing this
      way. */

    rp[un] = mpn_mul__1_(rp, up, un, vp[0]);
    rp += 1, vp += 1, vn -= 1;

    /* Now accumulate the product of up[] and the next higher limb from
      vp[]. */

    while (vn >= 1)
    {
        rp[un] = mpn_add_mul_1_(rp, up, un, vp[0]);
        rp += 1, vp += 1, vn -= 1;
    }
    return rp[un - 1];
}

__device__ void mpn_mul__n_(mp_ptr_ rp, mp_srcptr_ ap, mp_srcptr_ bp, mp_size_t_ n)
{
   mpn_mul_(rp, ap, n, bp, n);
}

__device__ void mpn_sqr_(mp_ptr_ rp, mp_srcptr_ ap, mp_size_t_ n)
{
    mpn_mul_(rp, ap, n, ap, n);
}

// rp = up << cnt
__device__ mp_limb_t_ mpn_lshift_(mp_ptr_ rp, mp_srcptr_ up, mp_size_t_ n, unsigned int cnt)
{
    mp_limb_t_ high_limb, low_limb;
    unsigned int tnc;
    mp_size_t_ i;
    mp_limb_t_ retval;

    // assert (n >= 1);
    // assert (cnt >= 1);
    // assert (cnt < GMP_LIMB_BITS);

    up += n;
    rp += n;

    tnc = GMP_LIMB_BITS - cnt;
    low_limb = *--up;
    retval = low_limb >> tnc;
    high_limb = (low_limb << cnt);

    for (i = n - 1; i != 0; i--)
    {
        low_limb = *--up;
        *--rp = high_limb | (low_limb >> tnc);
        high_limb = (low_limb << cnt);
    }
    *--rp = high_limb;

    return retval;
}

// rp = up >> cnt
__device__ mp_limb_t_ mpn_rshift_(mp_ptr_ rp, mp_srcptr_ up, mp_size_t_ n, unsigned int cnt)
{
    mp_limb_t_ high_limb, low_limb;
    unsigned int tnc;
    mp_size_t_ i;
    mp_limb_t_ retval;

    // assert (n >= 1);
    // assert (cnt >= 1);
    // assert (cnt < GMP_LIMB_BITS);

    tnc = GMP_LIMB_BITS - cnt;
    high_limb = *up++;
    retval = (high_limb << tnc);
    low_limb = high_limb >> cnt;

    for (i = n - 1; i != 0; i--)
    {
        high_limb = *up++;
        *rp++ = low_limb | (high_limb << tnc);
        low_limb = high_limb >> cnt;
    }
    *rp = low_limb;

    return retval;
}


/* MPN division interface. */
__device__ mp_limb_t_ mpn_invert_3by2_(mp_limb_t_ u1, mp_limb_t_ u0)
{
    mp_limb_t_ r, p, m;
    unsigned ul, uh;
    unsigned ql, qh;

    /* First, do a 2/1 inverse. */
    /* The inverse m is defined as floor( (B^2 - 1 - u1)/u1 ), so that 0 <
    * B^2 - (B + m) u1 <= u1 */
    // assert (u1 >= GMP_LIMB_HIGHBIT);

    ul = u1 & GMP_LLIMB_MASK;
    uh = u1 >> (GMP_LIMB_BITS / 2);

    qh = ~u1 / uh;
    r = ((~u1 - (mp_limb_t_) qh * uh) << (GMP_LIMB_BITS / 2)) | GMP_LLIMB_MASK;

    p = (mp_limb_t_) qh * ul;
    /* Adjustment steps taken from udiv_qrnnd_c */
    if (r < p)
    {
        qh--;
        r += u1;
        if (r >= u1) /* i.e. we didn't get carry when adding to r */
            if (r < p)
            {
                qh--;
                r += u1;
            } 
    }
    r -= p;

    /* Do a 3/2 division (with half limb size) */
    p = (r >> (GMP_LIMB_BITS / 2)) * qh + r;
    ql = (p >> (GMP_LIMB_BITS / 2)) + 1;

    /* By the 3/2 method, we don't need the high half limb. */
    r = (r << (GMP_LIMB_BITS / 2)) + GMP_LLIMB_MASK - ql * u1;

    if (r >= (p << (GMP_LIMB_BITS / 2)))
    {
        ql--;
        r += u1;
    }
    m = ((mp_limb_t_) qh << (GMP_LIMB_BITS / 2)) + ql;
    if (r >= u1)
    {
        m++;
        r -= u1;
    }

    if (u0 > 0)
    {
        mp_limb_t_ th, tl;
        r = ~r;
        r += u0;
        if (r < u0)
        {
            m--;
            if (r >= u1)
            {
                m--;
                r -= u1;
            }
            r -= u1;
        }
        gmp_umul_ppmm(th, tl, u0, m);
        r += th;
        if (r < th)
        {
            m--;
            if (r > u1 || (r == u1 && tl > u0))
                m--;
        }
    }

    return m;
}

struct gmp_div_inverse
{
    /* Normalization shift count. */
    unsigned shift;
    /* Normalized divisor (d0 unused for mpn_div_qr_1) */
    mp_limb_t_ d1, d0;
    /* Inverse, for 2/1 or 3/2. */
    mp_limb_t_ di;
};

__device__ static void mpn_div_qr_1_invert(struct gmp_div_inverse *inv, mp_limb_t_ d)
{
    unsigned shift;

    // assert (d > 0);
    gmp_clz(shift, d);
    inv->shift = shift;
    inv->d1 = d << shift;
    inv->di = mpn_invert_limb_(inv->d1);
}

__device__ static void mpn_div_qr_2_invert(struct gmp_div_inverse *inv, mp_limb_t_ d1, mp_limb_t_ d0)
{
    unsigned shift;

    // assert (d1 > 0);
    gmp_clz(shift, d1);
    inv->shift = shift;
    if (shift > 0)
    {
        d1 = (d1 << shift) | (d0 >> (GMP_LIMB_BITS - shift));
        d0 <<= shift;
    }
    inv->d1 = d1;
    inv->d0 = d0;
    inv->di = mpn_invert_3by2_(d1, d0);
}

__device__ static void mpn_div_qr_invert(struct gmp_div_inverse *inv, mp_srcptr_ dp, mp_size_t_ dn)
{
    // assert (dn > 0);

    if (dn == 1)
        mpn_div_qr_1_invert(inv, dp[0]);
    else if (dn == 2)
        mpn_div_qr_2_invert(inv, dp[1], dp[0]);
    else
    {
        unsigned shift;
        mp_limb_t_ d1, d0;

        d1 = dp[dn-1];
        d0 = dp[dn-2];
        // assert (d1 > 0);
        gmp_clz(shift, d1);
        inv->shift = shift;
        if (shift > 0)
        {
            d1 = (d1 << shift) | (d0 >> (GMP_LIMB_BITS - shift));
            d0 = (d0 << shift) | (dp[dn-3] >> (GMP_LIMB_BITS - shift));
        }
        inv->d1 = d1;
        inv->d0 = d0;
        inv->di = mpn_invert_3by2_(d1, d0);
    }
}

/* Not matching current public gmp interface, rather corresponding to
   the sbpi1_div_* functions. */
__device__ static mp_limb_t_ mpn_div_qr_1_preinv(mp_ptr_ qp, mp_srcptr_ np, mp_size_t_ nn, const struct gmp_div_inverse *inv)
{
    mp_limb_t_ d, di;
    mp_limb_t_ r;
    mp_ptr_ tp = NULL;

    if (inv->shift > 0)
    {
        tp = gmp_xalloc_limbs(nn);
        r = mpn_lshift_(tp, np, nn, inv->shift);
        np = tp;
    }
    else
        r = 0;

    d = inv->d1;
    di = inv->di;
    while (nn-- > 0)
    {
        mp_limb_t_ q;

        gmp_udiv_qrnnd_preinv(q, r, r, np[nn], d, di);
        if (qp)
            qp[nn] = q;
    }
    if (inv->shift > 0)
        gmp_free(tp);

    return r >> inv->shift;
}

__device__ static mp_limb_t_ mpn_div_qr_1(mp_ptr_ qp, mp_srcptr_ np, mp_size_t_ nn, mp_limb_t_ d)
{
    // assert (d > 0);

    /* Special case for powers of two. */
    if (d > 1 && (d & (d-1)) == 0)
    {
        unsigned shift;
        mp_limb_t_ r = np[0] & (d-1);
        gmp_ctz(shift, d);
        if (qp)
            mpn_rshift_(qp, np, nn, shift);

        return r;
    }
    else
    {
        struct gmp_div_inverse inv;
        mpn_div_qr_1_invert(&inv, d);
        return mpn_div_qr_1_preinv(qp, np, nn, &inv);
    }
}

__device__ static void mpn_div_qr_2_preinv(mp_ptr_ qp, mp_ptr_ rp, mp_srcptr_ np, mp_size_t_ nn, const struct gmp_div_inverse *inv)
{
    unsigned shift;
    mp_size_t_ i;
    mp_limb_t_ d1, d0, di, r1, r0;
    mp_ptr_ tp;

    // assert (nn >= 2);
    shift = inv->shift;
    d1 = inv->d1;
    d0 = inv->d0;
    di = inv->di;

    if (shift > 0)
    {
        tp = gmp_xalloc_limbs(nn);
        r1 = mpn_lshift_(tp, np, nn, shift);
        np = tp;
    }
    else
        r1 = 0;

    r0 = np[nn - 1];

    for (i = nn - 2; i >= 0; i--)
    {
        mp_limb_t_ n0, q;
        n0 = np[i];
        gmp_udiv_qr_3by2(q, r1, r0, r1, r0, n0, d1, d0, di);

        if (qp)
            qp[i] = q;
    }

    if (shift > 0)
    {
        // assert ((r0 << (GMP_LIMB_BITS - shift)) == 0);
        r0 = (r0 >> shift) | (r1 << (GMP_LIMB_BITS - shift));
        r1 >>= shift;

        gmp_free(tp);
    }

    rp[1] = r1;
    rp[0] = r0;
}


__device__ static void mpn_div_qr_pi1(mp_ptr_ qp,
		mp_ptr_ np, mp_size_t_ nn, mp_limb_t_ n1,
		mp_srcptr_ dp, mp_size_t_ dn,
		mp_limb_t_ dinv)
{
    mp_size_t_ i;

    mp_limb_t_ d1, d0;
    mp_limb_t_ cy, cy1;
    mp_limb_t_ q;

    // assert (dn > 2);
    // assert (nn >= dn);
    // assert ((dp[dn-1] & GMP_LIMB_HIGHBIT) != 0);

    d1 = dp[dn - 1];
    d0 = dp[dn - 2];

    /* Iteration variable is the index of the q limb.
    *
    * We divide <n1, np[dn-1+i], np[dn-2+i], np[dn-3+i],..., np[i]>
    * by            <d1,          d0,        dp[dn-3],  ..., dp[0] >
    */

    for (i = nn - dn; i >= 0; i--)
    {
        mp_limb_t_ n0 = np[dn-1 + i];

        if (n1 == d1 && n0 == d0)
        {
              q = GMP_LIMB_MAX;
              mpn_sub_mul_1_(np+i, dp, dn, q);
              n1 = np[dn-1+i];	/* update n1, last loop's value will now be invalid */
        }
        else
        {
            gmp_udiv_qr_3by2(q, n1, n0, n1, n0, np[dn-2+i], d1, d0, dinv);

            cy = mpn_sub_mul_1_(np + i, dp, dn-2, q);

            cy1 = n0 < cy;
            n0 = n0 - cy;
            cy = n1 < cy1;
            n1 = n1 - cy1;
            np[dn-2+i] = n0;

            if (cy != 0)
            {
                n1 += d1 + mpn_add__n_(np + i, np + i, dp, dn - 1);
                q--;
            }
        }

        if (qp)
            qp[i] = q;
      }

    np[dn - 1] = n1;
}

__device__ static void mpn_div_qr_preinv(mp_ptr_ qp, mp_ptr_ np, mp_size_t_ nn,
		   mp_srcptr_ dp, mp_size_t_ dn,
		   const struct gmp_div_inverse *inv)
{
  // assert (dn > 0);
  // assert (nn >= dn);

    if (dn == 1)
        np[0] = mpn_div_qr_1_preinv(qp, np, nn, inv);
    else if (dn == 2)
        mpn_div_qr_2_preinv(qp, np, np, nn, inv);
    else
    {
        mp_limb_t_ nh;
        unsigned shift;

        // assert (dp[dn-1] & GMP_LIMB_HIGHBIT);

        shift = inv->shift;
        if (shift > 0)
            nh = mpn_lshift_(np, np, nn, shift);
        else
            nh = 0;

        // assert (inv->d1 == dp[dn-1]);
        // assert (inv->d0 == dp[dn-2]);

        mpn_div_qr_pi1(qp, np, nn, nh, dp, dn, inv->di);

        if (shift > 0)
            gmp_assert_nocarry(mpn_rshift_(np, np, dn, shift));
    }
}

__device__ static void mpn_div_qr(mp_ptr_ qp, mp_ptr_ np, mp_size_t_ nn, mp_srcptr_ dp, mp_size_t_ dn)
{
    struct gmp_div_inverse inv;
    mp_ptr_ tp = NULL;

    // assert (dn > 0);
    // assert (nn >= dn);

    mpn_div_qr_invert(&inv, dp, dn);
    if (dn > 2 && inv.shift > 0)
    {
        tp = gmp_xalloc_limbs(dn);
        gmp_assert_nocarry(mpn_lshift_(tp, dp, dn, inv.shift));
        dp = tp;
    }
    mpn_div_qr_preinv(qp, np, nn, dp, dn, &inv);


    if (tp)
        gmp_free(tp);
}


/* MPN base conversion. */
__device__ static unsigned mpn_base_power_of_two_p(unsigned b)
{
  switch (b)
  {
      case 2: return 1;
      case 4: return 2;
      case 8: return 3;
      case 16: return 4;
      case 32: return 5;
      case 64: return 6;
      case 128: return 7;
      case 256: return 8;
      default: return 0;
  }
}

struct mpn_base_info
{
    /* bb is the largest power of the base which fits in one limb, and
      exp is the corresponding exponent. */
    unsigned exp;
    mp_limb_t_ bb;
};

__device__ static void mpn_get_base_info(struct mpn_base_info *info, mp_limb_t_ b)
{
    mp_limb_t_ m;
    mp_limb_t_ p;
    unsigned exp;

    m = GMP_LIMB_MAX / b;
    for (exp = 1, p = b; p <= m; exp++)
        p *= b;

    info->exp = exp;
    info->bb = p;
}

__device__ static mp_bitcnt_t_ mpn_limb_size_in_base_2(mp_limb_t_ u)
{
    unsigned shift;

    // assert (u > 0);
    gmp_clz(shift, u);
    return GMP_LIMB_BITS - shift;
}


__device__ static size_t mpn_get_str__bits(unsigned char *sp, unsigned bits, mp_srcptr_ up, mp_size_t_ un)
{
    unsigned char mask;
    size_t sn, j;
    mp_size_t_ i;
    int shift;

    sn = ((un - 1) * GMP_LIMB_BITS + mpn_limb_size_in_base_2(up[un-1]) + bits - 1) / bits;

    mask = (1U << bits) - 1;

    for (i = 0, j = sn, shift = 0; j-- > 0;)
    {
        unsigned char digit = up[i] >> shift;

        shift += bits;

        if (shift >= GMP_LIMB_BITS && ++i < un)
        {
            shift -= GMP_LIMB_BITS;
            digit |= up[i] << (bits - shift);
        }
        sp[j] = digit & mask;
    }
    return sn;
}


/* We generate digits from the least significant end, and reverse at
   the end. */
__device__ static size_t mpn_limb_get_str(unsigned char *sp, mp_limb_t_ w, const struct gmp_div_inverse *binv)
{
    mp_size_t_ i;
    for (i = 0; w > 0; i++)
    {
        mp_limb_t_ h, l, r;

        h = w >> (GMP_LIMB_BITS - binv->shift);
        l = w << binv->shift;

        gmp_udiv_qrnnd_preinv(w, r, h, l, binv->d1, binv->di);
        // assert ( (r << (GMP_LIMB_BITS - binv->shift)) == 0);
        r >>= binv->shift;

        sp[i] = r;
    }
    return i;
}

__device__ static size_t mpn_get_str__other(unsigned char *sp,
		   int base, const struct mpn_base_info *info,
		   mp_ptr_ up, mp_size_t_ un)
{
    struct gmp_div_inverse binv;
    size_t sn;
    size_t i;

    mpn_div_qr_1_invert(&binv, base);

    sn = 0;

    if (un > 1)
    {
        struct gmp_div_inverse bbinv;
        mpn_div_qr_1_invert(&bbinv, info->bb);

        do
        {
            mp_limb_t_ w;
            size_t done;
            w = mpn_div_qr_1_preinv(up, up, un, &bbinv);
            un -= (up[un-1] == 0);
            done = mpn_limb_get_str(sp + sn, w, &binv);

            for (sn += done; done < info->exp; done++)
                sp[sn++] = 0;
        }
        while (un > 1);
    }
    sn += mpn_limb_get_str(sp + sn, up[0], &binv);

    /* Reverse order */
    for (i = 0; 2 * i + 1 < sn; i++)
    {
        unsigned char t = sp[i];
        sp[i] = sp[sn - i - 1];
        sp[sn - i - 1] = t;
    }

    return sn;
}

__device__ size_t mpn_get_str_(unsigned char *sp, int base, mp_ptr_ up, mp_size_t_ un)
{
    unsigned bits;

    // assert (un > 0);
    // assert (up[un-1] > 0);

    bits = mpn_base_power_of_two_p(base);
    if (bits)
        return mpn_get_str__bits(sp, bits, up, un);
    else
    {
        struct mpn_base_info info;

        mpn_get_base_info(&info, base);
        return mpn_get_str__other(sp, base, &info, up, un);
    }
}

__device__ static mp_size_t_ mpn_set_str__bits(mp_ptr_ rp, const unsigned char *sp, size_t sn, unsigned bits)
{
    mp_size_t_ rn;
    size_t j;
    unsigned shift;

    for (j = sn, rn = 0, shift = 0; j-- > 0; )
    {
        if (shift == 0)
        {
            rp[rn++] = sp[j];
            shift += bits;
        }
        else
        {
            rp[rn-1] |= (mp_limb_t_) sp[j] << shift;
            shift += bits;
            if (shift >= GMP_LIMB_BITS)
            {
                shift -= GMP_LIMB_BITS;
                if (shift > 0)
                    rp[rn++] = (mp_limb_t_) sp[j] >> (bits - shift);
            }
        }
    }
    rn = mpn_normalized_size(rp, rn);
    return rn;
}

__device__ static mp_size_t_ mpn_set_str__other(mp_ptr_ rp, const unsigned char *sp, size_t sn,
		   mp_limb_t_ b, const struct mpn_base_info *info)
{
    mp_size_t_ rn;
    mp_limb_t_ w;
    unsigned first;
    unsigned k;
    size_t j;

    first = 1 + (sn - 1) % info->exp;

    j = 0;
    w = sp[j++];
    for (k = 1; k < first; k++)
        w = w * b + sp[j++];

    rp[0] = w;

    for (rn = (w > 0); j < sn;)
    {
        mp_limb_t_ cy;

        w = sp[j++];
        for (k = 1; k < info->exp; k++)
            w = w * b + sp[j++];

        cy = mpn_mul__1_(rp, rp, rn, info->bb);
        cy += mpn_add__1_(rp, rp, rn, w);
        if (cy > 0)
            rp[rn++] = cy;
    }
    // assert (j == sn);

    return rn;
}

__device__ mp_size_t_ mpn_set_str_(mp_ptr_ rp, const unsigned char *sp, size_t sn, int base)
{
    unsigned bits;

    if (sn == 0)
        return 0;

    bits = mpn_base_power_of_two_p(base);
    if (bits)
        return mpn_set_str__bits(rp, sp, sn, bits);
    else
    {
        struct mpn_base_info info;

        mpn_get_base_info(&info, base);
        return mpn_set_str__other(rp, sp, sn, base, &info);
    }
}


/* MPZ interface */
__device__ void mpz_init_(mpz_t_ r)
{
    r->_mp_alloc = 1;           
    r->_mp_size = 0;            
    r->_mp_d = gmp_xalloc_limbs(1);
}

__device__ void mpz_init_(mpz_t_ r, mp_size_t_ size)
{
    r->_mp_alloc = size;
    r->_mp_size = 0;
    r->_mp_d = gmp_xalloc_limbs(size);
}

/* The utility of this function is a bit limited, since many functions
   assings the result variable using mpz_swap_. */
__device__ void mpz_init_2_(mpz_t_ r, mp_bitcnt_t_ bits)
{
    mp_size_t_ rn;

    bits -= (bits != 0);		/* Round down, except if 0 */
    rn = 1 + bits / GMP_LIMB_BITS;

    r->_mp_alloc = rn;
    r->_mp_size = 0;
    r->_mp_d = gmp_xalloc_limbs(rn);
}

__device__ void mpz_clear_(mpz_t_ r)
{
    gmp_free(r->_mp_d);
}

__device__ static void * mpz_realloc(mpz_t_ r, mp_size_t_ size)
{
    size = GMP_MAX(size, 1);

    r->_mp_d = gmp_xrealloc_limbs(r->_mp_d, r->_mp_alloc, size);
    r->_mp_alloc = size;

    if (GMP_ABS(r->_mp_size) > size)
        r->_mp_size = 0;

    return r->_mp_d;
}

/* Realloc for an mpz_t_ WHAT if it has less than NEEDED limbs.  */
#define MPZ_REALLOC(z,n) ((n) > (z)->_mp_alloc			\
			  ? mpz_realloc(z,n)			\
			  : (z)->_mp_d)

/* MPZ assignment and basic conversions. */
__device__ void mpz_set__si_(mpz_t_ r, signed long int x)
{
    if (x >= 0)
        mpz_set__ui_(r, x);
    else /* (x < 0) */
    {
        r->_mp_size = -1;
        r->_mp_d[0] = GMP_NEG_CAST(unsigned long int, x);
    }
}

__device__ void mpz_set__ui_(mpz_t_ r, unsigned long int x)
{
    if (x > 0)
    {
        r->_mp_size = 1;
        r->_mp_d[0] = x;
    }
    else
        r->_mp_size = 0;
}

__device__ void mpz_set_(mpz_t_ r, const mpz_t_ x)
{
  /* Allow the NOP r == x */
  if (r != x)
  {
      mp_size_t_ n;
      mp_ptr_ rp;

      n = GMP_ABS(x->_mp_size);
      rp = (mp_ptr_) MPZ_REALLOC(r, n);

      mpn_copyi_(rp, x->_mp_d, n);
      r->_mp_size = x->_mp_size;
   }
}


__device__ void mpz_init__set__si_(mpz_t_ r, signed long int x)
{
    mpz_init_(r);
    mpz_set__si_(r, x);
}

__device__ void mpz_init__set__ui_(mpz_t_ r, unsigned long int x)
{
    mpz_init_(r);
    mpz_set__ui_(r, x);
}

__device__ void mpz_init__set_(mpz_t_ r, const mpz_t_ x)
{
    mpz_init_(r);
    mpz_set_(r, x);
}

__device__ int mpz_fits_slong_p_(const mpz_t_ u)
{
    mp_size_t_ us = u->_mp_size;

    if (us == 0)
        return 1;
    else if (us == 1)
        return u->_mp_d[0] < GMP_LIMB_HIGHBIT;
    else if (us == -1)
        return u->_mp_d[0] <= GMP_LIMB_HIGHBIT;
    else
        return 0;
}

__device__ int mpz_fits_ulong_p_(const mpz_t_ u)
{
    mp_size_t_ us = u->_mp_size;

    return us == 0 || us == 1;
}

__device__ long int mpz_get_si_(const mpz_t_ u)
{
    mp_size_t_ us = u->_mp_size;

    if (us > 0)
        return (long)(u->_mp_d[0] & ~GMP_LIMB_HIGHBIT);
    else if (us < 0)
        return (long)(- u->_mp_d[0] | GMP_LIMB_HIGHBIT);
    else
        return 0;
}

__device__ unsigned long int mpz_get_ui_(const mpz_t_ u)
{
    return u->_mp_size == 0 ? 0 : u->_mp_d[0];
}

__device__ size_t mpz_size_(const mpz_t_ u)
{
    return GMP_ABS(u->_mp_size);
}

__device__ mp_limb_t_ mpz_getlimbn_(const mpz_t_ u, mp_size_t_ n)
{
    if (n >= 0 && n < GMP_ABS(u->_mp_size))
        return u->_mp_d[n];
    else
        return 0;
}


__device__ int mpz_sgn_(const mpz_t_ u)
{
    mp_size_t_ usize = u->_mp_size;

    if (usize > 0)
        return 1;
    else if (usize < 0)
        return -1;
    else
        return 0;
}

__device__ int mpz_cmp__si_(const mpz_t_ u, long v)
{
    mp_size_t_ usize = u->_mp_size;

    if (usize < -1)
        return -1;
    else if (v >= 0)
        return mpz_cmp__ui_(u, v);
    else if (usize >= 0)
        return 1;
    else /* usize == -1 */
    {
        mp_limb_t_ ul = u->_mp_d[0];
        if ((mp_limb_t_)GMP_NEG_CAST(unsigned long int, v) < ul)
            return -1;
        else if ( (mp_limb_t_)GMP_NEG_CAST(unsigned long int, v) > ul)
            return 1;
    }
    return 0;
}

__device__ int mpz_cmp__ui_(const mpz_t_ u, unsigned long v)
{
    mp_size_t_ usize = u->_mp_size;

    if (usize > 1)
        return 1;
    else if (usize < 0)
        return -1;
    else
    {
        mp_limb_t_ ul = (usize > 0) ? u->_mp_d[0] : 0;
        if (ul > v)
            return 1;
        else if (ul < v)
            return -1;
    }
    return 0;
}

__device__ int mpz_cmp_(const mpz_t_ a, const mpz_t_ b)
{
    mp_size_t_ asize = a->_mp_size;
    mp_size_t_ bsize = b->_mp_size;

    if (asize > bsize)
        return 1;
    else if (asize < bsize)
        return -1;
    else if (asize > 0)
        return mpn_cmp_(a->_mp_d, b->_mp_d, asize);
    else if (asize < 0)
        return -mpn_cmp_(a->_mp_d, b->_mp_d, -asize);
    else
        return 0;
}

__device__ int mpz_cmp_abs__ui_(const mpz_t_ u, unsigned long v)
{
    mp_size_t_ un = GMP_ABS(u->_mp_size);
    mp_limb_t_ ul;

    if (un > 1)
        return 1;

    ul = (un == 1) ? u->_mp_d[0] : 0;

    if (ul > v)
        return 1;
    else if (ul < v)
        return -1;
    else
        return 0;
}

__device__ int mpz_cmp_abs_(const mpz_t_ u, const mpz_t_ v)
{
    return mpn_cmp_4(u->_mp_d, GMP_ABS(u->_mp_size),
        v->_mp_d, GMP_ABS(v->_mp_size));
}

__device__ void mpz_abs_(mpz_t_ r, const mpz_t_ u)
{
    if (r != u)
        mpz_set_(r, u);

    r->_mp_size = GMP_ABS(r->_mp_size);
}

__device__ void mpz_neg_(mpz_t_ r, const mpz_t_ u)
{
    if (r != u)
        mpz_set_(r, u);

    r->_mp_size = -r->_mp_size;
}

__device__ void mpz_swap_(mpz_t_ u, mpz_t_ v)
{
    mp_size_t__SWAP(u->_mp_size, v->_mp_size);
    mp_size_t__SWAP(u->_mp_alloc, v->_mp_alloc);
    mp_ptr__SWAP(u->_mp_d, v->_mp_d);
}



/* Adds to the absolute value. Returns new size, but doesn't store it. */
__device__ static mp_size_t_ mpz_abs__add_ui(mpz_t_ r, const mpz_t_ a, unsigned long b)
{
    mp_size_t_ an;
    mp_ptr_ rp;
    mp_limb_t_ cy;

    an = GMP_ABS(a->_mp_size);
    if (an == 0)
    {
        r->_mp_d[0] = b;
        return b > 0;
    }

    rp = (mp_ptr_)MPZ_REALLOC(r, an + 1);

    cy = mpn_add__1_(rp, a->_mp_d, an, b);
    rp[an] = cy;
    an += (cy > 0);

    return an;
}

/* Subtract from the absolute value. Returns new size, (or -1 on underflow),
   but doesn't store it. */
__device__ static mp_size_t_ mpz_abs__sub_ui(mpz_t_ r, const mpz_t_ a, unsigned long b)
{
    mp_size_t_ an = GMP_ABS(a->_mp_size);
    mp_ptr_ rp = (mp_ptr_)MPZ_REALLOC(r, an);

    if (an == 0)
    {
        rp[0] = b;
        return -(b > 0);
    }
    else if (an == 1 && a->_mp_d[0] < b)
    {
        rp[0] = b - a->_mp_d[0];
        return -1;
    }
    else
    {
        gmp_assert_nocarry(mpn_sub__1_(rp, a->_mp_d, an, b));
        return mpn_normalized_size(rp, an);
    }
}

__device__ void mpz_add__ui_(mpz_t_ r, const mpz_t_ a, unsigned long b)
{
    if (a->_mp_size >= 0)
        r->_mp_size = mpz_abs__add_ui(r, a, b);
    else
        r->_mp_size = -mpz_abs__sub_ui(r, a, b);
}

__device__ void mpz_sub__ui_(mpz_t_ r, const mpz_t_ a, unsigned long b)
{
  if (a->_mp_size < 0)
      r->_mp_size = -mpz_abs__add_ui(r, a, b);
  else
      r->_mp_size = mpz_abs__sub_ui(r, a, b);
}

__device__ void mpz_ui_sub_(mpz_t_ r, unsigned long a, const mpz_t_ b)
{
  if (b->_mp_size < 0)
      r->_mp_size = mpz_abs__add_ui(r, b, a);
  else
      r->_mp_size = -mpz_abs__sub_ui(r, b, a);
}

__device__ static mp_size_t_ mpz_abs__add(mpz_t_ r, const mpz_t_ a, const mpz_t_ b)
{
    mp_size_t_ an = GMP_ABS(a->_mp_size);
    mp_size_t_ bn = GMP_ABS(b->_mp_size);
    mp_size_t_ rn;
    mp_ptr_ rp;
    mp_limb_t_ cy;

    rn = GMP_MAX(an, bn);
    rp = (mp_ptr_)MPZ_REALLOC(r, rn + 1);
    if (an >= bn)
        cy = mpn_add_(rp, a->_mp_d, an, b->_mp_d, bn);
    else
        cy = mpn_add_(rp, b->_mp_d, bn, a->_mp_d, an);

    rp[rn] = cy;

    return rn + (cy > 0);
}

__device__ static mp_size_t_ mpz_abs__sub(mpz_t_ r, const mpz_t_ a, const mpz_t_ b)
{
    mp_size_t_ an = GMP_ABS(a->_mp_size);
    mp_size_t_ bn = GMP_ABS(b->_mp_size);
    int cmp;
    mp_ptr_ rp;

    cmp = mpn_cmp_4(a->_mp_d, an, b->_mp_d, bn);
    if (cmp > 0)
    {
        rp = (mp_ptr_)MPZ_REALLOC(r, an);
        gmp_assert_nocarry(mpn_sub_(rp, a->_mp_d, an, b->_mp_d, bn));
        return mpn_normalized_size(rp, an);
    }
    else if (cmp < 0)
    {
        rp = (mp_ptr_)MPZ_REALLOC(r, bn);
        gmp_assert_nocarry(mpn_sub_(rp, b->_mp_d, bn, a->_mp_d, an));
        return -mpn_normalized_size(rp, bn);
    }
    else
        return 0;
}

__device__ void mpz_add_(mpz_t_ r, const mpz_t_ a, const mpz_t_ b)
{
    mp_size_t_ rn;

    if ((a->_mp_size ^ b->_mp_size) >= 0)
        rn = mpz_abs__add(r, a, b);
    else
        rn = mpz_abs__sub(r, a, b);

    r->_mp_size = a->_mp_size >= 0 ? rn : - rn;
}

__device__ void mpz_sub_(mpz_t_ r, const mpz_t_ a, const mpz_t_ b)
{
    mp_size_t_ rn;

    if ((a->_mp_size ^ b->_mp_size) >= 0)
        rn = mpz_abs__sub(r, a, b);
    else
        rn = mpz_abs__add(r, a, b);

    r->_mp_size = a->_mp_size >= 0 ? rn : - rn;
}


/* MPZ multiplication */
__device__ void mpz_mul__si_(mpz_t_ r, const mpz_t_ u, long int v)
{
  if (v < 0)
  {
      mpz_mul__ui_(r, u, GMP_NEG_CAST(unsigned long int, v));
      mpz_neg_(r, r);
  }
  else
      mpz_mul__ui_(r, u, (unsigned long int) v);
}

__device__ void mpz_mul__ui_(mpz_t_ r, const mpz_t_ u, unsigned long int v)
{
    mp_size_t_ un;
    mpz_t_ t;
    mp_ptr_ tp;
    mp_limb_t_ cy;

    un = GMP_ABS(u->_mp_size);

    if (un == 0 || v == 0)
    {
        r->_mp_size = 0;
        return;
    }

    mpz_init_2_(t, (un + 1) * GMP_LIMB_BITS);

    tp = t->_mp_d;
    cy = mpn_mul__1_(tp, u->_mp_d, un, v);
    tp[un] = cy;

    t->_mp_size = un + (cy > 0);
    if (u->_mp_size < 0)
        t->_mp_size = - t->_mp_size;

    mpz_swap_(r, t);
    mpz_clear_(t);
}

__device__ void mpz_mul_(mpz_t_ r, const mpz_t_ u, const mpz_t_ v)
{
    int sign;
    mp_size_t_ un, vn, rn;
    mpz_t_ t;
    mp_ptr_ tp;

    un = GMP_ABS(u->_mp_size);
    vn = GMP_ABS(v->_mp_size);

    if (un == 0 || vn == 0)
    {
        r->_mp_size = 0;
        return;
    }

    sign = (u->_mp_size ^ v->_mp_size) < 0;

    mpz_init_2_(t, (un + vn) * GMP_LIMB_BITS);

    tp = t->_mp_d;
    if (un >= vn)
        mpn_mul_(tp, u->_mp_d, un, v->_mp_d, vn);
    else
        mpn_mul_(tp, v->_mp_d, vn, u->_mp_d, un);

    rn = un + vn;
    rn -= tp[rn-1] == 0;

    t->_mp_size = sign ? - rn : rn;
    mpz_swap_(r, t);
    mpz_clear_(t);
}

__device__ void mpz_mul__2exp_(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ bits)
{
    mp_size_t_ un, rn;
    mp_size_t_ limbs;
    unsigned shift;
    mp_ptr_ rp;

    un = GMP_ABS(u->_mp_size);
    if (un == 0)
    {
        r->_mp_size = 0;
        return;
    }

    limbs = bits / GMP_LIMB_BITS;
    shift = bits % GMP_LIMB_BITS;

    rn = un + limbs + (shift > 0);
    rp = (mp_ptr_)MPZ_REALLOC(r, rn);
    if (shift > 0)
    {
        mp_limb_t_ cy = mpn_lshift_(rp + limbs, u->_mp_d, un, shift);
        rp[rn-1] = cy;
        rn -= (cy == 0);
    }
    else
        mpn_copyd_(rp + limbs, u->_mp_d, un);


    while (limbs > 0)
        rp[--limbs] = 0;

    r->_mp_size = (u->_mp_size < 0) ? - rn : rn;
}

/* MPZ division */
enum mpz_div_round_mode { GMP_DIV_FLOOR, GMP_DIV_CEIL, GMP_DIV_TRUNC };

/* Allows q or r to be zero. Returns 1 iff remainder is non-zero. */
__device__ static int mpz_div_qr(mpz_t_ q, mpz_t_ r,
	    const mpz_t_ n, const mpz_t_ d, enum mpz_div_round_mode mode)
{
    mp_size_t_ ns, ds, nn, dn, qs;
    ns = n->_mp_size;
    ds = d->_mp_size;

    if (ds == 0)
        gmp_die("mpz_div_qr: Divide by zero.");

    if (ns == 0)
    {
        if (q)
            q->_mp_size = 0;
        if (r)
            r->_mp_size = 0;
        return 0;
    }

    nn = GMP_ABS(ns);
    dn = GMP_ABS(ds);

    qs = ds ^ ns;

    if (nn < dn)
    {
        if (mode == GMP_DIV_CEIL && qs >= 0)
        {
      /* q = 1, r = n - d */
          if (r)
              mpz_sub_(r, n, d);
          if (q)
              mpz_set__ui_(q, 1);
        }
        else if (mode == GMP_DIV_FLOOR && qs < 0)
        {
      /* q = -1, r = n + d */
          if (r)
              mpz_add_(r, n, d);
          if (q)
              mpz_set__si_(q, -1);
        }
        else
        {
          /* q = 0, r = d */
          if (r)
              mpz_set_(r, n);
          if (q)
              q->_mp_size = 0;
        }
        return 1;
    }
    else
    {
        mp_ptr_ np, qp;
        mp_size_t_ qn, rn;
        mpz_t_ tq, tr;

        mpz_init_(tr, 5);
        mpz_set_(tr, n);
        np = tr->_mp_d;

        qn = nn - dn + 1;

        if (q)
        {
            mpz_init_2_(tq, qn * GMP_LIMB_BITS);
            qp = tq->_mp_d;
        }
        else
            qp = NULL;

        mpn_div_qr(qp, np, nn, d->_mp_d, dn);    

        if (qp)
        {
            qn -= (qp[qn-1] == 0);

            tq->_mp_size = qs < 0 ? -qn : qn;
        }
        rn = mpn_normalized_size(np, dn);
        tr->_mp_size = ns < 0 ? - rn : rn;


        if (mode == GMP_DIV_FLOOR && qs < 0 && rn != 0)
        {
            if (q)
                mpz_sub__ui_ (tq, tq, 1);
            if (r)
                mpz_add_ (tr, tr, d);
        }
        else if (mode == GMP_DIV_CEIL && qs >= 0 && rn != 0)
        {
            if (q)
                mpz_add__ui_(tq, tq, 1);
            if (r)
                mpz_sub_(tr, tr, d);
        }

        if (q)
        {
            mpz_swap_(tq, q);
            mpz_clear_(tq);
        }
        if (r)
            mpz_swap_(tr, r);

        mpz_clear_(tr);

        return rn != 0;
    }
}


__device__ void mpz_cdiv_q_r_(mpz_t_ q, mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(q, r, n, d, GMP_DIV_CEIL);
}

__device__ void mpz_fdiv_q_r_(mpz_t_ q, mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(q, r, n, d, GMP_DIV_FLOOR);
}

__device__ void mpz_t_div_q_r_(mpz_t_ q, mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(q, r, n, d, GMP_DIV_TRUNC);
}

__device__ void mpz_cdiv_q_(mpz_t_ q, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(q, NULL, n, d, GMP_DIV_CEIL);
}

__device__ void mpz_fdiv_q_(mpz_t_ q, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(q, NULL, n, d, GMP_DIV_FLOOR);
}

__device__ void mpz_t_div_q_(mpz_t_ q, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(q, NULL, n, d, GMP_DIV_TRUNC);
}

__device__ void mpz_cdiv_r_(mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(NULL, r, n, d, GMP_DIV_CEIL);
}

__device__ void mpz_fdiv_r_(mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(NULL, r, n, d, GMP_DIV_FLOOR);
}

__device__ void mpz_t_div_r_(mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    mpz_div_qr(NULL, r, n, d, GMP_DIV_TRUNC);
}

__device__ void mpz_mod_(mpz_t_ r, const mpz_t_ n, const mpz_t_ d)
{
    if (d->_mp_size >= 0)
        mpz_div_qr(NULL, r, n, d, GMP_DIV_FLOOR);
    else
        mpz_div_qr(NULL, r, n, d, GMP_DIV_CEIL);
}


__device__ static void mpz_div_q_2exp(mpz_t_ q, const mpz_t_ u, mp_bitcnt_t_ bit_index,
		enum mpz_div_round_mode mode)
{
    mp_size_t_ un, qn;
    mp_size_t_ limb_cnt;
    mp_ptr_ qp;
    mp_limb_t_ adjust;

    un = u->_mp_size;
    if (un == 0)
    {
        q->_mp_size = 0;
        return;
    }
    limb_cnt = bit_index / GMP_LIMB_BITS;
    qn = GMP_ABS(un) - limb_cnt;
    bit_index %= GMP_LIMB_BITS;

    if (mode == ((un > 0) ? GMP_DIV_CEIL : GMP_DIV_FLOOR)) /* un != 0 here. */
      /* Note: Below, the final indexing at limb_cnt is valid because at
        that point we have qn > 0. */
        adjust = (qn <= 0
            || !mpn_zero_p(u->_mp_d, limb_cnt)
            || (u->_mp_d[limb_cnt]
          & (((mp_limb_t_) 1 << bit_index) - 1)));
    else
        adjust = 0;

    if (qn <= 0)
        qn = 0;

    else
    {
        qp = (mp_ptr_)MPZ_REALLOC(q, qn);

        if (bit_index != 0)
        {
            mpn_rshift_(qp, u->_mp_d + limb_cnt, qn, bit_index);
            qn -= qp[qn - 1] == 0;
        }
        else
            mpn_copyi_(qp, u->_mp_d + limb_cnt, qn);
    }

    q->_mp_size = qn;

    mpz_add__ui_(q, q, adjust);
    if (un < 0)
        mpz_neg_(q, q);
}

__device__ static void mpz_div_r_2exp(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ bit_index,
		enum mpz_div_round_mode mode)
{
    mp_size_t_ us, un, rn;
    mp_ptr_ rp;
    mp_limb_t_ mask;

    us = u->_mp_size;
    if (us == 0 || bit_index == 0)
    {
        r->_mp_size = 0;
        return;
    }
    rn = (bit_index + GMP_LIMB_BITS - 1) / GMP_LIMB_BITS;
    // assert (rn > 0);

    rp = (mp_ptr_)MPZ_REALLOC(r, rn);
    un = GMP_ABS(us);

    mask = GMP_LIMB_MAX >> (rn * GMP_LIMB_BITS - bit_index);

    if (rn > un)
    {
        /* Quotient (with truncation) is zero, and remainder is
           non-zero */
        if (mode == ((us > 0) ? GMP_DIV_CEIL : GMP_DIV_FLOOR)) /* us != 0 here. */
        {
            /* Have to negate and sign extend. */
            mp_size_t_ i;
            mp_limb_t_ cy;

            for (cy = 1, i = 0; i < un; i++)
            {
                mp_limb_t_ s = ~u->_mp_d[i] + cy;
                cy = s < cy;
                rp[i] = s;
            }
            // assert (cy == 0);
            for (; i < rn - 1; i++)
                rp[i] = GMP_LIMB_MAX;

            rp[rn-1] = mask;
            us = -us;
       } 
       else
       {
          /* Just copy */
            if (r != u)
                mpn_copyi_(rp, u->_mp_d, un);

            rn = un;
       }
    }
    else
    {
        if (r != u)
            mpn_copyi_(rp, u->_mp_d, rn - 1);

        rp[rn-1] = u->_mp_d[rn-1] & mask;

        if (mode == ((us > 0) ? GMP_DIV_CEIL : GMP_DIV_FLOOR)) /* us != 0 here. */
        {
            /* If r != 0, compute 2^{bit_count} - r. */
            mp_size_t_ i;

            for (i = 0; i < rn && rp[i] == 0; i++);

            if (i < rn)
            {
                /* r > 0, need to flip sign. */
                rp[i] = ~rp[i] + 1;
                for (i++; i < rn; i++)
                    rp[i] = ~rp[i];

                rp[rn-1] &= mask;

                /* us is not used for anything else, so we can modify it
            here to indicate flipped sign. */
                us = -us;
            }
        }
    }
    rn = mpn_normalized_size(rp, rn);
    r->_mp_size = us < 0 ? -rn : rn;
}

__device__ void mpz_cdiv_q__2exp_(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ cnt)
{
    mpz_div_q_2exp(r, u, cnt, GMP_DIV_CEIL);
}

__device__ void mpz_fdiv_q__2exp_(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ cnt)
{
    mpz_div_q_2exp(r, u, cnt, GMP_DIV_FLOOR);
}

__device__ void mpz_t_div_q__2exp_(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ cnt)
{
    mpz_div_q_2exp(r, u, cnt, GMP_DIV_TRUNC);
}

__device__ void mpz_cdiv_r__2exp_(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ cnt)
{
    mpz_div_r_2exp(r, u, cnt, GMP_DIV_CEIL);
}

__device__ void mpz_fdiv_r__2exp_ (mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ cnt)
{
    mpz_div_r_2exp(r, u, cnt, GMP_DIV_FLOOR);
}

__device__ void mpz_t_div_r__2exp_(mpz_t_ r, const mpz_t_ u, mp_bitcnt_t_ cnt)
{
    mpz_div_r_2exp(r, u, cnt, GMP_DIV_TRUNC);
}


__device__ void mpz_divexact_(mpz_t_ q, const mpz_t_ n, const mpz_t_ d)
{
    gmp_assert_nocarry(mpz_div_qr(q, NULL, n, d, GMP_DIV_TRUNC));
}

__device__ int mpz_divisible_p_(const mpz_t_ n, const mpz_t_ d)
{
    return mpz_div_qr(NULL, NULL, n, d, GMP_DIV_TRUNC) == 0;
}



__device__ static unsigned long mpz_div_qr_ui(mpz_t_ q, mpz_t_ r,
	       const mpz_t_ n, unsigned long d, enum mpz_div_round_mode mode)
{
    mp_size_t_ ns, qn;
    mp_ptr_ qp;
    mp_limb_t_ rl;
    mp_size_t_ rs;

    ns = n->_mp_size;
    if (ns == 0)
    {
        if (q)
            q->_mp_size = 0;
        if (r)
            r->_mp_size = 0;
        return 0;
    }

    qn = GMP_ABS(ns);
    if (q)
        qp = (mp_ptr_)MPZ_REALLOC(q, qn);
    else
        qp = NULL;

    rl = mpn_div_qr_1(qp, n->_mp_d, qn, d);
    // assert (rl < d);

    rs = rl > 0;
    rs = (ns < 0) ? -rs : rs;

    if (rl > 0 && ( (mode == GMP_DIV_FLOOR && ns < 0)
        || (mode == GMP_DIV_CEIL && ns >= 0)))
    {
        if (q)
            gmp_assert_nocarry(mpn_add__1_(qp, qp, qn, 1));
        rl = d - rl;
        rs = -rs;
    }

    if (r)
    {
        r->_mp_d[0] = rl;
        r->_mp_size = rs;
    }
    if (q)
    {
        qn -= (qp[qn-1] == 0);
        // assert (qn == 0 || qp[qn-1] > 0);

        q->_mp_size = (ns < 0) ? - qn : qn;
    }

    return rl;
}

__device__ unsigned long mpz_cdiv_q_r__ui_(mpz_t_ q, mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(q, r, n, d, GMP_DIV_CEIL);
}

__device__ unsigned long mpz_fdiv_q_r__ui_(mpz_t_ q, mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(q, r, n, d, GMP_DIV_FLOOR);
}

__device__ unsigned long mpz_t_div_q_r__ui_(mpz_t_ q, mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(q, r, n, d, GMP_DIV_TRUNC);
}

__device__ unsigned long mpz_cdiv_q__ui_(mpz_t_ q, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(q, NULL, n, d, GMP_DIV_CEIL);
}

__device__ unsigned long mpz_fdiv_q__ui_(mpz_t_ q, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(q, NULL, n, d, GMP_DIV_FLOOR);
}

__device__ unsigned long  mpz_t_div_q__ui_(mpz_t_ q, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(q, NULL, n, d, GMP_DIV_TRUNC);
}

__device__ unsigned long mpz_cdiv_r__ui_(mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, r, n, d, GMP_DIV_CEIL);
}

__device__ unsigned long mpz_fdiv_r__ui_ (mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, r, n, d, GMP_DIV_FLOOR);
}

__device__ unsigned long mpz_t_div_r__ui_ (mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, r, n, d, GMP_DIV_TRUNC);
}

__device__ unsigned long mpz_cdiv_ui_(const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, NULL, n, d, GMP_DIV_CEIL);
}

__device__ unsigned long mpz_fdiv_ui_(const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, NULL, n, d, GMP_DIV_FLOOR);
}

__device__ unsigned long mpz_t_div_ui_(const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, NULL, n, d, GMP_DIV_TRUNC);
}

__device__ unsigned long mpz_mod__ui_(mpz_t_ r, const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, r, n, d, GMP_DIV_FLOOR);
}


__device__ void mpz_divexact__ui_(mpz_t_ q, const mpz_t_ n, unsigned long d)
{
    gmp_assert_nocarry(mpz_div_qr_ui(q, NULL, n, d, GMP_DIV_TRUNC));
}

__device__ int mpz_divisible_ui_p_(const mpz_t_ n, unsigned long d)
{
    return mpz_div_qr_ui(NULL, NULL, n, d, GMP_DIV_TRUNC) == 0;
}

/* GCD */
__device__ static mp_limb_t_ mpn_gcd_11(mp_limb_t_ u, mp_limb_t_ v)
{
    unsigned shift;

    // assert ( (u | v) > 0);

    if (u == 0)
        return v;
    else if (v == 0)
        return u;

    gmp_ctz(shift, u | v);

    u >>= shift;
    v >>= shift;

    if ((u & 1) == 0)
        mp_limb_t__SWAP(u, v);

    while ( (v & 1) == 0)
        v >>= 1;

    while (u != v)
    {
        if (u > v)
        {
            u -= v;
            do
                u >>= 1;
            while ( (u & 1) == 0);
        }
        else
        {
            v -= u;
            do
                v >>= 1;
            while ((v & 1) == 0);
        }
    }
    return u << shift;
}


__device__ unsigned long mpz_gcd__ui_(mpz_t_ g, const mpz_t_ u, unsigned long v)
{
    mp_size_t_ un;

    if (v == 0)
    {
        if (g)
            mpz_abs_(g, u);
    }
    else
    {
        un = GMP_ABS(u->_mp_size);
        if (un != 0)
            v = mpn_gcd_11(mpn_div_qr_1(NULL, u->_mp_d, un, v), v);

        if (g)
            mpz_set__ui_(g, v);
    }

    return v;
}

__device__ static mp_bitcnt_t_ mpz_make_odd(mpz_t_ r, const mpz_t_ u)
{
    mp_size_t_ un, rn, i;
    mp_ptr_ rp;
    unsigned shift;

    un = GMP_ABS(u->_mp_size);
    // assert (un > 0);

    for (i = 0; u->_mp_d[i] == 0; i++);

    gmp_ctz(shift, u->_mp_d[i]);

    rn = un - i;
    rp = (mp_ptr_)MPZ_REALLOC(r, rn);
    if (shift > 0)
    {
        mpn_rshift_(rp, u->_mp_d + i, rn, shift);
        rn -= (rp[rn-1] == 0);
    }
    else
        mpn_copyi_(rp, u->_mp_d + i, rn);

    r->_mp_size = rn;
    return i * GMP_LIMB_BITS + shift;
}

__device__ void mpz_gcd_(mpz_t_ g, const mpz_t_ u, const mpz_t_ v)
{
    mpz_t_ tu, tv;
    mp_bitcnt_t_ uz, vz, gz;

    if (u->_mp_size == 0)
    {
        mpz_abs_(g, v);
        return;
    }
    if (v->_mp_size == 0)
    {
        mpz_abs_(g, u);
        return;
    }

    mpz_init_(tu, 5);
    mpz_init_(tv, 5);

    uz = mpz_make_odd(tu, u);
    vz = mpz_make_odd(tv, v);
    gz = GMP_MIN(uz, vz);

    if (tu->_mp_size < tv->_mp_size)
        mpz_swap_(tu, tv);

    mpz_t_div_r_(tu, tu, tv);
    if (tu->_mp_size == 0)
    {
        mpz_swap_(g, tv);
    }
    else
        for (;;)
        {
            int c;

            mpz_make_odd(tu, tu);
            c = mpz_cmp_(tu, tv);
            if (c == 0)
            {
                mpz_swap_(g, tu);
                break;
            }
            if (c < 0)
                mpz_swap_(tu, tv);

            if (tv->_mp_size == 1)
            {
                mp_limb_t_ vl = tv->_mp_d[0];
                mp_limb_t_ ul = mpz_t_div_ui_(tu, vl);
                mpz_set__ui_(g, mpn_gcd_11(ul, vl));
                break;
            }
            mpz_sub_(tu, tu, tv);
        }

    mpz_clear_(tu);
    mpz_clear_(tv);
    mpz_mul__2exp_(g, g, gz);
}


__device__ void mpz_gcd_ext(mpz_t_ g, mpz_t_ s, mpz_t_ t, const mpz_t_ u, const mpz_t_ v)
{
    mpz_t_ tu, tv, s0, s1, t0, t1;
    mp_bitcnt_t_ uz, vz, gz;
    mp_bitcnt_t_ power;

    if (u->_mp_size == 0)
    {
        /* g = 0 u + sgn(v) v */
        signed long sign = mpz_sgn_(v);
        mpz_abs_(g, v);
        if (s)
            mpz_set__ui_(s, 0);
        if (t)
            mpz_set__si_(t, sign);
        return;
    }

    if (v->_mp_size == 0)
    {
        /* g = sgn(u) u + 0 v */
        signed long sign = mpz_sgn_(u);
        mpz_abs_(g, u);
        if (s)
            mpz_set__si_(s, sign);
        if (t)
            mpz_set__ui_(t, 0);
        return;
    }

    mpz_init_(tu, 5);
    mpz_init_(tv, 5);
    mpz_init_(s0, 5);
    mpz_init_(s1, 5);
    mpz_init_(t0, 5);
    mpz_init_(t1, 5);

    uz = mpz_make_odd(tu, u);
    vz = mpz_make_odd(tv, v);
    gz = GMP_MIN(uz, vz);

    uz -= gz;
    vz -= gz;

    /* Cofactors corresponding to odd gcd. gz handled later. */
    if (tu->_mp_size < tv->_mp_size)
    {
        mpz_swap_(tu, tv);
        mpz_srcptr__SWAP(u, v);
        mpz_ptr__SWAP(s, t);
        mp_bitcnt_t__SWAP(uz, vz);
    }

    /* Maintain
    *
    * u = t0 tu + t1 tv
    * v = s0 tu + s1 tv
    *
    * where u and v denote the inputs with common factors of two
    * eliminated, and det (s0, t0; s1, t1) = 2^p. Then
    *
    * 2^p tu =  s1 u - t1 v
    * 2^p tv = -s0 u + t0 v
    */

    /* After initial division, tu = q tv + tu', we have
    *
    * u = 2^uz (tu' + q tv)
    * v = 2^vz tv
    *
    * or
    *
    * t0 = 2^uz, t1 = 2^uz q
    * s0 = 0,    s1 = 2^vz
    */

    mpz_set_bit_(t0, uz);
    mpz_t_div_q_r_(t1, tu, tu, tv);
    mpz_mul__2exp_(t1, t1, uz);

    mpz_set_bit_(s1, vz);
    power = uz + vz;

    if (tu->_mp_size > 0)
    {
        mp_bitcnt_t_ shift;
        shift = mpz_make_odd(tu, tu);
        mpz_mul__2exp_(t0, t0, shift);
        mpz_mul__2exp_(s0, s0, shift);
        power += shift;

        for (;;)
        {
            int c;
            c = mpz_cmp_(tu, tv);
            if (c == 0)
                break;

            if (c < 0)
            {
                /* tv = tv' + tu
                *
                * u = t0 tu + t1 (tv' + tu) = (t0 + t1) tu + t1 tv'
                * v = s0 tu + s1 (tv' + tu) = (s0 + s1) tu + s1 tv' */

                mpz_sub_(tv, tv, tu);
                mpz_add_(t0, t0, t1);
                mpz_add_(s0, s0, s1);

                shift = mpz_make_odd(tv, tv);
                mpz_mul__2exp_(t1, t1, shift);
                mpz_mul__2exp_(s1, s1, shift);
            }
            else
            {
                mpz_sub_(tu, tu, tv);
                mpz_add_(t1, t0, t1);
                mpz_add_(s1, s0, s1);

                shift = mpz_make_odd(tu, tu);
                mpz_mul__2exp_(t0, t0, shift);
                mpz_mul__2exp_(s0, s0, shift);
            }
            power += shift;
        }
      }

    /* Now tv = odd part of gcd, and -s0 and t0 are corresponding
      cofactors. */

    mpz_mul__2exp_(tv, tv, gz);
    mpz_neg_(s0, s0);

    /* 2^p g = s0 u + t0 v. Eliminate one factor of two at a time. To
      adjust cofactors, we need u / g and v / g */

    mpz_divexact_(s1, v, tv);
    mpz_abs_(s1, s1);
    mpz_divexact_(t1, u, tv);
    mpz_abs_(t1, t1);

    while (power-- > 0)
    {
        /* s0 u + t0 v = (s0 - v/g) u - (t0 + u/g) v */
        if (mpz_odd_p_(s0) || mpz_odd_p_(t0))
        {
            mpz_sub_(s0, s0, s1);
            mpz_add_(t0, t0, t1);
        }
        mpz_divexact__ui_(s0, s0, 2);
        mpz_divexact__ui_(t0, t0, 2);
    }

    /* Arrange so that |s| < |u| / 2g */
    mpz_add_(s1, s0, s1);
    if (mpz_cmp_abs_(s0, s1) > 0)
    {
        mpz_swap_(s0, s1);
        mpz_sub_(t0, t0, t1);
     }
    if (u->_mp_size < 0)
        mpz_neg_(s0, s0);
    if (v->_mp_size < 0)
        mpz_neg_(t0, t0);

    mpz_swap_(g, tv);
    if (s)
        mpz_swap_(s, s0);
    if (t)
        mpz_swap_(t, t0);

    mpz_clear_(tu);
    mpz_clear_(tv);
    mpz_clear_(s0);
    mpz_clear_(s1);
    mpz_clear_(t0);
    mpz_clear_(t1);
}


__device__ int mpz_t_stbit_(const mpz_t_ d, mp_bitcnt_t_ bit_index)
{
    mp_size_t_ limb_index;
    unsigned shift;
    mp_size_t_ ds;
    mp_size_t_ dn;
    mp_limb_t_ w;
    int bit;

    ds = d->_mp_size;
    dn = GMP_ABS(ds);
    limb_index = bit_index / GMP_LIMB_BITS;
    if (limb_index >= dn)
        return ds < 0;

    shift = bit_index % GMP_LIMB_BITS;
    w = d->_mp_d[limb_index];
    bit = (w >> shift) & 1;

    if (ds < 0)
    {
        /* d < 0. Check if any of the bits below is set: If so, our bit
    must be complemented. */
        if (shift > 0 && (w << (GMP_LIMB_BITS - shift)) > 0)
            return bit ^ 1;
        while (limb_index-- > 0)
            if (d->_mp_d[limb_index] > 0)
                return bit ^ 1;
    }
    return bit;
}

__device__ static void mpz_abs__add_bit(mpz_t_ d, mp_bitcnt_t_ bit_index)
{
    mp_size_t_ dn, limb_index;
    mp_limb_t_ bit;
    mp_ptr_ dp;

    dn = GMP_ABS(d->_mp_size);

    limb_index = bit_index / GMP_LIMB_BITS;
    bit = (mp_limb_t_)1 << (bit_index % GMP_LIMB_BITS);

    if (limb_index >= dn)
    {
        mp_size_t_ i;
        /* The bit should be set outside of the end of the number.
    We have to increase the size of the number. */
        dp = (mp_ptr_)MPZ_REALLOC(d, limb_index + 1);

        dp[limb_index] = bit;
        for (i = dn; i < limb_index; i++)
            dp[i] = 0;
        dn = limb_index + 1;
    }
    else
    {
        mp_limb_t_ cy;

        dp = d->_mp_d;

        cy = mpn_add__1_(dp + limb_index, dp + limb_index, dn - limb_index, bit);
        if (cy > 0)
        {
            dp = (mp_ptr_)MPZ_REALLOC(d, dn + 1);
            dp[dn++] = cy;
        }
    }

    d->_mp_size = (d->_mp_size < 0) ? - dn : dn;
}

__device__ static void mpz_abs__sub_bit(mpz_t_ d, mp_bitcnt_t_ bit_index)
{
    mp_size_t_ dn, limb_index;
    mp_ptr_ dp;
    mp_limb_t_ bit;

    dn = GMP_ABS(d->_mp_size);
    dp = d->_mp_d;

    limb_index = bit_index / GMP_LIMB_BITS;
    bit = (mp_limb_t_) 1 << (bit_index % GMP_LIMB_BITS);

    assert(limb_index < dn);

    gmp_assert_nocarry(mpn_sub__1_(dp + limb_index, dp + limb_index,
          dn - limb_index, bit));
    dn -= (dp[dn-1] == 0);
    d->_mp_size = (d->_mp_size < 0) ? - dn : dn;
}

__device__ void mpz_set_bit_(mpz_t_ d, mp_bitcnt_t_ bit_index)
{
  if (!mpz_t_stbit_(d, bit_index))
  {
      if (d->_mp_size >= 0)
	        mpz_abs__add_bit(d, bit_index);
      else
	        mpz_abs__sub_bit(d, bit_index);
  }
}


__device__ size_t mpz_size_inbase_(const mpz_t_ u, int base)
{
    mp_size_t_ un;
    mp_srcptr_ up;
    mp_ptr_ tp;
    mp_bitcnt_t_ bits;
    struct gmp_div_inverse bi;
    size_t ndigits;

    // assert (base >= 2);
    // assert (base <= 36);

    un = GMP_ABS(u->_mp_size);
    if (un == 0)
        return 1;

    up = u->_mp_d;

    bits = (un - 1) * GMP_LIMB_BITS + mpn_limb_size_in_base_2(up[un-1]);
    switch (base)
    {
      case 2:
          return bits;
      case 4:
          return (bits + 1) / 2;
      case 8:
          return (bits + 2) / 3;
      case 16:
          return (bits + 3) / 4;
      case 32:
          return (bits + 4) / 5;
        /* FIXME: Do something more clever for the common case of base
    10. */
    }

    tp = gmp_xalloc_limbs(un);
    mpn_copyi_(tp, up, un);
    mpn_div_qr_1_invert(&bi, base);

    for (ndigits = 0; un > 0; ndigits++)
    {
        mpn_div_qr_1_preinv(tp, tp, un, &bi);
        un -= (tp[un-1] == 0);
    }
    gmp_free(tp);
    return ndigits;
}



__device__ char * mpz_get_str_(char *sp, int base, const mpz_t_ u)
{
    unsigned bits;
    const char *digits;
    mp_size_t_ un;
    size_t i, sn;

    if (base >= 0)
    {
        digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    }
    else
    {
        base = -base;
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    }
    if (base <= 1)
        base = 10;
    if (base > 36)
        return NULL;

    sn = 1 + mpz_size_inbase_(u, base);
    if (!sp)
        sp = (char *)gmp_xalloc(1 + sn);

    un = GMP_ABS(u->_mp_size);

    if (un == 0)
    {
        sp[0] = '0';
        sp[1] = '\0';
        return sp;
    }

    i = 0;

    if (u->_mp_size < 0)
        sp[i++] = '-';

    bits = mpn_base_power_of_two_p(base);

    if (bits)
        /* Not modified in this case. */
        sn = i + mpn_get_str__bits((unsigned char *) sp + i, bits, u->_mp_d, un);
    else
    {
        struct mpn_base_info info;
        mp_ptr_ tp;

        mpn_get_base_info(&info, base);
        tp = gmp_xalloc_limbs(un);
        mpn_copyi_(tp, u->_mp_d, un);

        sn = i + mpn_get_str__other((unsigned char *) sp + i, base, &info, tp, un);
        gmp_free(tp);
    }

    for (; i < sn; i++)
        sp[i] = digits[(unsigned char) sp[i]];

    sp[sn] = '\0';
    return sp;
}

__device__ int mpz_set__str_(mpz_t_ r, const char *sp, int base)
{
    unsigned bits;
    mp_size_t_ rn, alloc;
    mp_ptr_ rp;
    size_t sn;
    size_t dn;
    int sign;
    unsigned char *dp;

    // assert (base == 0 || (base >= 2 && base <= 36));

    while (disspace((unsigned char) *sp))
        sp++;

    if (*sp == '-')
    {
        sign = 1;
        sp++;
    }
    else
        sign = 0;

    if (base == 0)
    {
        if (*sp == '0')
        {
            sp++;
            if (*sp == 'x' || *sp == 'X')
            {
                base = 16;
                sp++;
            }
            else if (*sp == 'b' || *sp == 'B')
            {
                base = 2;
                sp++;
            }
            else
                base = 8;
        }
        else
            base = 10;
    }

    sn = dstrlen(sp);
    dp = (unsigned char *)gmp_xalloc(sn + (sn == 0));

    for (dn = 0; *sp; sp++)
    {
        unsigned digit;

        if (disspace((unsigned char) *sp))
            continue;
        if (*sp >= '0' && *sp <= '9')
            digit = *sp - '0';
        else if (*sp >= 'a' && *sp <= 'z')
            digit = *sp - 'a' + 10;
        else if (*sp >= 'A' && *sp <= 'Z')
            digit = *sp - 'A' + 10;
        else
            digit = base; /* fail */

        if (digit >= base)
        {
            gmp_free(dp);
            r->_mp_size = 0;
            return -1;
        }

        dp[dn++] = digit;
    }

    bits = mpn_base_power_of_two_p(base);

    if (bits > 0)
    {
        alloc = (sn * bits + GMP_LIMB_BITS - 1) / GMP_LIMB_BITS;
        rp = (mp_ptr_)MPZ_REALLOC(r, alloc);
        rn = mpn_set_str__bits(rp, dp, dn, bits);
    }
    else
    {
        struct mpn_base_info info;
        mpn_get_base_info(&info, base);
        alloc = (sn + info.exp - 1) / info.exp;
        rp = (mp_ptr_)MPZ_REALLOC(r, alloc);
        rn = mpn_set_str__other(rp, dp, dn, base, &info);
    }
    // assert (rn <= alloc);
    gmp_free(dp);

    r->_mp_size = sign ? - rn : rn;

    return 0;
}

__device__ int mpz_init__set__str_ (mpz_t_ r, const char *sp, int base)
{
    mpz_init_(r);
    return mpz_set__str_(r, sp, base);
}