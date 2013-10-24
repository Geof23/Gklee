//===-- cudaArithIntrinsic.c ----------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <cuda/math_constants.h>

float __sinf(float);
float __cosf(float);
float sinf(float);
float cosf(float);
double sin(double);
double cos(double);
float __fmaf_rn(float, float, float);
float __log2f(float);
double exp(double);
float exp2f(float);
double exp2(double);
float log1p(float);
double log2(double);
float rsqrtf(float);
float sqrt(float);
float ldexpf(float, int);
int __isnan(double);
int __isinf(double);
int __float_as_int(float);
float __int_as_float(int);
int __double2hiint(double x);
long long int __double_as_longlong(double x);
double __longlong_as_double(long long int x);
float fabsf(float x);
double fabs(double x);
int __finitef(float);
int __isinff(float);
float truncf(float);
float rcpf(float);
int __signbit(double);

void __sincosf(float x, float *sptr, float *cptr) {
  *sptr = __sinf(x);
  *cptr = __cosf(x);
}

void sincosf(float x, float *sptr, float *cptr) {
  *sptr = sinf(x);
  *cptr = cosf(x);
}

void sincos(double x, double *sptr, double *cptr) {
  *sptr = sin(x);
  *cptr = cos(x);
}

float frexpf(float a, int *b) {
  float fa = fabsf(a);
  unsigned int expo;
  unsigned int denorm; 

  if (fa < CUDART_TWO_TO_M126_F) {
    a *= CUDART_TWO_TO_24_F;
    denorm = 24;
  } else {
    denorm = 0;
  }
  expo = ((__float_as_int(a) >> 23) & 0xff);
  if ((fa == 0.0f) || (expo == 0xff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 126;
    a = __int_as_float(((__float_as_int(a) & 0x807fffff) | 0x3f000000));
  }
  *b = expo;
  return a;
}

double frexp(double a, int *b) {
  double fa = fabs(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < CUDART_TWO_TO_M1022) {
    a *= CUDART_TWO_TO_54;
    denorm = 54;
  } else {
    denorm = 0;
  }
  expo = (__double2hiint(a) >> 20) & 0x7ff;
  if ((fa == 0.0) || (expo == 0x7ff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 1022;
    a = __longlong_as_double((__double_as_longlong(a) & 0x800fffffffffffffULL)|
                              0x3fe0000000000000ULL);
  }
  *b = expo;
  return a;
}

float __internal_fmad(float a, float b, float c) {
  return __fmaf_rn (a, b, c);
}

float erfinvf(float a) {
  float s, t, r;

#if __CUDA_ARCH__ >= 200
  s = __fmaf_rn (a, -a, 1.0f);
#else
  s = 1.0f + a;
  t = 1.0f - a;
  s = s * t;
#endif
  t = - __log2f (s);
  if (t > 8.2f) {
    t = rsqrtf (t);
    r =                        -5.8991436871681446E-001f;
    r = __internal_fmad (r, t, -6.6300422537735360E-001f);
    r = __internal_fmad (r, t,  1.5970110948817704E+000f);
    r = __internal_fmad (r, t, -6.7521557026467416E-001f);
    r = __internal_fmad (r, t, -9.5224791160399669E-002f);
    r = __internal_fmad (r, t,  8.3535343797791939E-001f);
    t = 1.0f / t;
    t = r * t;
    if (a < 0.0f) t = -t;
    return t;
  } else {
    r =                        -2.5172707606685652E-010f;
    r = __internal_fmad (r, t,  9.4274289432374619E-009f);
    r = __internal_fmad (r, t, -1.2054753059594516E-007f);
    r = __internal_fmad (r, t,  2.1697004698667657E-007f);
    r = __internal_fmad (r, t,  8.0621488510822390E-006f);
    r = __internal_fmad (r, t, -3.1675491789646913E-005f);
    r = __internal_fmad (r, t, -7.7436312951712784E-004f);
    r = __internal_fmad (r, t,  5.5465877941375773E-003f);
    r = __internal_fmad (r, t,  1.6082022430967846E-001f);
    r = __internal_fmad (r, t,  8.8622690039402130E-001f);
    return r * a;
  }
}

double erfinv(double a) {
  double p, q, t, fa;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  fa = fabs(a);
  if (fa >= 1.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;                    /* INDEFINITE */
    if (fa == 1.0) {
      t = a * exp(1000.0);        /* Infinity */
    }
  } else if (fa >= 0.9375) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
     */
    t = log1p(-fa);
    t = 1.0 / sqrt(-t);
    p =         2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q =     t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
    if (a < 0.0) t = -t;
  } else if (fa >= 0.75) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 39
    */
    t = a * a - .87890625;
    p =         .21489185007307062000e+0;
    p = p * t - .64200071507209448655e+1;
    p = p * t + .29631331505876308123e+2;
    p = p * t - .47644367129787181803e+2;
    p = p * t + .34810057749357500873e+2;
    p = p * t - .12954198980646771502e+2;
    p = p * t + .25349389220714893917e+1;
    p = p * t - .24758242362823355486e+0;
    p = p * t + .94897362808681080020e-2;
    q =     t - .12831383833953226499e+2;
    q = q * t + .41409991778428888716e+2;
    q = q * t - .53715373448862143349e+2;
    q = q * t + .33880176779595142685e+2;
    q = q * t - .11315360624238054876e+2;
    q = q * t + .20369295047216351160e+1;
    q = q * t - .18611650627372178511e+0;
    q = q * t + .67544512778850945940e-2;
    p = p / q;
    t = a * p;
  } else {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 18
    */
    t = a * a - .5625;
    p =       - .23886240104308755900e+2;
    p = p * t + .45560204272689128170e+3;
    p = p * t - .22977467176607144887e+4;
    p = p * t + .46631433533434331287e+4;
    p = p * t - .43799652308386926161e+4;
    p = p * t + .19007153590528134753e+4;
    p = p * t - .30786872642313695280e+3;
    q =     t - .83288327901936570000e+2;
    q = q * t + .92741319160935318800e+3;
    q = q * t - .35088976383877264098e+4;
    q = q * t + .59039348134843665626e+4;
    q = q * t - .48481635430048872102e+4;
    q = q * t + .18997769186453057810e+4;
    q = q * t - .28386514725366621129e+3;
    p = p / q;
    t = a * p;
  }
  return t;
}

float remquof(float a, float b, int* quo) {
  float twoa = 0.0f;
  unsigned int quot = 0;  /* trailing quotient bits */
  unsigned int sign;
  float orig_a = a;
  float orig_b = b;

  /* quo has a value whose sign is the sign of x/y */
  sign = 0 - ((__float_as_int(a) ^ __float_as_int(b)) < 0);
  a = fabsf(a);
  b = fabsf(b);
  if (!((a <= CUDART_INF_F) && (b <= CUDART_INF_F))) {
    *quo = quot;
    return orig_a + orig_b;
  }
  if ((a == CUDART_INF_F) || (b == CUDART_ZERO_F)) {
    *quo = quot;
    return rsqrtf (__int_as_float (0xffc00000));
  } else if (a >= b) {
#if !defined(__CUDA_FTZ)
    /* Need to be able to handle denormals correctly */
    int expoa = (a < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
#else /* !__CUDA_FTZ */
    float scaled_b = __int_as_float ((__float_as_int(b) & 0x007fffff) | 
                                     (__float_as_int(a) & 0x7f800000));
    if (scaled_b > a) {
      scaled_b *= 0.5f;
    }
    /* check wether divisor is a power of two */
    if (a == scaled_b) {
      if (__float_as_int(b) > 0x7e800000) {
          a *= 0.25f;
          b *= 0.25f;
      }
      a = __fdividef(a,b) + 0.5f;
      quot = (a < 8.0f) ? (int)a : 0;
      quot = quot ^ sign;
      quot = quot - sign;
      *quo = quot;
      return __int_as_float(__float_as_int(orig_a) & 0x80000000);
    }    
#endif /* !__CUDA_FTZ */
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5f;
    }
  }
  /* round quotient to nearest even */
#if !defined(__CUDA_FTZ)
  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
    a = __int_as_float(__float_as_int(a) | 0x80000000);
  }
#else /* !__CUDA_FTZ */
  if (a >= CUDART_TWO_TO_M126_F) {
    twoa = 2.0f * a;
    if ((twoa > b) || ((twoa == b) && (quot & 1))) {
      quot++;
      a -= b;
      a = __int_as_float(__float_as_int(a) | 0x80000000);
    }
  } else {
    /* a already got flushed to zero, so use twoa instead */
    if ((twoa > b) || ((twoa == b) && (quot & 1))) {
      quot++;
      a = 0.5f * (twoa - 2.0f * b);
      a = __int_as_float(__float_as_int(a) | 0x80000000);
    }
  }
#endif /* !__CUDA_FTZ */
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  quot = quot & CUDART_REMQUO_MASK_F;
  quot = quot ^ sign;
  quot = quot - sign;
  *quo = quot;
  return a;
}

double remquo(double a, double b, int *quo) {
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;
  int rem1 = 1; /* do FPREM1, a.k.a IEEE remainder */
  int expo_a;
  int expo_b;
  unsigned long long mant_a;
  unsigned long long mant_b;
  unsigned long long mant_c;
  unsigned long long temp;
  int sign_a;
  int sign_b;
  int sign_c;
  int expo_c;
  int expodiff;
  int quot = 0;                 /* initialize quotient */
  int l;
  int iter;

  cvt.d = a;
  mant_a = (cvt.l << 11) | 0x8000000000000000ULL;
  expo_a = (int)((cvt.l >> 52) & 0x7ff) - 1023;
  sign_a = (int)(cvt.l >> 63);

  cvt.d = b;
  mant_b = (cvt.l << 11) | 0x8000000000000000ULL;
  expo_b = (int)((cvt.l >> 52) & 0x7ff) - 1023;
  sign_b = (int)(cvt.l >> 63);

  sign_c = sign_a;  /* remainder has sign of dividend */
  expo_c = expo_a;  /* default */
      
  /* handled NaNs and infinities */
  if (__isnan(a) || __isnan(b)) {
    *quo = quot;
    return a + b;
  }
  if (__isinf(a) || (b == 0.0)) {
    *quo = quot;
    cvt.l = 0xfff8000000000000ULL;
    return cvt.d;
  }
  if ((a == 0.0) || (__isinf(b))) {
    *quo = quot;
    return a;
  }
  /* normalize denormals */
  if (expo_a < -1022) {
    mant_a = mant_a + mant_a;
    while (mant_a < 0x8000000000000000ULL) {
      mant_a = mant_a + mant_a;
      expo_a--;
    }
  } 
  if (expo_b < -1022) {
    mant_b = mant_b + mant_b;
    while (mant_b < 0x8000000000000000ULL) {
      mant_b = mant_b + mant_b;
      expo_b--;
    }
  }
  expodiff = expo_a - expo_b;
  /* clamp iterations if exponent difference negative */
  if (expodiff < 0) {
    iter = -1;
  } else {
    iter = expodiff;
  }
  /* Shift dividend and divisor right by one bit to prevent overflow
     during the division algorithm.
   */
  mant_a = mant_a >> 1;
  mant_b = mant_b >> 1;
  expo_c = expo_a - iter; /* default exponent of result   */

  /* Use binary longhand division (restoring) */
  for (l = 0; l < (iter + 1); l++) {
    mant_a = mant_a - mant_b;
    if (mant_a & 0x8000000000000000ULL) {
      mant_a = mant_a + mant_b;
      quot = quot + quot;
    } else {
      quot = quot + quot + 1;
    }
    mant_a = mant_a + mant_a;
  }

  /* Save current remainder */
  mant_c = mant_a;
  /* If remainder's mantissa is all zeroes, final result is zero. */
  if (mant_c == 0) {
    quot = quot & 7;
    *quo = (sign_a ^ sign_b) ? -quot : quot;
    cvt.l = (unsigned long long int)sign_c << 63;
    return cvt.d;
  }
  /* Normalize result */
  while (!(mant_c & 0x8000000000000000ULL)) {
    mant_c = mant_c + mant_c;
    expo_c--;
  }
  /* For IEEE remainder (quotient rounded to nearest-even we might need to 
     do a final subtraction of the divisor from the remainder.
  */
  if (rem1 && ((expodiff+1) >= 0)) {
    temp = mant_a - mant_b;
    /* round quotient to nearest even */
    if (((temp != 0ULL) && (!(temp & 0x8000000000000000ULL))) ||
        ((temp == 0ULL) && (quot & 1))) {
      mant_a = mant_a >> 1;
      quot++;
      /* Since the divisor is greater than the remainder, the result will
         have opposite sign of the dividend. To avoid a negative mantissa
         when subtracting the divisor from remainder, reverse subtraction
      */
      sign_c = 1 ^ sign_c;
      expo_c = expo_a - iter + 1;
      mant_c = mant_b - mant_a;
      /* normalize result */
      while (!(mant_c & 0x8000000000000000ULL)) {
        mant_c = mant_c + mant_c;
        expo_c--;
      }
    }
  }
  /* package up result */
  if (expo_c >= -1022) { /* normal */
    mant_c = ((mant_c >> 11) + 
              ((((unsigned long long)sign_c) << 63) +
               (((unsigned long long)(expo_c + 1022)) << 52)));
  } else { /* denormal */
    mant_c = ((((unsigned long long)sign_c) << 63) + 
              (mant_c >> (11 - expo_c - 1022)));
  }
  quot = quot & 7; /* mask quotient down to least significant three bits */
  *quo = (sign_a ^ sign_b) ? -quot : quot;
  cvt.l = mant_c;
  return cvt.d;
}

float modff(float a, float *b) {
  float t;
  if (__finitef(a)) {
    t = truncf(a);
    *b = t;
    t = a - t;
    t = __int_as_float(__float_as_int(t) | (__float_as_int(a) & 0x80000000));
    return t;
  } else if (__isinff(a)) {
    *b = a;
    return __int_as_float(__float_as_int(a) & 0x80000000);
  } else {
    *b = a + a;
    return a + a;
  }
}

double modf(double a, double *b) {
  float fb;
  float fa = modff((float)a, &fb);

  *b = (double)fb;

  return (double)fa;
}

float rcbrtf(float a) {                                      
  float s, t;

  s = fabsf(a);                                                                  
  t = exp2f(-CUDART_THIRD_F * __log2f(s));          /* initial approx */ 
  t = __internal_fmad(__internal_fmad(t*t, -s*t, 1.0f), CUDART_THIRD_F*t, t);
  if (__float_as_int(a) < 0) {
    t = -t;
  }                                                                              
  s = a + a;                                                                   
  if (s == a) t = rcpf(a); /* handle special cases */              
  return t;                                                                      
} 

double rcbrt(double a) {
  double s, t;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return 1.0 / a;
  }
  s = fabs(a);
  t = exp2(-CUDART_THIRD * log2(s));                /* initial approximation */
  t = ((t*t) * (-s*t) + 1.0) * (CUDART_THIRD*t) + t;/* refine approximation */
#if defined(__APPLE__)
  if (__signbitd(a))
#else /* __APPLE__ */
  if (__signbit(a))
#endif /* __APPLE__ */
  {
    t = -t;
  }
  return t;
}

