/*
 * TachyonOptiXShaders.cu - OptiX PTX shading and ray intersection routines 
 *
 *  $Id: TachyonOptiXShaders.cu,v 1.35 2021/05/04 04:04:21 johns Exp $
 */
/**
 *  \file TachyonOptiXShaders.cu
 *  \brief Tachyon ray tracing engine core routines compiled to PTX for
 *         runtime JIT to build complete ray tracing pipelines.
 *         Written for NVIDIA OptiX 7 and later.
 */


#include <optix_device.h>
#include <stdint.h>
#include "TachyonOptiXShaders.h"

// Macros related to ray origin epsilon stepping to prevent
// self-intersections with the surface we're leaving
// This is a cheesy way of avoiding self-intersection
// but it ameliorates the problem.
// Since changing the scene epsilon even to large values does not
// always cure the problem, this workaround is still required.
#define TACHYON_USE_RAY_STEP       1
#define TACHYON_TRANS_USE_INCIDENT 1
#define TACHYON_RAY_STEP           N*rtLaunch.scene.epsilon*4.0f
#define TACHYON_RAY_STEP2          ray_direction*rtLaunch.scene.epsilon*4.0f

// reverse traversal of any-hit rays for shadows/AO
#define REVERSE_RAY_STEP       (scene_epsilon*10.0f)
#define REVERSE_RAY_LENGTH     3.0f

// Macros to enable particular ray-geometry intersection variants that
// optimize for speed, or some combination of speed and accuracy
#define TACHYON_USE_SPHERES_HEARNBAKER 1

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif


//
// convert float3 rgb data to uchar4 with alpha channel set to 255.
//
static __device__ __inline__ uchar4 make_color_rgb4u(const float3& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),
                     255u);
}

//
// convert float4 rgba data to uchar4
//
static __device__ __inline__ uchar4 make_color_rgb4u(const float4& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.w)*255.99f));
}


//
// Various random number routines
//   https://en.wikipedia.org/wiki/List_of_random_number_generators
//

#define UINT32_RAND_MAX     4294967296.0f      // max uint32 random value
#define UINT32_RAND_MAX_INV 2.3283064365e-10f  // normalize uint32 RNs

//
// Survey of parallel RNGS suited to GPUs, by L'Ecuyer et al.:
//   Random numbers for parallel computers: Requirements and methods, 
//   with emphasis on GPUs. 
//   Pierre L'Ecuyer, David Munger, Boris Oreshkina, and Richard Simard.
//   Mathematics and Computers in Simulation 135:3-17, 2017.
//   https://doi.org/10.1016/j.matcom.2016.05.005
//
// Counter-based RNGs introduced by Salmon @ D.E. Shaw Research:
//   "Parallel random numbers: as easy as 1, 2, 3", by Salmon et al.,
//    D. E. Shaw Research:  
//   http://doi.org/10.1145/2063384.2063405
//   https://www.thesalmons.org/john/random123/releases/latest/docs/index.html
//   https://en.wikipedia.org/wiki/Counter-based_random_number_generator_(CBRNG)
//


//
// Quick and dirty 32-bit LCG random number generator [Fishman 1990]:
//   A=1099087573 B=0 M=2^32
//   Period: 10^9
// Fastest gun in the west, but fails many tests after 10^6 samples,
// and fails all statistics tests after 10^7 samples.
// It fares better than the Numerical Recipes LCG.  This is the fastest
// power of two rand, and has the best multiplier for 2^32, found by
// brute force[Fishman 1990].  Test results:
//   http://www.iro.umontreal.ca/~lecuyer/myftp/papers/testu01.pdf
//   http://www.shadlen.org/ichbin/random/
//
static __host__ __device__ __inline__ 
uint32_t qnd_rng(uint32_t &idum) {
  idum *= 1099087573;
  return idum; // already 32-bits, no need to mask result
}



//
// Middle Square Weyl Sequence ("msws")
//   This is an improved variant of von Neumann's middle square RNG
//   that uses Weyl sequences to provide a long period.  Claimed as 
//   fastest traditional seeded RNG that passes statistical tests.
//   V5: Bernard Widynski, May 2020.
//   https://arxiv.org/abs/1704.00358
//   
//   Additional notes and commentary:
//     https://en.wikipedia.org/wiki/Middle-square_method
//     https://pthree.org/2018/07/30/middle-square-weyl-sequence-prng/
//
//   Reported to passes both BigCrush and PractRand tests:
//     "An Empirical Study of Non-Cryptographically Secure 
//      Pseudorandom Number Generators," M. Singh, P. Singh and P. Kumar, 
//      2020 International Conference on Computer Science, Engineering 
//      and Applications (ICCSEA), 2020, 
//      http://doi.org/10.1109/ICCSEA49143.2020.9132873
//
static __host__ __device__ __inline__
uint32_t msws_rng(uint64_t &x, uint64_t &w) {
  const uint64_t s = 0xb5ad4eceda1ce2a9;
  x *= x;                // square the value per von Neumann's RNG
  w += s;                // add in Weyl sequence for longer period
  x += w;                // apply to x
  x = (x>>32) | (x<<32); // select "middle square" as per von Neumann's RNG
  return x;              // implied truncation to lower 32-bit result
}



//
// Squares: A Fast Counter-Based RNG
//   This is a counter-based RNG based on John von Neumann's 
//   Middle Square RNG, with the Weyl sequence added to provide a long period.
//   V3: Bernard Widynski, Nov 2020.
//   https://arxiv.org/abs/2004.06278
//
// This RNG claims to outperform all of the original the counter-based RNGs
// in "Parallel random numbers: as easy as 1, 2, 3", 
//   by Salmon et al., http://doi.org/10.1145/2063384.2063405
//   https://en.wikipedia.org/wiki/Counter-based_random_number_generator_(CBRNG)
// That being said, key generation technique is important in this case.
//
#define SQUARES_RNG_KEY1 0x1235d7fcb4dfec21  // a few good keys...
#define SQUARES_RNG_KEY2 0x418627e323f457a1  // a few good keys...
#define SQUARES_RNG_KEY3 0x83fc79d43614975f  // a few good keys...
#define SQUARES_RNG_KEY4 0xc62f73498cb654e3  // a few good keys...

// Template to allow compile-time selection of number of rounds (2, 3, 4).
// Roughly 5 integer ALU operations per round, 4 rounds is standard.
template<unsigned int ROUNDS> static __host__ __device__ __inline__
uint32_t squares_rng(uint64_t counter, uint64_t key) {
  uint64_t x, y, z;
  y = x = counter * key; 
  z = x + key;

  x = x*x + y;                // round 1, middle square, add Weyl seq
  x = (x>>32) | (x<<32);      // round 1, bit rotation

  x = x*x + z;                // round 2, middle square, add Weyl seq
  if (ROUNDS == 2) {
    return x >> 32;           // round 2, upper 32-bits are bit-rotated result
  } else {
    x = (x>>32) | (x<<32);    // round 2, bit rotation

    x = x*x + y;              // round 3, middle square, add Weyl seq
    if (ROUNDS == 3) {
      return x >> 32;         // round 3, upper 32-bits are bit-rotated result
    } else {
      x = (x>>32) | (x<<32);  // round 3, bit rotation

      x = x*x + z;            // round 4, middle square, add Weyl seq
      return x >> 32;         // round 4, upper 32-bits are bit-rotated result
    }
  }
}



//
// TEA, a tiny encryption algorithm.
// D. Wheeler and R. Needham, 2nd Intl. Workshop Fast Software Encryption,
// LNCS, pp. 363-366, 1994.
//
// GPU Random Numbers via the Tiny Encryption Algorithm
// F. Zafar, M. Olano, and A. Curtis.
// HPG '10 Proceedings of the Conference on High Performance Graphics,
// pp. 133-141, 2010.
// https://dl.acm.org/doi/10.5555/1921479.1921500
// 
// Tea has avalanche effect in output from one bit input delta after 6 rounds
//
template<unsigned int ROUNDS> static __host__ __device__ __inline__
unsigned int tea(uint32_t val0, uint32_t val1) {
  uint32_t v0 = val0;
  uint32_t v1 = val1;
  uint32_t s0 = 0;

  for (unsigned int n = 0; n < ROUNDS; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}


//
// Low discrepancy sequences based on the Golden Ratio, described in
// Golden Ratio Sequences for Low-Discrepancy Sampling,
// Colas Schretter and Leif Kobbelt, pp. 95-104, JGT 16(2), 2012.
//
// Other useful online references:
//   http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
//

// compute Nth value in 1-D sequence
static __device__ __inline__
float goldenratioseq1d(int n) {
  const double g = 1.61803398874989484820458683436563;
  const double a1 = 1.0 / g;
  const double seed = 0.5;
  double ngold;
  ngold = (seed + (a1 * n));
  return ngold - trunc(ngold);
}

// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq1d_incr(float &x) {
  const double g = 1.61803398874989484820458683436563;
  const double a1 = 1.0 / g;
  float ngold = x + a1;
  x = ngold - truncf(ngold);
}


// compute Nth point in 2-D sequence
static __device__ __inline__
void goldenratioseq2d(int n, float2 &xy) {
  const double g = 1.32471795724474602596;
  const double a1 = 1.0 / g;
  const double a2 = 1.0 / (g*g);
  const double seed = 0.5;
  double ngold;

  ngold = (seed + (a1 * n));
  xy.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a2 * n));
  xy.y = (float) (ngold - trunc(ngold));
}

// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq2d_incr(float2 &xy) {
  const float g = 1.32471795724474602596;
  const float a1 = 1.0 / g;
  const float a2 = 1.0 / (g*g);
  float ngold;

  ngold = xy.x + a1;
  xy.x = (ngold - trunc(ngold));

  ngold = xy.y + a2;
  xy.y = (ngold - trunc(ngold));
}


// compute Nth point in 3-D sequence
static __device__ __inline__
void goldenratioseq3d(int n, float3 &xyz) {
  const double g = 1.22074408460575947536;
  const double a1 = 1.0 / g;
  const double a2 = 1.0 / (g*g);
  const double a3 = 1.0 / (g*g*g);
  const double seed = 0.5;
  double ngold;

  ngold = (seed + (a1 * n));
  xyz.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a2 * n));
  xyz.y = (float) (ngold - trunc(ngold));

  ngold = (seed + (a3 * n));
  xyz.z = (float) (ngold - trunc(ngold));
}

// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq3d_incr(float3 &xyz) {
  const float g = 1.22074408460575947536;
  const float a1 = 1.0 / g;
  const float a2 = 1.0 / (g*g);
  const float a3 = 1.0 / (g*g*g);
  float ngold;

  ngold = xyz.x + a1;
  xyz.x = (ngold - trunc(ngold));

  ngold = xyz.y + a2;
  xyz.y = (ngold - trunc(ngold));

  ngold = xyz.z + a3;
  xyz.z = (ngold - trunc(ngold));
}


// compute Nth point in 4-D sequence
static __device__ __inline__
void goldenratioseq4d(int n, float2 &xy1, float2 &xy2) {
  const double g = 1.167303978261418740;
  const double a1 = 1.0 / g;
  const double a2 = 1.0 / (g*g);
  const double a3 = 1.0 / (g*g*g);
  const double a4 = 1.0 / (g*g*g*g);
  const double seed = 0.5;
  double ngold;

  ngold = (seed + (a1 * n));
  xy1.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a2 * n));
  xy1.y = (float) (ngold - trunc(ngold));

  ngold = (seed + (a3 * n));
  xy2.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a4 * n));
  xy2.y = (float) (ngold - trunc(ngold));
}

// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq4d_incr(float2 &xy1, float2 &xy2) {
  const double g = 1.167303978261418740;
  const float a1 = 1.0 / g;
  const float a2 = 1.0 / (g*g);
  const float a3 = 1.0 / (g*g*g);
  const float a4 = 1.0 / (g*g*g*g);
  float ngold;

  ngold = xy1.x + a1;
  xy1.x = (ngold - trunc(ngold));

  ngold = xy1.y + a2;
  xy1.y = (ngold - trunc(ngold));

  ngold = xy2.x + a3;
  xy2.x = (ngold - trunc(ngold));

  ngold = xy2.y + a4;
  xy2.y = (ngold - trunc(ngold));
}



//
// stochastic sampling helper routines
//

// Generate an offset to jitter AA samples in the image plane
static __device__ __inline__
void jitter_offset2f(unsigned int &pval, float2 &xy) {
  xy.x = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
  xy.y = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
}


// Generate an offset to jitter DoF samples in the Circle of Confusion
static __device__ __inline__
void jitter_disc2f(unsigned int &pval, float2 &xy, float radius) {
#if 1
  // Since the GPU RT currently uses super cheap/sleazy LCG RNGs,
  // it is best to avoid using sample picking, which can fail if
  // we use a multiply-only RNG and we hit a zero in the PRN sequence.
  // The special functions are slow, but have bounded runtime and
  // minimal branch divergence.
  float   r=(qnd_rng(pval) * UINT32_RAND_MAX_INV);
  float phi=(qnd_rng(pval) * UINT32_RAND_MAX_INV) * 2.0f * M_PIf;
  __sincosf(phi, &xy.x, &xy.y); // fast approximation
  xy *= sqrtf(r) * radius;
#else
  // Pick uniform samples that fall within the disc --
  // this scheme can hang in an endless loop if a poor quality
  // RNG is used and it gets stuck in a short PRN sub-sequence
  do {
    xy.x = 2.0f * (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 1.0f;
    xy.y = 2.0f * (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 1.0f;
  } while ((xy.x*xy.x + xy.y*xy.y) > 1.0f);
  xy *= radius;
#endif
}

// Generate an offset to jitter AA samples in the image plane using
// a low-discrepancy sequence
static __device__ __inline__
void jitter_offset2f_qrn(float2 qrnxy, float2 &xy) {
  xy = qrnxy - make_float2(0.5f, 0.5f);
}

// Generate an offset to jitter DoF samples in the Circle of Confusion,
// using low-discrepancy sequences based on the Golden Ratio
static __device__ __inline__
void jitter_disc2f_qrn(float2 &qrnxy, float2 &xy, float radius) {
  goldenratioseq2d_incr(qrnxy);
  float   r=qrnxy.x;
  float phi=qrnxy.y * 2.0f * M_PIf;
  __sincosf(phi, &xy.x, &xy.y); // fast approximation
  xy *= sqrtf(r) * radius;
}


// Generate a randomly oriented ray
static __device__ __inline__
void jitter_sphere3f(unsigned int &pval, float3 &dir) {
#if 1
  //
  // Use GPU fast/approximate math routines
  //
  /* Archimedes' cylindrical projection scheme       */
  /* generate a point on a unit cylinder and project */
  /* back onto the sphere.  This approach is likely  */
  /* faster for SIMD hardware, despite the use of    */
  /* transcendental functions.                       */
  float u1 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  dir.z = 2.0f * u1 - 1.0f;
  float R = __fsqrt_rn(1.0f - dir.z*dir.z);  // fast approximation
  float u2 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  float phi = 2.0f * M_PIf * u2;
  float sinphi, cosphi;
  __sincosf(phi, &sinphi, &cosphi); // fast approximation
  dir.x = R * cosphi;
  dir.y = R * sinphi;
#elif 1
  /* Archimedes' cylindrical projection scheme       */
  /* generate a point on a unit cylinder and project */
  /* back onto the sphere.  This approach is likely  */
  /* faster for SIMD hardware, despite the use of    */
  /* transcendental functions.                       */
  float u1 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  dir.z = 2.0f * u1 - 1.0f;
  float R = sqrtf(1.0f - dir.z*dir.z);

  float u2 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  float phi = 2.0f * M_PIf * u2;
  float sinphi, cosphi;
  sincosf(phi, &sinphi, &cosphi);
  dir.x = R * cosphi;
  dir.y = R * sinphi;
#else
  /* Marsaglia's uniform sphere sampling scheme           */
  /* In order to correctly sample a sphere, using rays    */
  /* generated randomly within a cube we must throw out   */
  /* direction vectors longer than 1.0, otherwise we'll   */
  /* oversample the corners of the cube relative to       */
  /* a true sphere.                                       */
  float len;
  float3 d;
  do {
    d.x = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
    d.y = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
    d.z = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
    len = dot(d, d);
  } while (len > 0.250f);
  float invlen = rsqrtf(len);

  /* finish normalizing the direction vector */
  dir = d * invlen;
#endif
}


//
// OptiX ray processing programs
//


/// launch parameters in constant memory, filled  by optixLaunch)
extern "C" __constant__ tachyonLaunchParams rtLaunch;

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 ) {
  const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
  void*           ptr = reinterpret_cast<void*>( uptr );
  return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ) {
  const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD() {
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}



//
// Per-ray data
//
struct PerRayData_radiance {
  float3 result;     // final shaded surface color
  float alpha;       // alpha value to back-propagate to framebuffer
  float importance;  // importance of recursive ray tree
  int depth;         // current recursion depth
  int transcnt;      // transmission ray surface count/depth
};

struct PerRayData_shadow {
  float3 attenuation;
};


static int __inline__ __device__ subframe_count() {
//  return (accumCount + progressiveSubframeIndex);
  return rtLaunch.frame.subframe_index; 
}



//
// Device functions for clipping rays by geometric primitives
//

// fade_start: onset of fading
//   fade_end: fully transparent, begin clipping of geometry
__device__ void sphere_fade_and_clip(const float3 &hit_point,
                                     const float3 &cam_pos,
                                     float fade_start, float fade_end,
                                     float &alpha) {
  float camdist = length(hit_point - cam_pos);

  // we can omit the distance test since alpha modulation value is clamped
  // if (1 || camdist < fade_start) {
    float fade_len = fade_start - fade_end;
    alpha *= __saturatef((camdist - fade_start) / fade_len);
  // }
}


__device__ void ray_sphere_clip_interval(float3 ray_origin,  
                                         float3 ray_direction, float3 center,
                                         float rad, float2 &tinterval) {
  float3 V = center - ray_origin;
  float b = dot(V, ray_direction);
  float disc = b*b + rad*rad - dot(V, V);

  // if the discriminant is positive, the ray hits...
  if (disc > 0.0f) {
    disc = sqrtf(disc);
    tinterval.x = b-disc;
    tinterval.y = b+disc;
  } else {
    tinterval.x = -RT_DEFAULT_MAX;
    tinterval.y =  RT_DEFAULT_MAX;
  }
}


__device__ void clip_ray_by_plane(float3 ray_origin,
                                  float3 ray_direction, 
                                  float &tmin, float &tmax,
                                  const float4 plane) {
  float3 n = make_float3(plane);
  float dt = dot(ray_direction, n);
  float t = (-plane.w - dot(n, ray_origin))/dt;
  if(t > tmin && t < tmax) {
    if (dt <= 0) {
      tmax = t;
    } else {
      tmin = t;
    }
  } else {
    // ray interval lies completely on one side of the plane.  Test one point.
    float3 p = ray_origin + tmin * ray_direction;
    if (dot(make_float4(p.x, p.y, p.z, 1.0f), plane) < 0) {
      tmin = tmax = RT_DEFAULT_MAX; // cull geometry
    }
  }
}



//
// Default Tachyon exception handling program
//   Any OptiX state on the stack will be gone post-exception, so if we 
//   want to store anything it would need to be written to a global 
//   memory allocation
//
extern "C" __global__ void __exception__all() {
  const int code = optixGetExceptionCode();
  const uint3 launch_index = optixGetLaunchIndex();

  switch (code) {
    case OPTIX_EXCEPTION_CODE_STACK_OVERFLOW:
      printf("Stack overflow at launch index (%u,%u):\n",
            launch_index.x, launch_index.y );
      break;

    case OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED:
      printf("Max trace depth exceeded at launch index (%u,%u):\n",
            launch_index.x, launch_index.y );
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED:
      printf("Max traversal depth exceeded at launch index (%u,%u):\n",
            launch_index.x, launch_index.y );
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT:
      printf("Invalid miss SBT record index at launch index (%u,%u):\n",
             launch_index.x, launch_index.y );
      // optixGetExceptionInvalidSbtOffset()
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT:
      printf("Invalid hit SBT record index at launch index (%u,%u):\n",
             launch_index.x, launch_index.y );
      // optixGetExceptionInvalidSbtOffset()
      break;

    case OPTIX_EXCEPTION_CODE_INVALID_RAY:
      printf("Trace call containing Inf/NaN at launch index (%u,%u):\n",
             launch_index.x, launch_index.y );
      // optixGetExceptionInvalidRay()
      break;

    case OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH:
      printf("Callable param mismatch at launch index (%d,%d):\n",
             launch_index.x, launch_index.y );
      // optixGetExceptionParameterMismatch()
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE:
    default:
      printf("Caught exception 0x%X (%d) at launch index (%u,%u)\n",
             code, code, launch_index.x, launch_index.y );
      break;
  }

  // and write to frame buffer ...
  const int idx = launch_index.x + launch_index.y*rtLaunch.frame.size.x;
  rtLaunch.frame.framebuffer[idx] = make_color_rgb4u(make_float3(0.f, 0.f, 0.f));
}


//
// Shadow ray programs
// 
// The shadow PRD attenuation factor represents what fraction of the light
// is visible.  An attenuation factor of 0 indicates full shadow, no light
// makes it to the surface.  An attenuation factor of 1.0 indicates a 
// complete lack of shadow.
//

extern "C" __global__ void __closesthit__shadow_nop() {
  // no-op
}


//
// Shadow miss program for any kind of geometry
//   Regardless what type of geometry we're rendering, if we end up running
//   the miss program, then we know we didn't hit an occlusion, so the 
//   light attenuation factor should be 1.0.
extern "C" __global__ void __miss__shadow_nop() {
  // For scenes with either opaque or transmissive objects, 
  // a "miss" always indicates that there was no (further) occlusion, 
  // and thus no shadow.

  // no-op
}


// Shadow AH program for purely opaque geometry
//   If we encounter an opaque object during an anyhit traversal,
//   it sets the light attentuation factor to 0 (full shadow) and
//   immediately terminates the shadow traversal.
extern "C" __global__ void __anyhit__shadow_opaque() {
  // this material is opaque, so it fully attenuates all shadow rays
  PerRayData_shadow &prd = *(PerRayData_shadow*)getPRD<PerRayData_shadow>();
  prd.attenuation = make_float3(0.0f, 0.0f, 0.0f);

  // full shadow should cause us to early-terminate AH search
  optixTerminateRay();
}


// Shadow programs for scenes containing a mix of both opaque and transparent
//   objects.  In the case that a scene contains a mix of both fully opaque
//   and transparent objects, we have two different types of AH programs
//   for the two cases.  Prior to launching the shadow rays, the PRD attenuation
//   factor is set to 1.0 to indicate no shadowing, and it is subsequently
//   modified by the AH programs associated with the different objects in 
//   the scene.
//
//   To facilitate best performance in scenes that contain a mix of 
//   fully opaque and transparent geometry, we could run an anyhit against
//   opaque geometry first, and only if we had a miss, we could continue with
//   running anyhit traversal on just the transmissive geometry.
//

// Any hit program required for shadow filtering through transparent materials
extern "C" __global__ void __anyhit__shadow_transmission() {
  // use a VERY simple shadow filtering scheme based on opacity
  PerRayData_shadow &prd = *(PerRayData_shadow*)getPRD<PerRayData_shadow>();
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();

  float opacity;
  if (optixIsTriangleHit()) {
    opacity = rtLaunch.materials[sbtHG.materialindex].opacity;
  } else {
    opacity = 1.0f;  // XXXX hack
  }

#if 0
  const uint3 launch_index = optixGetLaunchIndex();
  if (launch_index.x == 994) {
    printf("AH xy:%d %d mat[%d] diffuse: %g  opacity: %g  atten: %g\n", 
           launch_index.x, launch_index.y, 
           sbtHG.materialindex, mat.diffuse, mat.opacity,
           prd.attenuation);
  }
#endif

  prd.attenuation *= make_float3(1.0f - opacity);

  // check to see if we've hit 100% shadow or not
  if (fmaxf(prd.attenuation) < 0.001f) {
    optixTerminateRay();
  } else {
#if defined(TACHYON_RAYSTATS)
    raystats2_buffer[launch_index].y++; // increment trans ray skip counter
#endif
    optixIgnoreIntersection();
  }
}



// Any hit program required for shadow filtering when an
// HMD/camera fade-and-clip is active, through both
// solid and transparent materials
extern "C" __global__ void any_hit_shadow_clip_sphere() {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();

  float opacity;
  if (optixIsTriangleHit()) {
    opacity = rtLaunch.materials[sbtHG.materialindex].opacity;
  } else {
    opacity = 1.0f;  // XXXX hack
  }

  // compute world space hit point for use in evaluating fade/clip effect
  float3 hit_point = ray_origin + t_hit * ray_direction;

  // compute additional attenuation from clipping sphere if enabled
  float clipalpha = 1.0f;
  if (rtLaunch.clipview_mode == 2) {
    sphere_fade_and_clip(hit_point, rtLaunch.cam.pos, rtLaunch.clipview_start, 
                         rtLaunch.clipview_end, clipalpha);
  }


  // use a VERY simple shadow filtering scheme based on opacity
  PerRayData_shadow &prd = *(PerRayData_shadow*)getPRD<PerRayData_shadow>();
  prd.attenuation = make_float3(1.0f - (clipalpha * opacity));

  // check to see if we've hit 100% shadow or not
  if (fmaxf(prd.attenuation) < 0.001f) {
    optixTerminateRay();
  } else {
#if defined(TACHYON_RAYSTATS)
    raystats2_buffer[launch_index].y++; // increment trans ray skip counter
#endif
    optixIgnoreIntersection();
  }
}


// 
// OptiX anyhit program for radiance rays, a no-op
// 

extern "C" __global__ void __anyhit__radiance_nop() {
  // no-op
}


//
// OptiX miss programs for drawing the background color or
// background color gradient when no objects are hit
//

// Miss program for solid background
extern "C" __global__ void __miss__radiance_solid_bg() {
  // Fog overrides the background color if we're using
  // Tachyon radial fog, but not for OpenGL style fog.
  PerRayData_radiance &prd = *(PerRayData_radiance*)getPRD<PerRayData_radiance>();
  prd.result = rtLaunch.scene.bg_color;
  prd.alpha = 0.0f; // alpha of background is 0.0f;
#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].w++; // increment miss counter
#endif
}


// Miss program for gradient background with perspective projection.
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
extern "C" __global__ void __miss__radiance_gradient_bg_sky_sphere() {
  PerRayData_radiance &prd = *(PerRayData_radiance*)getPRD<PerRayData_radiance>();
  // project ray onto the world "up" gradient, and compute the 
  // scalar color interpolation parameter
  float IdotG = dot(optixGetWorldRayDirection(), rtLaunch.scene.gradient);
  float val = (IdotG - rtLaunch.scene.gradient_botval) * 
              rtLaunch.scene.gradient_invrange;

#if 1
  // Add noise to gradient backgrounds to prevent Mach banding effects, 
  // particularly noticable in video streams or movie renderings.
  // Compute the delta between the top and bottom gradient colors and 
  // calculate the noise magnitude required, such that by adding it to the
  // scalar interpolation parameter we get more than +/-1ulp in the 
  // resulting interpolated color, as represented in an 8bpp framebuffer.
  //
  // XXX this is computing the same random seed on each miss presently,
  //     they should best be using an updated sample/seed each time
  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int randseed = tea<4>(launch_dim.x*launch_index.y+launch_index.x, subframe_count());
  float maxcoldelta = fmaxf(fabsf(rtLaunch.scene.bg_color_grad_top - rtLaunch.scene.bg_color_grad_bot));

  // Ideally the noise mag calc would take into account both max color delta
  // and launch_dim.y to avoid bending even with very subtle gradients.
  float noisemag = (3.0f/256.0f) / (maxcoldelta + 0.0005);
  float noise = noisemag * ((qnd_rng(randseed) * UINT32_RAND_MAX_INV) - 0.5f);

#if 0
  if ((launch_index.x == 1024) && (launch_index.y & 0x4)) {
    printf("y: %d  md: %g nm: %g  n: %g  v: %g  v+n: %g\n",
           launch_index.y, maxcoldelta, noisemag, noise, val, val+noise);
  }
#endif

  val += noise; // add the noise to the interpolation parameter
#endif

  val = __saturatef(val); // clamp the interpolation param to [0:1]
  float3 col = val * rtLaunch.scene.bg_color_grad_top +
               (1.0f - val) * rtLaunch.scene.bg_color_grad_bot;
  prd.result = col;
  prd.alpha = 0.0f; // alpha of background is 0.0f;
#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].w++; // increment miss counter
#endif
}


// Miss program for gradient background with orthographic projection.
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
extern "C" __global__ void __miss__radiance_gradient_bg_sky_plane() {
  PerRayData_radiance &prd = *(PerRayData_radiance*)getPRD<PerRayData_radiance>();
  float IdotG = dot(optixGetWorldRayDirection(), rtLaunch.scene.gradient);
  float val = (IdotG - rtLaunch.scene.gradient_botval) * 
              rtLaunch.scene.gradient_invrange;

#if 1
  // Add noise to gradient backgrounds to prevent Mach banding effects, 
  // particularly noticable in video streams or movie renderings.
  // Compute the delta between the top and bottom gradient colors and 
  // calculate the noise magnitude required, such that by adding it to the
  // scalar interpolation parameter we get more than +/-1ulp in the 
  // resulting interpolated color, as represented in an 8bpp framebuffer.
  //
  // XXX this is computing the same random seed on each miss presently,
  //     they should best be using an updated sample/seed each time
  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  unsigned int randseed = tea<4>(launch_dim.x*launch_index.y+launch_index.x, subframe_count());
  float maxcoldelta = fmaxf(fabsf(rtLaunch.scene.bg_color_grad_top - rtLaunch.scene.bg_color_grad_bot));

  // Ideally the noise mag calc would take into account both max color delta
  // and launch_dim.y to avoid bending even with very subtle gradients.
  float noisemag = (3.0f/256.0f) / (maxcoldelta + 0.0005);
  float noise = noisemag * ((qnd_rng(randseed) * UINT32_RAND_MAX_INV) - 0.5f);

  val += noise; // add the noise to the interpolation parameter
#endif

  val = __saturatef(val); // clamp the interpolation param to [0:1]
  float3 col = val * rtLaunch.scene.bg_color_grad_top +
               (1.0f - val) * rtLaunch.scene.bg_color_grad_bot;
  prd.result = col;
  prd.alpha = 0.0f; // alpha of background is 0.0f;
#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].w++; // increment miss counter
#endif
}


#if 0

//
// Clear the raystats buffers to zeros
//
#if defined(TACHYON_RAYSTATS)
extern "C" __global__ void clear_raystats_buffers() {
  raystats1_buffer[launch_index] = make_uint4(0, 0, 0, 0); // clear ray counters to zero
  raystats2_buffer[launch_index] = make_uint4(0, 0, 0, 0); // clear ray counters to zero
}
#endif


//
// Clear the accumulation buffer to zeros
//
extern "C" __global__ void clear_accumulation_buffer() {
  accumulation_buffer[launch_index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}


//
// Copy the contents of the accumulation buffer to the destination
// framebuffer, while converting the representation of the data from
// floating point down to unsigned chars, performing clamping and any
// other postprocessing at the same time.
//
extern "C" __global__ void draw_accumulation_buffer() {
#if defined(TACHYON_TIME_COLORING) && defined(TACHYON_TIME_COMBINE_MAX)
  float4 curcol = accumulation_buffer[launch_index];

  // divide time value by normalization factor (to scale up the max value)
  curcol.x /= accumulation_normalization_factor;

  // multiply the remaining color components (to average them)
  curcol.y *= accumulation_normalization_factor;
  curcol.z *= accumulation_normalization_factor;
  curcol.w *= accumulation_normalization_factor;

  framebuffer[launch_index] = make_color_rgb4u(curcol);
#else
  framebuffer[launch_index] = make_color_rgb4u(accumulation_buffer[launch_index] * accumulation_normalization_factor);
#endif
}

// no-op placeholder used when running with progressive rendering
extern "C" __global__ void draw_accumulation_buffer_stub() {
}

#endif

//
// OptiX programs that implement the camera models and ray generation code
//


//
// Ray gen accumulation buffer helper routines
//
static void __inline__ __device__ accumulate_color(float3 &col,
                                                   float alpha = 1.0f) {
  col *= rtLaunch.accumulation_normalization_factor;
  alpha *= rtLaunch.accumulation_normalization_factor;

  // and write to frame buffer ...
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = launch_index.x + launch_index.y*rtLaunch.frame.size.x;
  rtLaunch.frame.framebuffer[idx] = make_color_rgb4u(make_float4(col, alpha));

#if 0
//  const auto &accumulation_buffer = rtLaunch.frame.accumulation_buffer;
//  const uint3 launch_index = optixGetLaunchIndex();
//  const int idx = launch_index.x+launch_index.y*rtLaunch.frame.size.x;
#if defined(TACHYON_OPTIX_PROGRESSIVEAPI)
  if (progressive_enabled) {
    col *= accumulation_normalization_factor;
    alpha *= accumulation_normalization_factor;

#if OPTIX_VERSION < 3080
    // XXX prior to OptiX 3.8, a hard-coded gamma correction was required
    // VCA gamma correction workaround, changes gamma 2.2 back to gamma 1.0
    float invgamma = 1.0f / 0.4545f;
    col.x = powf(col.x, invgamma);
    col.y = powf(col.y, invgamma);
    col.z = powf(col.z, invgamma);
#endif

    // for optix-vca progressive mode accumulation is handled in server code
    accumulation_buffer[idx]  = make_float4(col, alpha);
  } else {
    // For batch mode we accumulate ourselves
    accumulation_buffer[idx] += make_float4(col, alpha);
  }
#else
  // For batch mode we accumulate ourselves
  accumulation_buffer[idx] += make_float4(col, alpha);
#endif
#endif
}



//
// CUDA device function for computing the new ray origin
// and ray direction, given the radius of the circle of confusion disc,
// and an orthonormal basis for each ray.
//
#if 1

static __device__ __inline__
void dof_ray(const float cam_dof_focal_dist, const float cam_dof_aperture_rad,
             const float3 &ray_origin_orig, float3 &ray_origin,
             const float3 &ray_direction_orig, float3 &ray_direction,
             unsigned int &randseed, const float3 &up, const float3 &right) {
  float3 focuspoint = ray_origin_orig + ray_direction_orig * cam_dof_focal_dist;
  float2 dofjxy;
  jitter_disc2f(randseed, dofjxy, cam_dof_aperture_rad);
  ray_origin = ray_origin_orig + dofjxy.x*right + dofjxy.y*up;
  ray_direction = normalize(focuspoint - ray_origin);
}

#else

// use low-discrepancy sequences for sampling the circle of confusion disc
static __device__ __inline__
void dof_ray(const float cam_dof_focal_dist, const float cam_dof_aperture_rad,
             const float3 &ray_origin_orig, float3 &ray_origin,
             const float3 &ray_direction_orig, float3 &ray_direction,
             float2 &qrnxy, const float3 &up, const float3 &right) {
  float3 focuspoint = ray_origin_orig + ray_direction_orig * cam_dof_focal_dist;
  float2 dofjxy;
  jitter_disc2f_qrn(qrnxy, dofjxy, cam_dof_aperture_rad);
  ray_origin = ray_origin_orig + dofjxy.x*right + dofjxy.y*up;
  ray_direction = normalize(focuspoint - ray_origin);
}

#endif


//
// Templated perspective camera ray generation code
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_perspective_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  float3 eyepos;
  uint viewport_sz_y, viewport_idx_y;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // right image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyepos = cam.pos + cam.U * cam.stereo_eyesep * 0.5f;
    } else {
      // left image
      viewport_idx_y = launch_index.y;
      eyepos = cam.pos - cam.U * cam.stereo_eyesep * 0.5f;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam.pos;
  }


  //
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam.zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);
  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(viewport_idx_y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f, 0.0f, 0.0f);
  float alpha = 0.0f;
  float3 ray_origin = eyepos;
  for (int s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam.U + jxy.y*cam.V + cam.W);

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              eyepos, ray_origin, ray_direction, ray_direction,
              randseed, cam.V, cam.U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.result = make_float3(0.0f, 0.0f, 0.0f);
    prd.importance = 1.f;
    prd.alpha = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    // send aasample counter to CH for much better AO sampling
    // when we run multiple AA samples per-pass
    unsigned int p2 = s;

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               u0, u1,                        // PRD ptr in 2x uint32
               p2);

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[fbIndex].x+=rtLaunch.aa_samples; // increment primary ray counter
#endif

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col, alpha);
#endif
}

extern "C" __global__ void __raygen__camera_perspective() {
  tachyon_camera_perspective_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_perspective_dof() {
  tachyon_camera_perspective_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_perspective_stereo() {
  tachyon_camera_perspective_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_perspective_stereo_dof() {
  tachyon_camera_perspective_general<1, 1>();
}




//
// Templated orthographic camera ray generation code
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_orthographic_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  float3 eyepos;
  uint viewport_sz_y, viewport_idx_y;
  float3 view_direction;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // right image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyepos = cam.pos + cam.U * cam.stereo_eyesep * 0.5f;
    } else {
      // left image
      viewport_idx_y = launch_index.y;
      eyepos = cam.pos - cam.U * cam.stereo_eyesep * 0.5f;
    }
    view_direction = normalize(cam.pos-eyepos + normalize(cam.W) * cam.stereo_convergence_dist);
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam.pos;
    view_direction = normalize(cam.W);
  }

  //
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam.zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);

  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(viewport_idx_y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f, 0.0f, 0.0f);
  float alpha = 0.0f;
  float3 ray_direction = view_direction;
  for (int s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);
    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_origin = eyepos + jxy.x*cam.U + jxy.y*cam.V;

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, view_direction, ray_direction,
              randseed, cam.V, cam.U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.alpha = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    // send aasample counter to CH for much better AO sampling
    // when we run multiple AA samples per-pass
    unsigned int p2 = s;

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               u0, u1,                        // PRD ptr in 2x uint32
               p2);

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].x+=rtLaunch.aa_samples; // increment primary ray counter
#endif

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col, alpha);
#endif
}

extern "C" __global__ void __raygen__camera_orthographic() {
  tachyon_camera_orthographic_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_orthographic_dof() {
  tachyon_camera_orthographic_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_orthographic_stereo() {
  tachyon_camera_orthographic_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_orthographic_stereo_dof() {
  tachyon_camera_orthographic_general<1, 1>();
}




//
// 360-degree stereoscopic cubemap image format for use with
// Oculus, Google Cardboard, and similar VR headsets
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_cubemap_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const auto &cam = rtLaunch.cam;

  // compute which cubemap face we're drawing by the X index.
  uint facesz = launch_dim.y; // square cube faces, equal to image height
  uint face = (launch_index.x / facesz) % 6;
  uint2 face_idx = make_uint2(launch_index.x % facesz, launch_index.y);

  // For the OTOY ORBX viewer, Oculus VR software, and some of the
  // related apps, the cubemap image is stored with the X axis oriented
  // such that when viewed as a 2-D image, they are all mirror images.
  // The mirrored left-right orientation used here corresponds to what is
  // seen standing outside the cube, whereas the ray tracer shoots
  // rays from the inside, so we flip the X-axis pixel storage order.
  // The top face of the cubemap has both the left-right and top-bottom
  // orientation flipped also.
  // Set per-face orthonormal basis for camera
  float3 face_U, face_V, face_W;
  switch (face) {
    case 0: // back face
      face_U =  cam.U;
      face_V =  cam.V;
      face_W = -cam.W;
      break;

    case 1: // front face
      face_U =  -cam.U;
      face_V =  cam.V;
      face_W =  cam.W;
      break;

    case 2: // top face
      face_U = -cam.W;
      face_V =  cam.U;
      face_W =  cam.V;
      break;

    case 3: // bottom face
      face_U = -cam.W;
      face_V = -cam.U;
      face_W = -cam.V;
      break;

    case 4: // left face
      face_U = -cam.W;
      face_V =  cam.V;
      face_W = -cam.U;
      break;

    case 5: // right face
      face_U =  cam.W;
      face_V =  cam.V;
      face_W =  cam.U;
      break;
  }

  // Stereoscopic rendering is provided by rendering in a side-by-side
  // format with the left eye image into the left half of a double-wide
  // framebuffer, and the right eye into the right half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // into an efficient cubemap texture.
  uint viewport_sz_x, viewport_idx_x;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-wide framebuffer when stereo is enabled
    viewport_sz_x = launch_dim.x >> 1;
    if (launch_index.x >= viewport_sz_x) {
      // right image
      viewport_idx_x = launch_index.x - viewport_sz_x;
      eyeshift =  0.5f * cam.stereo_eyesep;
    } else {
      // left image
      viewport_idx_x = launch_index.x;
      eyeshift = -0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_x = launch_dim.x;
    viewport_idx_x = launch_index.x;
    eyeshift = 0.0f;
  }

  //
  // general primary ray calculations, locked to 90-degree FoV per face...
  //
  float facescale = 1.0f / facesz;
  float2 d = make_float2(face_idx.x, face_idx.y) * facescale * 2.f - 1.0f; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+viewport_idx_x, subframe_count());

  float3 col = make_float3(0.0f, 0.0f, 0.0f);
  for (int s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);
    jxy = jxy * facescale * 2.f + d;
    float3 ray_direction = normalize(jxy.x*face_U + jxy.y*face_V + face_W);

    float3 ray_origin = cam.pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam.V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, face_V, face_U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    // send aasample counter to CH for much better AO sampling
    // when we run multiple AA samples per-pass
    unsigned int p2 = s;
    
    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               u0, u1,                        // PRD ptr in 2x uint32
               p2);

    col += prd.result;
  }

#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].x+=rtLaunch.aa_samples; // increment primary ray counter
#endif

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}


extern "C" __global__ void __raygen__camera_cubemap() {
  tachyon_camera_cubemap_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_cubemap_dof() {
  tachyon_camera_cubemap_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_cubemap_stereo() {
  tachyon_camera_cubemap_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_cubemap_stereo_dof() {
  tachyon_camera_cubemap_general<1, 1>();
}



//
// Camera ray generation code for planetarium dome display
// Generates a fisheye style frame with ~180 degree FoV
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_dome_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam.stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float fov = M_PIf; // dome FoV in radians

  // half FoV in radians, pixels beyond this distance are outside
  // of the field of view of the projection, and are set black
  float thetamax = 0.5 * fov;

  // The dome angle from center of the projection is proportional
  // to the image-space distance from the center of the viewport.
  // viewport_sz contains the viewport size, radperpix contains the
  // radians/pixel scaling factors in X/Y, and viewport_mid contains
  // the midpoint coordinate of the viewpoint used to compute the
  // distance from center.
  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 radperpix = fov / viewport_sz;
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f, 0.0f, 0.0f);
  float alpha = 0.0f;
  for (int s=0; s<rtLaunch.aa_samples; s++) {
    // compute the jittered image plane sample coordinate
    float2 jxy;
    jitter_offset2f(randseed, jxy);
    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;

    // compute the ray angles in X/Y and total angular distance from center
    float2 p = (viewport_idx - viewport_mid) * radperpix;
    float theta = hypotf(p.x, p.y);

    // pixels outside the dome FoV are treated as black by not
    // contributing to the color accumulator
    if (theta < thetamax) {
      float3 ray_direction;
      float3 ray_origin = cam.pos;

      if (theta == 0) {
        // handle center of dome where azimuth is undefined by
        // setting the ray direction to the zenith
        ray_direction = cam.W;
      } else {
        float sintheta, costheta;
        sincosf(theta, &sintheta, &costheta);
        float rsin = sintheta / theta; // normalize component
        ray_direction = cam.U*rsin*p.x + cam.V*rsin*p.y + cam.W*costheta;
        if (STEREO_ON) {
          // assumes a flat dome, where cam.W also points in the
          // audience "up" direction
          ray_origin += eyeshift * cross(ray_direction, cam.W);
        }

        if (DOF_ON) {
          float rcos = costheta / theta; // normalize component
          float3 ray_up    = -cam.U*rcos*p.x  -cam.V*rcos*p.y + cam.W*sintheta;
          float3 ray_right =  cam.U*(p.y/theta) + cam.V*(-p.x/theta);
          dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
                  ray_origin, ray_origin, ray_direction, ray_direction,
                  randseed, ray_up, ray_right);
        }
      }

      // trace the new ray...
      PerRayData_radiance prd;
      prd.importance = 1.f;
      prd.alpha = 1.f;
      prd.depth = 0;
      prd.transcnt = rtLaunch.max_trans;

      // the values we store the PRD pointer in:
      uint32_t u0, u1;
      packPointer( &prd, u0, u1 );

      // send aasample counter to CH for much better AO sampling
      // when we run multiple AA samples per-pass
      unsigned int p2 = s;
    
      optixTrace(rtLaunch.traversable,
                 ray_origin,
                 ray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 u0, u1,                        // PRD ptr in 2x uint32
                 p2);

      col += prd.result;
      alpha += prd.alpha;
    }
  }

#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].x+=rtLaunch.aa_samples; // increment primary ray counter
#endif

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col, alpha);
#endif
}


extern "C" __global__ void __raygen__camera_dome_master() {
  tachyon_camera_dome_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_dome_master_dof() {
  tachyon_camera_dome_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_dome_master_stereo() {
  tachyon_camera_dome_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_dome_master_stereo_dof() {
  tachyon_camera_dome_general<1, 1>();
}


//
// Camera ray generation code for 360 degre FoV
// equirectangular (lat/long) projection suitable
// for use a texture map for a sphere, e.g. for
// immersive VR HMDs, other spheremap-based projections.
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_equirectangular_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const auto &cam = rtLaunch.cam;

  // The Samsung GearVR OTOY ORBX players have the left eye image on top,
  // and the right eye image on the bottom.
  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam.stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 radperpix = M_PIf / viewport_sz * make_float2(2.0f, 1.0f);
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f, 0.0f, 0.0f);
  for (int s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;
    float2 rangle = (viewport_idx - viewport_mid) * radperpix;

    float sin_ax, cos_ax, sin_ay, cos_ay;
    sincosf(rangle.x, &sin_ax, &cos_ax);
    sincosf(rangle.y, &sin_ay, &cos_ay);

    float3 ray_direction = normalize(cos_ay * (cos_ax * cam.W + sin_ax * cam.U) + sin_ay * cam.V);

    float3 ray_origin = cam.pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam.V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      float3 ray_right = normalize(cos_ay * (-sin_ax * cam.W - cos_ax * cam.U) + sin_ay * cam.V);
      float3 ray_up = cross(ray_direction, ray_right);
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, ray_up, ray_right);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    // send aasample counter to CH for much better AO sampling
    // when we run multiple AA samples per-pass
    unsigned int p2 = s;
    
    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               u0, u1,                        // PRD ptr in 2x uint32
               p2);

    col += prd.result;
  }

#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].x+=rtLaunch.aa_samples; // increment primary ray counter
#endif

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

extern "C" __global__ void __raygen__camera_equirectangular() {
  tachyon_camera_equirectangular_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_equirectangular_dof() {
  tachyon_camera_equirectangular_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_equirectangular_stereo() {
  tachyon_camera_equirectangular_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_equirectangular_stereo_dof() {
  tachyon_camera_equirectangular_general<1, 1>();
}




//
// Templated Oculus Rift perspective camera ray generation code
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_oculus_rift_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in a side-by-side
  // format with the left eye image in the left half of a double-wide
  // framebuffer, and the right eye in the right half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_x, viewport_idx_x;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-wide framebuffer when stereo is enabled
    viewport_sz_x = launch_dim.x >> 1;
    if (launch_index.x >= viewport_sz_x) {
      // right image
      viewport_idx_x = launch_index.x - viewport_sz_x;
      eyeshift =  0.5f * cam.stereo_eyesep;
    } else {
      // left image
      viewport_idx_x = launch_index.x;
      eyeshift = -0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_x = launch_dim.x;
    viewport_idx_x = launch_index.x;
    eyeshift = 0.0f;
  }

  //
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(viewport_sz_x) / float(launch_dim.y), 1.0f) * cam.zoom;
  float2 viewportscale = 1.0f / make_float2(viewport_sz_x, launch_dim.y);
  float2 d = make_float2(viewport_idx_x, launch_index.y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane


  // Compute barrel distortion required to correct for the pincushion inherent
  // in the plano-convex optics in the Oculus Rift, Google Cardboard, etc.
  // Barrel distortion involves computing distance of the pixel from the
  // center of the eye viewport, and then scaling this distance by a factor
  // based on the original distance:
  //   rnew = 0.24 * r^4 + 0.22 * r^2 + 1.0
  // Since we are only using even powers of r, we can use efficient
  // squared distances everywhere.
  // The current implementation doesn't discard rays that would have fallen
  // outside of the original viewport FoV like most OpenGL implementations do.
  // The current implementation computes the distortion for the initial ray
  // but doesn't apply these same corrections to antialiasing jitter, to
  // depth-of-field jitter, etc, so this leaves something to be desired if
  // we want best quality, but this raygen code is really intended for
  // interactive display on an Oculus Rift or Google Cardboard type viewer,
  // so I err on the side of simplicity/speed for now.
  float2 cp = make_float2(viewport_sz_x >> 1, launch_dim.y >> 1) * viewportscale * aspect * 2.f - aspect;;
  float2 dr = d - cp;
  float r2 = dr.x*dr.x + dr.y*dr.y;
  float r = 0.24f*r2*r2 + 0.22f*r2 + 1.0f;
  d = r * dr;

  int subframecount = subframe_count();
  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+viewport_idx_x, subframecount);

  float3 eyepos = cam.pos;
  if (STEREO_ON) {
    eyepos += eyeshift * cam.U;
  }

  float3 ray_origin = eyepos;
  float3 col = make_float3(0.0f, 0.0f, 0.0f);
  for (int s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    // don't jitter the first sample, since when using an HMD we often run
    // with only one sample per pixel unless the user wants higher fidelity
    jxy *= (subframecount > 0 || s > 0);

    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam.U + jxy.y*cam.V + cam.W);

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              eyepos, ray_origin, ray_direction, ray_direction,
              randseed, cam.V, cam.U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    // send aasample counter to CH for much better AO sampling
    // when we run multiple AA samples per-pass
    unsigned int p2 = s;

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               u0, u1,                        // PRD ptr in 2x uint32
               p2);

    col += prd.result;
  }

#if defined(TACHYON_RAYSTATS)
  raystats1_buffer[launch_index].x+=rtLaunch.aa_samples; // increment primary ray counter
#endif

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

extern "C" __global__ void __raygen__camera_oculus_rift() {
  tachyon_camera_oculus_rift_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_oculus_rift_dof() {
  tachyon_camera_oculus_rift_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_oculus_rift_stereo() {
  tachyon_camera_oculus_rift_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_oculus_rift_stereo_dof() {
  tachyon_camera_oculus_rift_general<1, 1>();
}



//
// Shared utility functions needed by custom geometry intersection or
// shading helper functions.
//

// normal calc routine needed only to simplify the macro to produce the
// complete combinatorial expansion of template-specialized
// closest hit radiance functions
static __inline__ __device__ 
float3 calc_ffworld_normal(const float3 &Nshading, const float3 &Ngeometric) {
  float3 world_shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(Nshading));
  float3 world_geometric_normal = normalize(optixTransformNormalFromObjectToWorldSpace(Ngeometric));
  const float3 ray_dir = optixGetWorldRayDirection();
  return faceforward(world_shading_normal, -ray_dir, world_geometric_normal);
}



//
// Object and/or vertex/color/normal buffers...
//


//
// Color-per-cone array primitive
//
extern "C" __global__ void __intersection__cone_array_color() {
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();
  const int primID = optixGetPrimitiveIndex();

  float3 base = sbtHG.cone.base[primID];
  float3 apex = sbtHG.cone.apex[primID];
  float baserad = sbtHG.cone.baserad[primID];
  float apexrad = sbtHG.cone.apexrad[primID];

  float3 axis = (apex - base);
  float3 obase = ray_origin - base;
  float3 oapex = ray_origin - apex;
  float m0 = dot(axis, axis);
  float m1 = dot(obase, axis);
  float m2 = dot(obj_ray_direction, axis);
  float m3 = dot(obj_ray_direction, obase);
  float m5 = dot(obase, obase);
  float m9 = dot(oapex, axis);

  // caps...

  float rr = baserad - apexrad;
  float hy = m0 + rr*rr;
  float k2 = m0*m0    - m2*m2*hy;
  float k1 = m0*m0*m3 - m1*m2*hy + m0*baserad*(rr*m2*1.0f             );
  float k0 = m0*m0*m5 - m1*m1*hy + m0*baserad*(rr*m1*2.0f - m0*baserad);
  float h = k1*k1 - k2*k0;
  if (h < 0.0f) 
    return; // no intersection

  float t = (-k1-sqrt(h))/k2;
  float y = m1 + t*m2;
  if (y < 0.0f || y > m0) 
    return; // no intersection
  
  optixReportIntersection(t, RT_HIT_CONE);
}


static __host__ __device__ __inline__
void get_shadevars_cone_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:

  float3 base = sbtHG.cone.base[primID];
  float3 apex = sbtHG.cone.apex[primID];
  float baserad = sbtHG.cone.baserad[primID];
  float apexrad = sbtHG.cone.apexrad[primID];
  
  float3 axis = (apex - base);
  float3 obase = ray_origin - base;
  float3 oapex = ray_origin - apex;
  float m0 = dot(axis, axis);
  float m1 = dot(obase, axis);
  float m2 = dot(ray_direction, axis);
  float m3 = dot(ray_direction, obase);
  float m5 = dot(obase, obase);
  float m9 = dot(oapex, axis);

  // caps...

  float rr = baserad - apexrad;
  float hy = m0 + rr*rr;
  float k2 = m0*m0    - m2*m2*hy;
  float k1 = m0*m0*m3 - m1*m2*hy + m0*baserad*(rr*m2*1.0f             );
  float k0 = m0*m0*m5 - m1*m1*hy + m0*baserad*(rr*m1*2.0f - m0*baserad);
  float h = k1*k1 - k2*k0;
//  if (h < 0.0f) 
//    return; // no intersection
 
  float t = (-k1-sqrt(h))/k2;
  float y = m1 + t*m2;
//  if (y < 0.0f || y > m0) 
//    return; // no intersection

  float3 hit = t * ray_direction; 
  float3 Ng = normalize(m0*(m0*(obase + hit) + rr*axis*baserad) - axis*hy*y);

  shading_normal = calc_ffworld_normal(Ng, Ng);
}



//
// Color-per-cylinder array primitive
//
// XXX not yet handling Obj vs. World coordinate xforms
extern "C" __global__ void __intersection__cylinder_array_color() {
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();
  const int primID = optixGetPrimitiveIndex();

  float3 start = sbtHG.cyl.start[primID];
  float3 end = sbtHG.cyl.end[primID];
  float radius = sbtHG.cyl.radius[primID];

  float3 axis = (end - start);
  float3 rc = ray_origin - start;
  float3 n = cross(obj_ray_direction, axis);
  float lnsq = dot(n, n);

  // check if ray is parallel to cylinder
  if (lnsq == 0.0f) {
    return; // ray is parallel, we missed or went through the "hole"
  }
  float invln = rsqrtf(lnsq);
  n *= invln;
  float d = fabsf(dot(rc, n));

  // check for cylinder intersection
  if (d <= radius) {
    float3 O = cross(rc, axis);
    float t = -dot(O, n) * invln;
    O = cross(n, axis);
    O = normalize(O);
    float s = dot(obj_ray_direction, O);
    s = fabs(sqrtf(radius*radius - d*d) / s);
    float axlen = length(axis);
    float3 axis_u = normalize(axis);

    // test hit point against cylinder ends
    float tin = t - s;
    float3 hit = ray_origin + obj_ray_direction * tin;
    float3 tmp2 = hit - start;
    float tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      optixReportIntersection(tin, RT_HIT_CYLINDER);
    }

    // continue with second test...
    float tout = t + s;
    hit = ray_origin + obj_ray_direction * tout;
    tmp2 = hit - start;
    tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      optixReportIntersection(tout, RT_HIT_CYLINDER);
    }
  }
}


static __host__ __device__ __inline__
void get_shadevars_cylinder_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:
  float3 start = sbtHG.cyl.start[primID];
  float3 end = sbtHG.cyl.end[primID];
  float3 axis_u = normalize(end-start);
  float3 hit = ray_origin + ray_direction * t_hit;
  float3 tmp2 = hit - start;
  float tmp = dot(tmp2, axis_u);
  float3 Ng = normalize(hit - (tmp * axis_u + start));
  shading_normal = calc_ffworld_normal(Ng, Ng);
}



#if 0

extern "C" __global__ void cylinder_array_color_bounds(int primIdx, float result[6]) {
  const float3 start = cylinder_buffer[primIdx].start;
  const float3 end = start + cylinder_buffer[primIdx].axis;
  const float3 rad = make_float3(cylinder_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = fminf(start - rad, end - rad);
    aabb->m_max = fmaxf(start + rad, end + rad);
  } else {
    aabb->invalidate();
  }
}

#endif


//
// Ring array primitive
//
extern "C" __global__ void __intersection__ring_array() {
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();
  const float3 obj_ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();
  const int primID = optixGetPrimitiveIndex();

  const float3 center = sbtHG.ring.center[primID];
  const float3 norm = sbtHG.ring.norm[primID];
  const float inrad = sbtHG.ring.inrad[primID];
  const float outrad = sbtHG.ring.outrad[primID];

  float d = -dot(center, norm);
  float t = -(d + dot(norm, obj_ray_origin));
  float td = dot(norm, obj_ray_direction);
  if (td != 0.0f) {
    t /= td;
    if (t >= 0.0f) {
      float3 hit = obj_ray_origin + t * obj_ray_direction;
      float rd = length(hit - center);
      if ((rd > inrad) && (rd < outrad)) {
        optixReportIntersection(t, RT_HIT_RING);
      }
    }
  }
}


static __host__ __device__ __inline__
void get_shadevars_ring_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:
  float3 Ng = sbtHG.ring.norm[primID];
  shading_normal = calc_ffworld_normal(Ng, Ng);
}



#if 0

extern "C" __global__ void ring_array_color_bounds(int primIdx, float result[6]) {
  const float3 center = ring_buffer[primIdx].center;
  const float3 rad = make_float3(ring_buffer[primIdx].outrad);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = center - rad;
    aabb->m_max = center + rad;
  } else {
    aabb->invalidate();
  }
}

#endif



#if defined(TACHYON_USE_SPHERES_HEARNBAKER)

// Ray-sphere intersection method with improved floating point precision
// for cases where the sphere size is small relative to the distance
// from the camera to the sphere.  This implementation is based on
// Eq. 10-72, p.603 of "Computer Graphics with OpenGL", 3rd Ed., by 
// Donald Hearn and Pauline Baker, 2004, Eq. 10, p.639 in the 4th edition 
// (Hearn, Baker, Carithers), and in Ray Tracing Gems, 
// Precision Improvements for Ray/Sphere Intersection, pp. 87-94, 2019.
static __host__ __device__ __inline__
void sphere_intersect_hearn_baker(float3 center, float rad) {
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();

  // if scaling xform was been applied, the ray length won't be normalized, 
  // so we have to scale the resulting t hitpoints to world coords
  float ray_invlen;
  const float3 ray_direction = normalize_len(obj_ray_direction, ray_invlen);

  float3 deltap = center - ray_origin;
  float ddp = dot(ray_direction, deltap);
  float3 remedyTerm = deltap - ddp * ray_direction;
  float disc = rad*rad - dot(remedyTerm, remedyTerm);
  if (disc >= 0.0f) {
    float disc_root = sqrtf(disc);
    float t1 = ddp - disc_root;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);

    float t2 = ddp + disc_root;
    t2 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t2, RT_HIT_SPHERE);
  }
}

#else

//
// Ray-sphere intersection using standard geometric solution approach
//
static __host__ __device__ __inline__
void sphere_intersect_classic(float3 center, float rad) {
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();

  // if scaling xform was been applied, the ray length won't be normalized, 
  // so we have to scale the resulting t hitpoints to world coords
  float ray_invlen;
  const float3 ray_direction = normalize_len(obj_ray_direction, ray_invlen);

  float3 V = center - ray_origin;
  float b = dot(V, ray_direction);
  float disc = b*b + rad*rad - dot(V, V);
  if (disc > 0.0f) {
    disc = sqrtf(disc);

//#define FASTONESIDEDSPHERES 1
#if defined(FASTONESIDEDSPHERES)
    // only calculate the nearest intersection, for speed
    float t1 = b - disc;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);
#else
    float t2 = b + disc;
    t2 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t2, RT_HIT_SPHERE);

    float t1 = b - disc;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);
#endif
  }
}

#endif


//
// Sphere array primitive
//
extern "C" __global__ void __intersection__sphere_array() {
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();
  const int primID = optixGetPrimitiveIndex();
  float3 center = sbtHG.sphere.center[primID];
  float radius = sbtHG.sphere.radius[primID];

#if defined(TACHYON_USE_SPHERES_HEARNBAKER)
  sphere_intersect_hearn_baker(center, radius);
#else
  sphere_intersect_classic(center, radius);
#endif
}


static __host__ __device__ __inline__
void get_shadevars_sphere_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:
  float3 center = sbtHG.sphere.center[primID];
  float radius = sbtHG.sphere.radius[primID];
  float3 deltap = center - ray_origin;
  float3 Ng = (t_hit * ray_direction - deltap) * (1.0f / radius);
  shading_normal = calc_ffworld_normal(Ng, Ng);
}


#if 0
// OptiX 6.x bounds code
extern "C" __global__ void sphere_array_bounds(int primIdx, float result[6]) {
  const float3 cen = sphere_buffer[primIdx].center;
  const float3 rad = make_float3(sphere_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}
#endif


//
// Color-per-sphere sphere array
//
extern "C" __global__ void __intersection__sphere_array_color() {
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*)optixGetSbtDataPointer();
  const int primID = optixGetPrimitiveIndex();
  float3 center = sbtHG.sphere.center[primID];
  float radius = sbtHG.sphere.radius[primID];

#if defined(TACHYON_USE_SPHERES_HEARNBAKER)
  sphere_intersect_hearn_baker(center, radius);
#else
  sphere_intersect_classic(center, radius);
#endif
}


#if 0
// OptiX 6.x bounds code
extern "C" __global__ void sphere_array_color_bounds(int primIdx, float result[6]) {
  const float3 cen = sphere_color_buffer[primIdx].center;
  const float3 rad = make_float3(sphere_color_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}
#endif



//
// Triangle mesh/array primitives
//

static __host__ __device__ __inline__
void get_shadevars_trimesh(const GeomSBTHG &sbtHG, float3 &hit_color,
                           float3 &shading_normal, int &matidx) {
  const int primID = optixGetPrimitiveIndex();

  const int3 index = sbtHG.trimesh.index[primID];
  const float2 barycentrics = optixGetTriangleBarycentrics();

  // compute geometric and shading normals:
  float3 Ng, Ns;
  if (sbtHG.trimesh.packednormals != nullptr) {
    Ng = unpackNormal(sbtHG.trimesh.packednormals[primID].x);

    const float3& n0 = unpackNormal(sbtHG.trimesh.packednormals[index.x].y);
    const float3& n1 = unpackNormal(sbtHG.trimesh.packednormals[index.y].z);
    const float3& n2 = unpackNormal(sbtHG.trimesh.packednormals[index.z].w);

    // interpolate triangle normal from barycentrics
    Ns = normalize(n0 * (1.0f - barycentrics.x - barycentrics.y) +
                   n1 * barycentrics.x + n2 * barycentrics.y);
  } else if (sbtHG.trimesh.normals != nullptr) {
    const float3 &A = sbtHG.trimesh.vertex[index.x];
    const float3 &B = sbtHG.trimesh.vertex[index.y];
    const float3 &C = sbtHG.trimesh.vertex[index.z];
    Ng = normalize(cross(B-A, C-A));

    const float3& n0 = sbtHG.trimesh.normals[index.x];
    const float3& n1 = sbtHG.trimesh.normals[index.y];
    const float3& n2 = sbtHG.trimesh.normals[index.z];

    // interpolate triangle normal from barycentrics
    Ns = normalize(n0 * (1.0f - barycentrics.x - barycentrics.y) +
                   n1 * barycentrics.x + n2 * barycentrics.y);
  } else {
    const float3 &A = sbtHG.trimesh.vertex[index.x];
    const float3 &B = sbtHG.trimesh.vertex[index.y];
    const float3 &C = sbtHG.trimesh.vertex[index.z];
    Ns = Ng = normalize(cross(B-A, C-A));
  }
  shading_normal = calc_ffworld_normal(Ns, Ng);

  // Assign vertex-interpolated, per-primitive or uniform color
  if (sbtHG.trimesh.vertexcolors3f != nullptr) {
    const float3 c0 = sbtHG.trimesh.vertexcolors3f[index.x];
    const float3 c1 = sbtHG.trimesh.vertexcolors3f[index.y];
    const float3 c2 = sbtHG.trimesh.vertexcolors3f[index.z];

    // interpolate triangle color from barycentrics
    hit_color = (c0 * (1.0f - barycentrics.x - barycentrics.y) +
                 c1 * barycentrics.x + c2 * barycentrics.y);
  } else if (sbtHG.trimesh.vertexcolors3f != nullptr) {
    const float ci2f = 1.0f / 255.0f;
    const float3 c0 = sbtHG.trimesh.vertexcolors4u[index.x] * ci2f;
    const float3 c1 = sbtHG.trimesh.vertexcolors4u[index.y] * ci2f;
    const float3 c2 = sbtHG.trimesh.vertexcolors4u[index.z] * ci2f;

    // interpolate triangle color from barycentrics
    hit_color = (c0 * (1.0f - barycentrics.x - barycentrics.y) +
                 c1 * barycentrics.x + c2 * barycentrics.y);
  } else if (sbtHG.prim_color != nullptr) {
    hit_color = sbtHG.prim_color[primID];
  } else {
    hit_color = sbtHG.uniform_color;
  }

  matidx = sbtHG.materialindex;
}



#if 0

// inline device function for computing triangle bounding boxes
__device__ __inline__ void generic_tri_bounds(optix::Aabb *aabb,
                                              float3 v0, float3 v1, float3 v2) {
#if 1
  // conventional paranoid implementation that culls degenerate triangles
  float area = length(cross(v1-v0, v2-v0));
  if (area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf(fminf(v0, v1), v2);
    aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
  } else {
    aabb->invalidate();
  }
#else
  // don't cull any triangles, even if they might be degenerate
  aabb->m_min = fminf(fminf(v0, v1), v2);
  aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
#endif
}


//
// triangle mesh with vertices, geometric normal, uniform color
//
extern "C" __global__ void ort_tri_intersect(int primIdx) {
  float3 v0 = stri_buffer[primIdx].v0;
  float3 v1 = tri_buffer[primIdx].v1;
  float3 v2 = tri_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      shading_normal = geometric_normal = normalize(n);

      // uniform color for the entire object
      prim_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

extern "C" __global__ void ort_tri_bounds(int primIdx, float result[6]) {
  float3 v0 = tri_buffer[primIdx].v0;
  float3 v1 = tri_buffer[primIdx].v1;
  float3 v2 = tri_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}


//
// triangle mesh with vertices, smoothed normals, uniform color
//
extern "C" __global__ void ort_stri_intersect(int primIdx) {
  float3 v0 = stri_buffer[primIdx].v0;
  float3 v1 = stri_buffer[primIdx].v1;
  float3 v2 = stri_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = stri_buffer[primIdx].n0;
      float3 n1 = stri_buffer[primIdx].n1;
      float3 n2 = stri_buffer[primIdx].n2;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma));
      geometric_normal = normalize(n);

      // uniform color for the entire object
      prim_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

extern "C" __global__ void ort_stri_bounds(int primIdx, float result[6]) {
  float3 v0 = stri_buffer[primIdx].v0;
  float3 v1 = stri_buffer[primIdx].v1;
  float3 v2 = stri_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}


//
// triangle mesh with vertices, smoothed normals, colors
//
extern "C" __global__ void ort_vcstri_intersect(int primIdx) {
  float3 v0 = vcstri_buffer[primIdx].v0;
  float3 v1 = vcstri_buffer[primIdx].v1;
  float3 v2 = vcstri_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = vcstri_buffer[primIdx].n0;
      float3 n1 = vcstri_buffer[primIdx].n1;
      float3 n2 = vcstri_buffer[primIdx].n2;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma));
      geometric_normal = normalize(n);

      float3 c0 = vcstri_buffer[primIdx].c0;
      float3 c1 = vcstri_buffer[primIdx].c1;
      float3 c2 = vcstri_buffer[primIdx].c2;
      prim_color = c1*beta + c2*gamma + c0*(1.0f-beta-gamma);
      rtReportIntersection(0);
    }
  }
}

extern "C" __global__ void ort_vcstri_bounds(int primIdx, float result[6]) {
  float3 v0 = vcstri_buffer[primIdx].v0;
  float3 v1 = vcstri_buffer[primIdx].v1;
  float3 v2 = vcstri_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}

#endif



//
// Support functions for closest hit and any hit programs for radiance rays
//  

// Fog implementation
static __device__ float fog_coord(float3 hit_point) {
  // Compute planar fog (e.g. to match OpenGL) by projecting t value onto
  // the camera view direction vector to yield a planar a depth value.
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();

  const auto &scene = rtLaunch.scene;

  float r = dot(ray_direction, rtLaunch.cam.W) * t_hit;
  float f=1.0f;
  float v;

  switch (scene.fog_mode) {
    case 1: // RT_FOG_LINEAR
      f = (scene.fog_end - r) / (scene.fog_end - scene.fog_start);
      break;

    case 2: // RT_FOG_EXP
      // XXX Tachyon needs to allow fog_start to be non-zero for 
      //     exponential fog, but fixed-function OpenGL does not...
      // float v = fog_density * (r - fog_start);
      v = scene.fog_density * r;
      f = expf(-v);
      break;

    case 3: // RT_FOG_EXP2
      // XXX Tachyon needs to allow fog_start to be non-zero for 
      //     exponential fog, but fixed-function OpenGL does not...
      // float v = fog_density * (r - fog_start);
      v = scene.fog_density * r;
      f = expf(-v*v);
      break;

    case 0: // RT_FOG_NONE
    default:
      break;
  }

  return __saturatef(f);
}


static __device__ float3 fog_color(float fogmod, float3 hit_col) {
  float3 col = (fogmod * hit_col) + ((1.0f - fogmod) * rtLaunch.scene.bg_color);
  return col;
}



//
// trivial ambient occlusion implementation
//
static __device__ float3 shade_ambient_occlusion(float3 hit, float3 N, float aoimportance) {
  // unweighted non-importance-sampled scaling factor
  float lightscale = 2.0f / float(rtLaunch.lights.ao_samples);
  float3 inten = make_float3(0.0f);

#if 1
  // Improve AO RNG seed generation when more than one AA sample is run 
  // per rendering pass.  The classic OptiX 6 formulation doesn't work 
  // as well now that we do our own subframe counting, and with RTX hardware
  // we typically want multiple AA samples per pass now unlike before.
  unsigned int aas = 1+optixGetPayload_2();
  int sf = subframe_count() * 313331337;
  int teabits1 = sf * aas;
  unsigned int randseed = tea<2>(teabits1, teabits1);
#else
  unsigned int randseed = tea<2>(subframe_count(), subframe_count());
#endif

#if 0
  const uint3 launch_index = optixGetLaunchIndex();
  if (launch_index.x == 994 && launch_index.y == 600) {
    printf("AO: xy:%d %d sf: %d s: %u  rs: %d\n", 
           launch_index.x, launch_index.y, sf, aas, randseed);
  }
#endif

  PerRayData_shadow shadow_prd;
  // do all the samples requested, with no observance of importance
  for (int s=0; s<rtLaunch.lights.ao_samples; s++) {
    float3 dir;
    jitter_sphere3f(randseed, dir);
    float ndotambl = dot(N, dir);

    // flip the ray so it's in the same hemisphere as the surface normal
    if (ndotambl < 0.0f) {
      ndotambl = -ndotambl;
      dir = -dir;
    }

    float3 aoray_origin, aoray_direction;
    float tmax=rtLaunch.lights.ao_maxdist;
#ifdef USE_REVERSE_SHADOW_RAYS
    if (shadows_enabled == RT_SHADOWS_ON_REVERSE) {
      // reverse any-hit ray traversal direction for increased perf
      // XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
      //     it works well for scenes that fall within the view volume,
      //     given the relationship between the model and camera coordinate
      //     systems, but this would be best computed by the diagonal of the
      //     AABB for the full scene, and then scaled into camera coordinates.
      //     The REVERSE_RAY_STEP size is computed to avoid self intersection
      //     with the surface we're shading.
      tmax = REVERSE_RAY_LENGTH - REVERSE_RAY_STEP;
      aoray_origin = hit + dir * REVERSE_RAY_LENGTH;
      aoray_direction = -dir;
    } else
#endif
    {
#if defined(TACHYON_USE_RAY_STEP)
      aoray_origin = hit + TACHYON_RAY_STEP;
#else
      aoray_origin = hit;
#endif
      aoray_direction = dir;
    }

    shadow_prd.attenuation = make_float3(1.0f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &shadow_prd, u0, u1 );

    optixTrace(rtLaunch.traversable,
               aoray_origin,
               aoray_direction,
               0.0f,                          // tmin
               tmax,                          // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
#if 1
               OPTIX_RAY_FLAG_NONE,
#elif 1
               // Hard shadows only, no opacity filtering.
               // For shadow rays skip any/closest hit and terminate 
               // on first intersection with anything.
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
#endif
               RT_RAY_TYPE_SHADOW,            // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_SHADOW,            // missSBTIndex
               u0, u1);                       // PRD ptr in 2x uint32

#if defined(TACHYON_RAYSTATS)
    raystats1_buffer[launch_index].z++; // increment AO shadow ray counter
#endif
    inten += ndotambl * shadow_prd.attenuation;
  }

  return inten * lightscale;
}



template<int SHADOWS_ON>       /// scene-wide shading property
static __device__ __inline__ void shade_light(float3 &result,
                                              float3 &hit_point,
                                              float3 &N, float3 &L,
                                              float p_Kd,
                                              float p_Ks,
                                              float p_phong_exp,
                                              float3 &col,
                                              float3 &phongcol,
                                              float shadow_tmax) {
  float inten = dot(N, L);

  // cast shadow ray
  float3 light_attenuation = make_float3(static_cast<float>(inten > 0.0f));
  if (SHADOWS_ON && rtLaunch.lights.shadows_enabled && inten > 0.0f) {
    PerRayData_shadow shadow_prd;
    shadow_prd.attenuation = make_float3(1.0f);

    float3 shadowray_origin, shadowray_direction;
    float tmax=shadow_tmax;
#ifdef USE_REVERSE_SHADOW_RAYS
    if (shadows_enabled == RT_SHADOWS_ON_REVERSE) {
      // reverse any-hit ray traversal direction for increased perf
      // XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
      //     it works well for scenes that fall within the view volume,
      //     given the relationship between the model and camera coordinate
      //     systems, but this would be best computed by the diagonal of the
      //     AABB for the full scene, and then scaled into camera coordinates.
      //     The REVERSE_RAY_STEP size is computed to avoid self intersection
      //     with the surface we're shading.
      tmax = REVERSE_RAY_LENGTH - REVERSE_RAY_STEP;
      shadowray_origin = hit_point + L * REVERSE_RAY_LENGTH;
      shadowray_direction = -L
      tmax = fminf(tmax, shadow_tmax));
    }
    else
#endif
    {
      shadowray_origin = hit_point + TACHYON_RAY_STEP;
      shadowray_direction = L;
    }

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &shadow_prd, u0, u1 );

    optixTrace(rtLaunch.traversable,
               shadowray_origin,
               shadowray_direction,
               0.0f,                          // tmin
               tmax,                          // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
#if 1
               OPTIX_RAY_FLAG_NONE,
#else
               // Hard shadows only, no opacity filtering.
               // For shadow rays skip any/closest hit and terminate 
               // on first intersection with anything.
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
#endif
               RT_RAY_TYPE_SHADOW,            // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_SHADOW,            // missSBTIndex
               u0, u1);                       // PRD ptr in 2x uint32

#if defined(TACHYON_RAYSTATS)
    raystats1_buffer[launch_index].y++; // increment shadow ray counter
#endif
    light_attenuation = shadow_prd.attenuation;
  }

  // If not completely shadowed, light the hit point.
  // When shadows are disabled, the light can't possibly be attenuated.
  if (!SHADOWS_ON || fmaxf(light_attenuation) > 0.0f) {
    result += col * p_Kd * inten * light_attenuation;

    // add specular hightlight using Blinn's halfway vector approach
    const float3 ray_direction = optixGetWorldRayDirection();
    float3 H = normalize(L - ray_direction);
    float nDh = dot(N, H);
    if (nDh > 0) {
      float power = powf(nDh, p_phong_exp);
      phongcol += make_float3(p_Ks) * power * light_attenuation;
    }
  }
}




//
// Partial re-implementation of the key portions of Tachyon's "full" shader.
//
// This shader has been written to be expanded into a large set of
// fully specialized shaders generated through combinatorial expansion
// of each of the major shader features associated with scene-wide or
// material-specific shading properties.
// At present, there are three scene-wide properties (fog, shadows, AO),
// and three material-specific properties (outline, reflection, transmission).
// There can be a performance cost for OptiX work scheduling of disparate
// materials if too many unique materials are used in a scene.
// Although there are 8 combinations of scene-wide parameters and
// 8 combinations of material-specific parameters (64 in total),
// the scene-wide parameters are uniform for the whole scene.
// We will therefore only have at most 8 different shader variants
// in use in a given scene, due to the 8 possible combinations
// of material-specific (outline, reflection, transmission) properties.
//
// The macros that generate the full set of 64 possible shader variants
// are at the very end of this source file.
//
template<int CLIP_VIEW_ON,     /// scene-wide shading property
         int HEADLIGHT_ON,     /// scene-wide shading property
         int FOG_ON,           /// scene-wide shading property
         int SHADOWS_ON,       /// scene-wide shading property
         int AO_ON,            /// scene-wide shading property
         int OUTLINE_ON,       /// material-specific shading property
         int REFLECTION_ON,    /// material-specific shading property
         int TRANSMISSION_ON>  /// material-specific shading property
static __device__ void shader_template(float3 prim_color, float3 N,
                                       float p_Ka, float p_Kd, float p_Ks,
                                       float p_phong_exp, float p_reflectivity,
                                       float p_opacity,
                                       float p_outline, float p_outlinewidth,
                                       int p_transmode) {
  PerRayData_radiance &prd = *(PerRayData_radiance *) getPRD<PerRayData_radiance>();
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();

  float3 hit_point = ray_origin + t_hit * ray_direction;
  float3 result = make_float3(0.0f);
  float3 phongcol = make_float3(0.0f);

  // add depth cueing / fog if enabled
  // use fog coordinate to modulate importance for AO rays, etc.
  float fogmod = 1.0f;
  if (FOG_ON && rtLaunch.scene.fog_mode != 0) {
    fogmod = fog_coord(hit_point);
  }

#if 1
  // don't render transparent surfaces if we've reached the max count
  // this implements the same logic as the -trans_max_surfaces argument
  // in the CPU version of Tachyon.
  if ((p_opacity < 1.0f) && (prd.transcnt < 1)) {
    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - p_opacity);
    new_prd.alpha = 1.0f;
    new_prd.result = rtLaunch.scene.bg_color;
    new_prd.depth = prd.depth; // don't increment recursion depth
    new_prd.transcnt = prd.transcnt - 1;
    if (new_prd.importance >= 0.001f &&
        new_prd.depth <= rtLaunch.max_depth) {

      float3 transray_direction = ray_direction;
      float3 transray_origin;
#if defined(TACHYON_USE_RAY_STEP)
#if defined(TACHYON_TRANS_USE_INCIDENT)
      // step the ray in the incident ray direction
      transray_origin = hit_point + TACHYON_RAY_STEP2;
#else
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      transray_origin = hit_point - TACHYON_RAY_STEP;
#endif
#else
      transray_origin = hit_point;
#endif

      // the values we store the PRD pointer in:
      uint32_t u0, u1;
      packPointer( &new_prd, u0, u1 );

      // send inherited aasample counter to CH for much better AO sampling
      // when we run multiple AA samples per-pass
      unsigned int p2 = optixGetPayload_2();

      optixTrace(rtLaunch.traversable,
                 transray_origin,
                 transray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 u0, u1,                        // PRD ptr in 2x uint32
                 p2);

#if defined(TACHYON_RAYSTATS)
      raystats2_buffer[launch_index].x++; // increment trans ray counter
#endif
    }
    prd.result = new_prd.result;
    return; // early-exit
  }
#endif

  // execute the object's texture function
  float3 col = prim_color; // XXX no texturing implemented yet

  // compute lighting from directional lights
  for (int i = 0; i < rtLaunch.lights.num_dir_lights; ++i) {
    float3 L = rtLaunch.lights.dir_lights[i];
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp,
                            col, phongcol, RT_DEFAULT_MAX);
  }

  // compute lighting from positional lights
  for (int i = 0; i < rtLaunch.lights.num_pos_lights; ++i) {
    float3 Lpos = rtLaunch.lights.pos_lights[i];
    float3 L = Lpos - hit_point;
    float shadow_tmax = length(L); // compute positional light shadow tmax
    L = normalize(L);
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp,
                            col, phongcol, shadow_tmax);
  }

  // add point light for camera headlight need for Oculus Rift HMDs,
  // equirectangular panorama images, and planetarium dome master images
  if (HEADLIGHT_ON && (rtLaunch.lights.headlight_mode != 0)) {
    float3 L = rtLaunch.cam.pos - hit_point;
    float shadow_tmax = length(L); // compute positional light shadow tmax
    L = normalize(L);
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp,
                            col, phongcol, shadow_tmax);
  }

  // add ambient occlusion diffuse lighting, if enabled
  if (AO_ON && rtLaunch.lights.ao_samples > 0) {
    result *= rtLaunch.lights.ao_direct;
    result += rtLaunch.lights.ao_ambient * col * p_Kd * shade_ambient_occlusion(hit_point, N, fogmod * p_opacity);
  }

  // add edge shading if applicable
  if (OUTLINE_ON && p_outline > 0.0f) {
    float edgefactor = dot(N, ray_direction);
    edgefactor *= edgefactor;
    edgefactor = 1.0f - edgefactor;
    edgefactor = 1.0f - powf(edgefactor, (1.0f - p_outlinewidth) * 32.0f);
    float outlinefactor = __saturatef((1.0f - p_outline) + (edgefactor * p_outline));
    result *= outlinefactor;
  }

  result += make_float3(p_Ka); // white ambient contribution
  result += phongcol;          // add phong highlights

#if 1
  // spawn reflection rays if necessary
  if (REFLECTION_ON && p_reflectivity > 0.0f) {
    // ray tree attenuation
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * p_reflectivity;
    new_prd.depth = prd.depth + 1;
    new_prd.transcnt = prd.transcnt;

    // shoot a reflection ray
    if (new_prd.importance >= 0.001f && new_prd.depth <= rtLaunch.max_depth) {
      float3 reflray_direction = reflect(ray_direction, N);

      float3 reflray_origin;
#if defined(TACHYON_USE_RAY_STEP)
      reflray_origin = hit_point + TACHYON_RAY_STEP;
#else
      reflray_origin = hit_point;
#endif

      // the values we store the PRD pointer in:
      uint32_t u0, u1;
      packPointer( &new_prd, u0, u1 );

      // send inherited aasample counter to CH for much better AO sampling
      // when we run multiple AA samples per-pass
      unsigned int p2 = optixGetPayload_2();

      optixTrace(rtLaunch.traversable,
                 reflray_origin,
                 reflray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 u0, u1,                        // PRD ptr in 2x uint32
                 p2);

#if defined(TACHYON_RAYSTATS)
      raystats2_buffer[launch_index].w++; // increment refl ray counter
#endif
      result += p_reflectivity * new_prd.result;
    }
  }
#endif

  // spawn transmission rays if necessary
  float alpha = p_opacity;

#if 1
  if (CLIP_VIEW_ON && (rtLaunch.clipview_mode == 2))
    sphere_fade_and_clip(hit_point, rtLaunch.cam.pos,
                         rtLaunch.clipview_start, rtLaunch.clipview_end, alpha);
#else
  if (CLIP_VIEW_ON && (rtLaunch.clipview_mode == 2)) {
    // draft implementation of a smooth "fade-out-and-clip sphere"
    float fade_start = 1.00f; // onset of fading
    float fade_end   = 0.20f; // fully transparent
    float camdist = length(hit_point - rtLaunch.cam.pos);

    // XXX we can omit the distance test since alpha modulation value is clamped
    // if (1 || camdist < fade_start) {
      float fade_len = fade_start - fade_end;
      alpha *= __saturatef((camdist - fade_start) / fade_len);
    // }
  }
#endif

#if 1
  // TRANSMISSION_ON: handles transparent surface shading, test is only
  // performed when the geometry has a known-transparent material
  // CLIP_VIEW_ON: forces check of alpha value for all geom as per transparent
  // material, since all geometry may become tranparent with the
  // fade+clip sphere active
  if ((TRANSMISSION_ON || CLIP_VIEW_ON) && alpha < 0.999f ) {
    // Emulate Tachyon/Raster3D's angle-dependent surface opacity if enabled
    if (p_transmode) {
      alpha = 1.0f + cosf(3.1415926f * (1.0f-alpha) * dot(N, ray_direction));
      alpha = alpha*alpha * 0.25f;
    }

    result *= alpha; // scale down current lighting by opacity

    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - alpha);
    new_prd.alpha = 1.0f;
    new_prd.result = rtLaunch.scene.bg_color;
    new_prd.depth = prd.depth + 1;
    new_prd.transcnt = prd.transcnt - 1;
    if (new_prd.importance >= 0.001f &&
        new_prd.depth <= rtLaunch.max_depth) {

      float3 transray_direction = ray_direction;
      float3 transray_origin;
#if defined(TACHYON_USE_RAY_STEP)
#if defined(TACHYON_TRANS_USE_INCIDENT)
      // step the ray in the incident ray direction
      transray_origin = hit_point + TACHYON_RAY_STEP2;
#else
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      transray_origin = hit_point - TACHYON_RAY_STEP;
#endif
#else
      transray_origin = hit_point;
#endif

      // the values we store the PRD pointer in:
      uint32_t u0, u1;
      packPointer( &new_prd, u0, u1 );

      // send inherited aasample counter to CH for much better AO sampling
      // when we run multiple AA samples per-pass
      unsigned int p2 = optixGetPayload_2();

      optixTrace(rtLaunch.traversable,
                 transray_origin,
                 transray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 u0, u1,                        // PRD ptr in 2x uint32
                 p2);

#if defined(TACHYON_RAYSTATS)
      raystats2_buffer[launch_index].x++; // increment trans ray counter
#endif
    }
    result += (1.0f - alpha) * new_prd.result;
    prd.alpha = alpha + (1.0f - alpha) * new_prd.alpha;
  }
#endif

  // add depth cueing / fog if enabled
  if (FOG_ON && fogmod < 1.0f) {
    result = fog_color(fogmod, result);
  }

  prd.result = result; // pass the color back up the tree
}



//
// OptiX closest hit and anyhit programs for radiance rays
//  

// general-purpose any-hit program, with all template options enabled,
// intended for shader debugging and comparison with the original
// Tachyon full_shade() code.
extern "C" __global__ void __closesthit__radiance_general() {
  // shading variables that need to be computed/set by primitive-specific code
  int matidx = 0;
  float3 shading_normal;
  float3 hit_color;
  const GeomSBTHG &sbtHG = *(const GeomSBTHG*) optixGetSbtDataPointer();

  // Handle normal and color computations according to primitive type
  unsigned int hit_kind = optixGetHitKind();
  OptixPrimitiveType hit_prim_type = optixGetPrimitiveType(hit_kind);


  // XXX eventually we need a full switch block for triangles/curves/custom
  if (hit_prim_type == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    get_shadevars_trimesh(sbtHG, hit_color, shading_normal, matidx);
  } else {
    // For OPTIX_PRIMITIVE_TYPE_CUSTOM we check the lowest 7 bits of 
    // hit_kind to determine our user-defined primitive type.
    // For peak traversal performance, calculation of surface normals 
    // colors, etc is deferred until CH/AH shading herein. 
    switch (hit_kind) {
      case RT_HIT_CONE: 
        get_shadevars_cone_array(sbtHG, shading_normal);
        break;

      case RT_HIT_CYLINDER: 
        get_shadevars_cylinder_array(sbtHG, shading_normal);
        break;

      case RT_HIT_RING: 
        get_shadevars_ring_array(sbtHG, shading_normal);
        break;

      case RT_HIT_SPHERE: 
        get_shadevars_sphere_array(sbtHG, shading_normal);
        break;
    }

    // Assign either per-primitive or uniform color
    if (sbtHG.prim_color != nullptr) {
      const int primID = optixGetPrimitiveIndex();
      hit_color = sbtHG.prim_color[primID];
    } else {
      hit_color = sbtHG.uniform_color;
    }

    matidx = sbtHG.materialindex;
  }

  const auto &mat = rtLaunch.materials[matidx];
  shader_template<1, 1, 1, 1, 1, 1, 1, 1>(hit_color, shading_normal,
                                          mat.ambient, mat.diffuse, 
                                          mat.specular, mat.shininess,
                                          mat.reflectivity, mat.opacity,
                                          mat.outline, mat.outlinewidth,
                                          mat.transmode);
}




