/*
 * TachyonOptiXShaders.h - prototypes for OptiX PTX shader routines 
 *
 *  $Id: TachyonOptiXShaders.h,v 1.23 2021/05/04 04:04:03 johns Exp $
 */
/**
 *  \file TachyonOptiXShaders.h
 *  \brief Tachyon ray tracing engine core routines and data structures
 *         compiled to PTX for runtime JIT to build complete ray tracing 
 *         pipelines.  Key data structures defined here are shared both by
 *         the compiled PTX core ray tracing routines, and by the host code
 *         that assembles the complete ray tracing pipeline and launches
 *         the pipeline kernels.
 *         Written for NVIDIA OptiX 7 and later.
 */

#ifndef TACHYONOPTIXSHADERS_H
#define TACHYONOPTIXSHADERS_H

// Compile-time flag for collection and reporting of ray statistics
#if 0
#define TACHYON_RAYSTATS 1
#endif


//
// Constants shared by both host and device code
//
#define RT_DEFAULT_MAX 1e27f


//
// Vector math helper routines
//

//
// float2 vector operators
//
inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator+(const float2& a, const float s) {
  return make_float2(a.x + s, a.y + s);
}

inline __host__ __device__ float2 operator-(const float2& a, const float2& b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator-(const float2& a, const float s) {
  return make_float2(a.x - s, a.y - s);
}

inline __host__ __device__ float2 operator-(const float s, const float2& a) {
  return make_float2(s - a.x, s - a.y);
}

inline __host__ __device__ float2 operator*(const float2& a, const float2& b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float2 operator*(const float s, const float2& a) {
  return make_float2(a.x * s, a.y * s);
}

inline __host__ __device__ float2 operator*(const float2& a, const float s) {
  return make_float2(a.x * s, a.y * s);
}

inline __host__ __device__ void operator*=(float2& a, const float s) {
  a.x *= s; a.y *= s;
}

inline __host__ __device__ float2 operator/(const float s, const float2& a) {
  return make_float2(s/a.x, s/a.y);
}



//
// float3 vector operators
//
inline __host__ __device__ float3 make_float3(const float s) {
  return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float4& a) {
  return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float3 operator-(const float3& a) {
  return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ void operator+=(float3& a, const float3& b) {
  a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ float3 operator*(const float3& a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(float s, const float3 & a) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}

inline __host__ __device__ float3 operator*(const float3& a, const float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ void operator*=(float3& a, const float s) {
  a.x *= s; a.y *= s; a.z *= s;
}

inline __host__ __device__ void operator*=(float3& a, const float3 &b) {
  a.x *= b.x; a.y *= b.y; a.z *= b.z;
}


//
// float4 vector operators
//
inline __host__ __device__ float4 make_float4(const float3& a, const float b) {
  return make_float4(a.x, a.y, a.z, b);
}

inline __host__ __device__ void operator+=(float4& a, const float4& b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}


//
// operators with subsequent type conversions
//
inline __host__ __device__ float3 operator*(char4 a, const float s) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}

inline __host__ __device__ float3 operator*(uchar4 a, const float s) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}


//
// math fctns...
//
inline __host__ __device__ float3 fabsf(const float3& a) {
  return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

inline __host__ __device__ float3 fmaxf(const float3& a, const float3& b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ float fmaxf(const float3& a) {
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

inline __host__ __device__ float dot(const float3 & a, const float3 & b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float dot(const float4 & a, const float4 & b) {
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline __host__ __device__ float length(const float3 & v) {
  return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 normalize(const float3 & v) {
#if defined(__CUDACC__) || defined(__NVCC__)
  float invlen = rsqrtf(dot(v, v));
#else
  float invlen = 1.0f / sqrtf(dot(v, v));
#endif
  float3 out;
  out.x = v.x * invlen;
  out.y = v.y * invlen;
  out.z = v.z * invlen;
  return out;
}

inline __host__ __device__ float3 normalize_len(const float3 v, float &l) {
  l = length(v);
  float invlen = 1.0f / l;
  float3 out;
  out.x = v.x * invlen;
  out.y = v.y * invlen;
  out.z = v.z * invlen;
  return out;
}

inline __host__ __device__ float3 normalize_invlen(const float3 v, float &invlen) {
#if defined(__CUDACC__) || defined(__NVCC__)
  invlen = rsqrtf(dot(v, v));
#else
  invlen = 1.0f / sqrtf(dot(v, v));
#endif
  float3 out;
  out.x = v.x * invlen;
  out.y = v.y * invlen;
  out.z = v.z * invlen;
  return out;
}

inline __host__ __device__ float3 cross(const float3 & a, const float3 & b) {
  float3 out;
  out.x =  a.y * b.z - b.y * a.z;
  out.y = -a.x * b.z + b.x * a.z;
  out.z =  a.x * b.y - b.x * a.y;
  return out;
}

inline __host__ __device__ float3 reflect(const float3& i, const float3& n) {
  return i - 2.0f * n * dot(n, i);
}

inline __host__ __device__ float3 faceforward(const float3& n, const float3& i, const float3& nref) {
  return n * copysignf(1.0f, dot(i, nref));
}



//
// Beginning of OptiX data structures
//

// Enable reversed traversal of any-hit rays for shadows/AO.
// This optimization yields a 20% performance gain in many cases.
// #define USE_REVERSE_SHADOW_RAYS 1

// Use reverse rays by default rather than only when enabled interactively
// #define USE_REVERSE_SHADOW_RAYS_DEFAULT 1
enum RtShadowMode { RT_SHADOWS_OFF=0,        ///< shadows disabled
                    RT_SHADOWS_ON=1,         ///< shadows on, std. impl.
                    RT_SHADOWS_ON_REVERSE=2  ///< any-hit traversal reversal
                  };

enum RayType { RT_RAY_TYPE_RADIANCE=0,      ///< normal radiance rays
               RT_RAY_TYPE_SHADOW=1,        ///< shadow probe/AO rays
               RT_RAY_TYPE_COUNT };         ///< total count of ray types

//
// OptiX 7.x geometry type-associated "hit kind" enums
//
enum RtHitKind { RT_HIT_HWTRIANGLE=0,       ///< RTX triangle
                 RT_HIT_CONE,               ///< custom prim cone
                 RT_HIT_CYLINDER,           ///< custom prim cyliner
                 RT_HIT_RING,               ///< custom prim ring
                 RT_HIT_SPHERE,             ///< custom prim sphere
                 RT_HIT_COUNT };      

// Enums used for custom primitive PGM indexing in SBT + GAS
enum RtCustPrim { RT_PRIM_CONE=0,
                  RT_PRIM_CYLINDER,
                  RT_PRIM_RING,
                  RT_PRIM_SPHERE,
                  RT_PRIM_COUNT };
                   



/// structure containing material properties
typedef struct {
  float ambient;
  float diffuse;
  float specular;
  float shininess;
  float reflectivity;
  float opacity;
  float outline;
  float outlinewidth;
  int transmode;
  int ind;
} rt_material;


//
// Lighting data structures
//
typedef struct {
  float3 dir;
//  float3 color; // not yet used
} rt_directional_light;

typedef struct {
  float3 pos;
//  float3 color; // not yet used
} rt_positional_light;


struct ConeArraySBT {
  float3 *base;
  float3 *apex;
  float  *baserad;
  float  *apexrad;
};

struct CylinderArraySBT {
  float3 *start;
  float3 *end;
  float  *radius;
};

struct RingArraySBT {
  float3 *center;
  float3 *norm;
  float  *inrad;
  float  *outrad; 
};

struct SphereArraySBT {
  float3 *center;
  float  *radius;
};

struct TriMeshSBT {
  float3 *vertex;
  int3 *index;
  float3 *normals;
  uint4 *packednormals;    ///< packed normals: ng [n0 n1 n2]
  float3 *vertexcolors3f; 
  uchar4 *vertexcolors4u;  ///< unsigned char color representation
};

struct GeomSBTHG {
  float3 *prim_color;      ///< optional per-primitive color array
  float3 uniform_color;    ///< uniform color for entire sphere array
  int materialindex;       ///< material index for this array

  union {
    ConeArraySBT cone;
    CylinderArraySBT cyl;
    RingArraySBT ring;
    SphereArraySBT sphere;
    TriMeshSBT trimesh;
  };
};


/// SBT record for an exception program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) ExceptionRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

/// SBT record for a raygen program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

/// SBT record for a miss program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

/// SBT record for a hitgroup program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HGRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  GeomSBTHG data;
};


// Store all hitgroup records for a given geometry together for 
// simpler dynamic updates.  At present, we have pairs of records,
// for radiance and shadow rayss.  Records differ only in their header.
// Each HGRecordGroup contains RT_RAY_TYPE_COUNT HGRecords, so when querying
// the size of any vector containers or other data structures to count total
// hitgroup records, we need to remember to multiply by RT_RAY_TYPE_COUNT.
struct HGRecordGroup {
  HGRecord radiance;
  HGRecord shadow;
};


struct tachyonLaunchParams {
  struct {
    int2 size;
    int subframe_index { 0 };
    uchar4 *framebuffer;
    float4 *accumulation_buffer;
#if defined(TACHYON_RAYSTATS)
    uint4 *raystats1_buffer;     ///< x=prim, y=shad-dir, z=shad-ao, w=miss
    uint4 *raystats2_buffer;     ///< x=trans, y=trans-skip, z=?, w=refl
#endif
  } frame;

  struct {
    float3 bg_color;             ///< miss background color
    float3 bg_color_grad_top;    ///< miss background sky gradient (top)
    float3 bg_color_grad_bot;    ///< miss background sky gradient (bottom)
    float3 gradient;             ///< miss background sky gradient direction
    float  gradient_topval;      ///< miss background sky gradient top value
    float  gradient_botval;      ///< miss background sky gradient bottom value
    float  gradient_invrange;    ///< miss background sky gradient inverse range
    int    fog_mode;             ///< fog type (or off)
    float  fog_start;            ///< radial/linear fog start distance
    float  fog_end;              ///< radial/linear fog end/max distance
    float  fog_density;          ///< exponential fog density
    float  epsilon;              ///< global epsilon value
  } scene;

  struct {
    int shadows_enabled;
    int ao_samples;
    float ao_ambient;
    float ao_direct;
    float ao_maxdist;
    int headlight_mode;
    int num_dir_lights;
    float3 *dir_lights;          ///< list of directional light directions
    int num_pos_lights;
    float3 *pos_lights;          ///< list of positional light positions
  } lights;

  struct {
    float3 pos;
    float3 U;
    float3 V;
    float3 W;
    float zoom;
    float stereo_eyesep;
    float stereo_convergence_dist;
    int   dof_enabled;
    float dof_aperture_rad;
    float dof_focal_dist;
  } cam;

  // VR HMD fade+clipping plane/sphere
  int clipview_mode;
  float clipview_start;
  float clipview_end;

  rt_material *materials;

  int max_depth;
  int max_trans;
  int aa_samples;
  int accum_count;
  float accumulation_normalization_factor;

  OptixTraversableHandle traversable;
};


//
// Methods for packing normals into a 4-byte quantity, such as a
// [u]int or [u]char4, and similar.  See JCGT article by Cigolle et al.,
// "A Survey of Efficient Representations for Independent Unit Vectors",
// J. Computer Graphics Techniques 3(2), 2014.
// http://jcgt.org/published/0003/02/01/
//

#if 1

//
// oct32: 32-bit octahedral normal encoding using [su]norm16x2 quantization
// Meyer et al., "On Floating Point Normal Vectors", In Proc. 21st
// Eurographics Conference on Rendering.
//   http://dx.doi.org/10.1111/j.1467-8659.2010.01737.x
// Others:
// https://twitter.com/Stubbesaurus/status/937994790553227264
// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding
//
static __host__ __device__ __inline__ float3 OctDecode(float2 projected) {
  float3 n = make_float3(projected.x,
                         projected.y,
                         1.0f - (fabsf(projected.x) + fabsf(projected.y)));
  if (n.z < 0.0f) {
    float oldX = n.x;
    n.x = copysignf(1.0f - fabsf(n.y), oldX);
    n.y = copysignf(1.0f - fabsf(oldX), n.y);
  }

  return n;
}

//
// XXX TODO: implement a high-precision OctPEncode() variant, based on
//           floored snorms and an error minimization scheme using a
//           comparison of internally decoded values for least error
//

static __host__ __device__ __inline__ float2 OctEncode(float3 n) {
  const float invL1Norm = 1.0f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
  float2 projected;
  if (n.z < 0.0f) {
    projected = 1.0f - make_float2(fabsf(n.y), fabsf(n.x)) * invL1Norm;
    projected.x = copysignf(projected.x, n.x);
    projected.y = copysignf(projected.y, n.y);
  } else {
    projected = make_float2(n.x, n.y) * invL1Norm;
  }

  return projected;
}


static __host__ __device__ __inline__ uint convfloat2uint32(float2 f2) {
  f2 = f2 * 0.5f + 0.5f;
  uint packed;
  packed = ((uint) (f2.x * 65535)) | ((uint) (f2.y * 65535) << 16);
  return packed;
}

static __host__ __device__ __inline__ float2 convuint32float2(uint packed) {
  float2 f2;
  f2.x = (float)((packed      ) & 0x0000ffff) / 65535;
  f2.y = (float)((packed >> 16) & 0x0000ffff) / 65535;
  return f2 * 2.0f - 1.0f;
}


static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  float2 octf2 = OctEncode(normal);
  return convfloat2uint32(octf2);
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  float2 octf2 = convuint32float2(packed);
  return OctDecode(octf2);
}

#elif 0

//
// snorm10x3: signed 10-bit-per-component scalar unit real representation
// Better representation than unorm.
// Supported by most fixed-function graphics hardware.
// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_snorm.txt
//   i=round(clamp(r,-1,1) * (2^(b-1) - 1)
//   r=clamp(i/(2^(b-1) - 1), -1, 1)
//

#elif 1

// OpenGL GLbyte signed quantization scheme
//   i = r * (2^b - 1) - 0.5;
//   r = (2i + 1)/(2^b - 1)
static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  const float3 N = normal * 127.5f - 0.5f;
  const char4 packed = make_char4(N.x, N.y, N.z, 0);
  return *((uint *) &packed);
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  char4 c4norm = *((char4 *) &packed);

  // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  // float = (2c+1)/(2^8-1)
  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  float3 N = c4norm * cn2f + ci2f;

  return N;
}

#endif


#endif
