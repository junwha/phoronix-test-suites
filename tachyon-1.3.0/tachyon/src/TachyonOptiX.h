/*
 * TachyonOptiX.h - OptiX host-side RT engine APIs and data structures
 *
 *  $Id: TachyonOptiX.h,v 1.10 2021/12/11 03:37:20 johns Exp $
 */
/**
 *  \file TachyonOptiX.h
 *  \brief Tachyon ray tracing host side routines and internal APIs that 
 *         provide the core ray OptiX-based RTX-accelerated tracing engine.
 *         The major responsibilities of the core engine are to manage
 *         runtime RT pipeline construction and JIT-linked shaders 
 *         to build complete ray tracing pipelines, management of 
 *         RT engine state, and managing associated GPU hardware.
 *         Written for NVIDIA OptiX 7 and later.
 *
 *         The declarations and prototypes needed to drive
 *         the raytracer.  Third party driver code should only use the 
 *         functions in this header file to interface with the rendering engine.
 */

#ifndef TACHYONOPTIX_H
#define TACHYONOPTIX_H

#if defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
// OptiX headers require NOMINMAX be defined for Windows builds
#define NOMINMAX 1
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "WKFUtils.h"
#include "TachyonOptiXShaders.h"


struct ConeArray {
  std::vector<float3> base;
  std::vector<float3> apex;
  std::vector<float> baserad;
  std::vector<float> apexrad;
  float3 uniform_color;
  int materialindex;

  void addCone(const float3 &cbase, const float3 &capex, 
               const float cbaserad, const float capexrad) {
    base.push_back(cbase);
    apex.push_back(capex);
    baserad.push_back(cbaserad);
    apexrad.push_back(capexrad);
  }
    
};


struct CylinderArray {
  std::vector<float3> start;
  std::vector<float3> end;
  std::vector<float> radius;
  float3 uniform_color;
  int materialindex;

  void addCylinder(const float3 &cstart, const float3 &cend, const float crad) {
    start.push_back(cstart);
    end.push_back(cend);
    radius.push_back(crad);
  }
    
};


struct RingArray {
  std::vector<float3> center;
  std::vector<float3> normal;
  std::vector<float> inrad;
  std::vector<float> outrad;
  float3 uniform_color;
  int materialindex;

  void addRing(const float3 &ricenter, const float3 &rinorm,
               const float riinrad, const float rioutrad) {
    center.push_back(ricenter);
    normal.push_back(rinorm);
    inrad.push_back(riinrad);
    outrad.push_back(rioutrad);
  }
    
};


struct SphereArray {
  std::vector<float3> center;
  std::vector<float> radius;
  float3 uniform_color;
  int materialindex;

  void addSphere(const float3 &spcenter, const float &spradius) {
    center.push_back(spcenter);
    radius.push_back(spradius);
  }
    
};


struct TriangleMesh {
  std::vector<float3> vertex;
  std::vector<int3> index;
  float3 uniform_color;
  int materialindex;

  void addCube(const float3 &center, const float3 &s /* size */) {
    int firstVertexID = (int)vertex.size();
    vertex.push_back(center + make_float3(s.x*0.f, s.y*0.f, s.z*0.f));
    vertex.push_back(center + make_float3(s.x*1.f, s.y*0.f, s.z*0.f));
    vertex.push_back(center + make_float3(s.x*0.f, s.y*1.f, s.z*0.f));
    vertex.push_back(center + make_float3(s.x*1.f, s.y*1.f, s.z*0.f));
    vertex.push_back(center + make_float3(s.x*0.f, s.y*0.f, s.z*1.f));
    vertex.push_back(center + make_float3(s.x*1.f, s.y*0.f, s.z*1.f));
    vertex.push_back(center + make_float3(s.x*0.f, s.y*1.f, s.z*1.f));
    vertex.push_back(center + make_float3(s.x*1.f, s.y*1.f, s.z*1.f));
    int indices[] = {0,1,3, 2,3,0,
                     5,7,6, 5,6,4,
                     0,4,5, 0,5,1,
                     2,3,7, 2,7,6,
                     1,5,7, 1,7,3,
                     4,0,2, 4,2,6
                     };
    for (int i=0; i<12; i++)
      index.push_back(make_int3(indices[3*i+0] + firstVertexID,
                                indices[3*i+1] + firstVertexID,
                                indices[3*i+2] + firstVertexID));
  }
    
};


/// Several OptiX APIs make use of CUDA driver API pointer types
/// (CUdevicepointer) so it becomes worthwhile to manage these 
/// in a templated class supporting easy memory management for
/// vectors of special template types, and simple copies to and
/// from the associated CUDA device.
struct CUMemBuf {
  size_t sz    { 0 };           ///< device memory buffer size in bytes
  void  *d_ptr { nullptr };     ///< pointer to device memory buffer

  inline void * ptr() const { 
    return d_ptr; 
  }

  inline CUdeviceptr cu_dptr() const { 
    return (CUdeviceptr) d_ptr; 
  }

  /// query current buffer size in bytes
  size_t get_size(void) {
    return sz;
  }    

  /// (re)allocate buffer of requested size
  void set_size(size_t newsize) {
    // don't reallocate unless it is strictly necessary since 
    // CUDA memory management operations are very costly
    if (newsize != sz) {
      if (d_ptr) 
        this->free();

      sz = newsize;
      cudaMalloc((void**) &d_ptr, sz);
    }
  }
 
  /// free allocated memory
  void free() {
    cudaFree(d_ptr);
    d_ptr = nullptr;
    sz = 0;
  }


  //
  // synchronous copies that also allocate/resize the buffer 
  //
  template<typename T>
  void resize_upload(const std::vector<T> &vecT) {
    set_size(vecT.size()*sizeof(T));
    upload((const T*) vecT.data(), vecT.size());
  }
  
  template<typename T>
  void resize_upload(const T *t, size_t cnt) {
    set_size(cnt*sizeof(T));
    cudaMemcpy(d_ptr, (void *)t, cnt*sizeof(T), cudaMemcpyHostToDevice);
  }

  // 
  // synchronous DMA copies
  //
  template<typename T>
  void upload(const T *t, size_t cnt) {
    cudaMemcpy(d_ptr, (void *)t, cnt*sizeof(T), cudaMemcpyHostToDevice);
  }
  
  template<typename T>
  void download(T *t, size_t cnt) {
    cudaMemcpy((void *)t, d_ptr, cnt*sizeof(T), cudaMemcpyDeviceToHost);
  }


  //
  // asynchronous DMA copies
  //
  template<typename T>
  void upload(const T *t, size_t cnt, cudaStream_t stream) {
    cudaMemcpyAsync(d_ptr, (void *)t, 
                    cnt*sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  template<typename T>
  void download_async(T *t, size_t cnt, cudaStream_t stream) {
    cudaMemcpyAsync((void *)t, d_ptr, 
                    cnt*sizeof(T), cudaMemcpyDeviceToHost, stream);
  }
};


struct TachyonInstanceGroup {
  // 
  // primitive array buffers
  // 
  std::vector<ConeArray>   conearrays;
  std::vector<CylinderArray> cyarrays;
  std::vector<RingArray>     riarrays;
  std::vector<SphereArray>   sparrays;
  std::vector<TriangleMesh>  meshes;

  // GASes get rebuilt upon any scene geometry changes
  std::vector<HGRecordGroup> sbtHGRecGroups;
  CUMemBuf custprimsGASBuffer;             ///< final, compacted GAS
  CUMemBuf trimeshesGASBuffer;             ///< final, compacted GAS
  std::vector<float *> transforms;
};
 

class TachyonOptiX {
public: 
  enum ViewClipMode {
    RT_VIEWCLIP_NONE=0,                    ///< no frustum clipping
    RT_VIEWCLIP_PLANE=1,                   ///< cam/frustum front clipping plane
    RT_VIEWCLIP_SPHERE=2                   ///< cam/frustum clipping sphere
  };

  enum HeadlightMode { 
    RT_HEADLIGHT_OFF=0,                    ///< no VR headlight
    RT_HEADLIGHT_ON=1                      ///< VR headlight at cam center
  };

  enum FogMode  { 
    RT_FOG_NONE=0,                         ///< No fog
    RT_FOG_LINEAR=1,                       ///< Linear fog w/ Z-depth
    RT_FOG_EXP=2,                          ///< Exp fog w/ Z-depth
    RT_FOG_EXP2=3                          ///< Exp^2 fog w/ Z-depth
    // XXX should explicitly define radial/omnidirection fog types also
  };

  enum CameraType { 
    RT_PERSPECTIVE=0,                      ///< conventional perspective
    RT_ORTHOGRAPHIC=1,                     ///< conventional orthographic
    RT_CUBEMAP=2,                          ///< omnidirectional cubemap
    RT_DOME_MASTER=3,                      ///< planetarium dome master
    RT_EQUIRECTANGULAR=4,                  ///< omnidirectional lat/long
    RT_OCULUS_RIFT                         ///< VR HMD
  };

  enum Verbosity { 
    RT_VERB_MIN=0,                         ///< No console output
    RT_VERB_TIMING=1,                      ///< Output timing/perf data only
    RT_VERB_DEBUG=2                        ///< Output fully verbose debug info
  };

  enum BGMode { 
    RT_BACKGROUND_TEXTURE_SOLID=0,
    RT_BACKGROUND_TEXTURE_SKY_SPHERE=1,
    RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE=2 
  };

private:
  int context_created;                     ///< flag when context is valid
  CUcontext cudactx;                       ///< CUDA driver context for OptiX
  CUstream stream;                         ///< CUDA driver stream for OptiX
  OptixDeviceContext ctx;                  ///< OptiX context 
  OptixResult lasterr;                     ///< Last OptiX error code if any
  Verbosity verbose;                       ///< console perf/debugging output

  char shaderpath[8192];                   ///< path to OptiX shader PTX file

  wkf_timerhandle rt_timer;                ///< general purpose timer
  double time_ctx_create;                  ///< time taken to create ctx
  double time_ctx_setup;                   ///< time taken to setup/init ctx
  double time_ctx_validate;                ///< time for ctx compile+validate
  double time_ctx_AS_build;                ///< time for AS build
  double time_ctx_destroy_scene;           ///< time to destroy existing scene
  double time_ray_tracing;                 ///< time to trace the rays...
  double time_image_io;                    ///< time to write image(s) to disk

  //
  // OptiX framebuffers and associated state
  // 
  int width;                               ///< image width in pixels
  int height;                              ///< image height in pixels


  //
  // OptiX pipeline and shader compilations
  //

  // the PTX module that contains all device programs
  char * rt_ptx_code_string = {};
  OptixPipelineCompileOptions pipeCompOpts = {};
  OptixModule module;

  std::vector<OptixProgramGroup> exceptionPGs;
  std::vector<OptixProgramGroup> raygenPGs;
  std::vector<OptixProgramGroup> missPGs;
  std::vector<OptixProgramGroup> trimeshPGs;
  std::vector<OptixProgramGroup> custprimPGs;

  // SBT-associated GPU data structures
  // SBT must be entirely rebuilt upon any change to the rendering pipeline
  CUMemBuf exceptionRecordsBuffer;
  CUMemBuf raygenRecordsBuffer;
  CUMemBuf missRecordsBuffer;

  std::vector<TachyonInstanceGroup> sceneinstancegroups;

  std::vector<HGRecordGroup> hitgroupRecordGroups;
  CUMemBuf hitgroupRecordsBuffer;
  OptixShaderBindingTable sbt = {};

  // OptiX RT pipeline produced from all of raygen/miss/hitgroup PGs
  OptixPipeline pipe;

  // GASes get rebuilt upon any scene geometry changes
  CUMemBuf custprimsGASBuffer;             ///< final, compacted GAS
  CUMemBuf trimeshesGASBuffer;             ///< final, compacted GAS

  // IAS to combine all triangle, curve, and custom primitive GASes together
  CUMemBuf IASBuffer;                      ///< final, compacted IAS

  tachyonLaunchParams rtLaunch;            ///< host-side launch params
  CUMemBuf launchParamsBuffer;             ///< device-side launch params buffer
  CUMemBuf framebuffer;                    ///< device-side framebuffer
  CUMemBuf accumulation_buffer;            ///< device-side accumulation buffer


  //  
  // primitive array buffers
  //  
  std::vector<CUMemBuf>      coneAabbBuffers;
  std::vector<CUMemBuf>      coneBaseBuffers;
  std::vector<CUMemBuf>      coneApexBuffers;
  std::vector<CUMemBuf>      coneBaseRadBuffers;
  std::vector<CUMemBuf>      coneApexRadBuffers;
  std::vector<CUMemBuf>      conePrimColorBuffers;
  std::vector<ConeArray> conearrays;

  std::vector<CUMemBuf>      cyAabbBuffers;
  std::vector<CUMemBuf>      cyStartBuffers;
  std::vector<CUMemBuf>      cyEndBuffers;
  std::vector<CUMemBuf>      cyRadiusBuffers;
  std::vector<CUMemBuf>      cyPrimColorBuffers;
  std::vector<CylinderArray> cyarrays;

  std::vector<CUMemBuf>      riAabbBuffers;
  std::vector<CUMemBuf>      riCenterBuffers;
  std::vector<CUMemBuf>      riNormalBuffers;
  std::vector<CUMemBuf>      riInRadiusBuffers;
  std::vector<CUMemBuf>      riOutRadiusBuffers;
  std::vector<CUMemBuf>      riPrimColorBuffers;
  std::vector<RingArray>     riarrays;

  std::vector<CUMemBuf>      spAabbBuffers;
  std::vector<CUMemBuf>      spCenterBuffers;
  std::vector<CUMemBuf>      spRadiusBuffers;
  std::vector<CUMemBuf>      spPrimColorBuffers;
  std::vector<SphereArray>   sparrays;
  
  std::vector<CUMemBuf>      triMeshVertBuffers;
  std::vector<CUMemBuf>      triMeshIdxBuffers;
  std::vector<CUMemBuf>      triMeshVertNormalBuffers;
  std::vector<CUMemBuf>      triMeshVertPackedNormalBuffers;
  std::vector<CUMemBuf>      triMeshVertColor3fBuffers;
  std::vector<CUMemBuf>      triMeshVertColor4uBuffers;
  std::vector<CUMemBuf>      triMeshPrimColorBuffers;
  std::vector<TriangleMesh>  meshes;
  

  //
  // OptiX shader state variables and the like
  //
  unsigned int scene_max_depth;            ///< max ray recursion depth
  int scene_max_trans;                     ///< max transmission ray depth

  float scene_epsilon;                     ///< scene-wide epsilon value

  int clipview_mode;                       ///< VR fade+clipping sphere/plane
  float clipview_start;                    ///< VR fade+clipping sphere/plane
  float clipview_end;                      ///< VR fade+clipping sphere/plane

  int headlight_mode;                      ///< VR HMD headlight

  float ao_ambient;                        ///< AO ambient lighting scalefactor
  float ao_direct;                         ///< AO direct lighting scalefactor
  float ao_maxdist;                        ///< AO maximum occlusion distance

  // shadow rendering mode
  int shadows_enabled;                     ///< shadow enable/disable flag

  float cam_pos[3];                        ///< camera position
  float cam_U[3];                          ///< camera ONB "right" direction
  float cam_V[3];                          ///< camera ONB "up" direction
  float cam_W[3];                          ///< camera ONB "view" direction
  float cam_zoom;                          ///< camera zoom factor
  float cam_stereo_eyesep;                 ///< stereo eye separation
  float cam_stereo_convergence_dist;       ///< stereo convergence distance

  int cam_dof_enabled;                     ///< DoF enable/disable flag
  float cam_dof_focal_dist;                ///< DoF focal distance
  float cam_dof_fnumber;                   ///< DoF f/stop number

  CameraType camera_type;                  ///< camera type

  int ext_aa_loops;                        ///< Multi-pass AA iterations
  int aa_samples;                          ///< AA samples per pixel
  int ao_samples;                          ///< AO samples per pixel

  // background color and/or gradient parameters
  BGMode scene_background_mode;            ///< which miss program to use...
  float scene_bg_color[3];                 ///< background color
  float scene_bg_grad_top[3];              ///< background gradient top color
  float scene_bg_grad_bot[3];              ///< background gradient bottom color
  float scene_gradient[3];                 ///< background gradient vector
  float scene_gradient_topval;             ///< background gradient top value
  float scene_gradient_botval;             ///< background gradient bot value
  float scene_gradient_invrange;           ///< background gradient rcp range

  // clipping plane/sphere parameters
  int clip_mode;                           ///< clip mode
  float clip_start;                        ///< clip start (Z or radial dist)
  float clip_end;                          ///< clip end (Z or radial dist)

  // fog / depth cueing parameters
  int fog_mode;                            ///< fog mode
  float fog_start;                         ///< fog start
  float fog_end;                           ///< fog end
  float fog_density;                       ///< fog density

  std::vector<rt_material> materialcache;  ///< cache of materials
  std::vector<int> materialvalid;          ///< material slot valid
  int regen_optix_materials;               ///< flag to force re-upload to GPU
  CUMemBuf materialsBuffer;                ///< device-side materials buffer

  std::vector<rt_directional_light> directional_lights; ///< list of directional lights
  std::vector<rt_positional_light> positional_lights;   ///< list of positional lights
  int regen_optix_lights;                  ///< flag to force re-upload to GPU
  CUMemBuf directionalLightsBuffer;        ///< device-side dir light buffer
  CUMemBuf positionalLightsBuffer;         ///< device-side pos light buffer


  //
  // Scene and geometric primitive counters
  //

  // state variables to hold scene geometry
  int scene_created;

  // cylinder array primitive
  long cylinder_array_cnt;                 ///< number of cylinder in scene

  // color-per-cylinder array primitive
  long cylinder_array_color_cnt;           ///< number of cylinders in scene


  // color-per-ring array primitive
  long ring_array_color_cnt;              ///< number of rings in scene


  // sphere array primitive
  long sphere_array_cnt;                  ///< number of spheres in scene

  // color-per-sphere array primitive
  long sphere_array_color_cnt;            ///< number of spheres in scene


  // triangle mesh primitives of various types
  long tricolor_cnt;                      ///< number of triangles scene
  long trimesh_c4u_n3b_v3f_cnt;           ///< number of triangles scene
  long trimesh_n3b_v3f_cnt;               ///< number of triangles scene
  long trimesh_n3f_v3f_cnt;               ///< number of triangles scene
  long trimesh_v3f_cnt;                   ///< number of triangles scene


  //
  // Internal methods
  //

  /// sub-init routines to compile, link, the complete RT pipeline
  int read_ptx_src(const char *ptxfilename, char **ptxstring);
 
  void context_create_exception_pgms(void);
  void context_destroy_exception_pgms(void);

  void context_create_raygen_pgms(void);
  void context_destroy_raygen_pgms(void);

  void context_create_miss_pgms(void);
  void context_destroy_miss_pgms(void);

  void context_create_hitgroup_pgms(void);
  void context_destroy_hitgroup_pgms(void);

  void context_create_intersection_pgms(void);
  void context_destroy_intersection_pgms(void);

  void context_create_module(void);
  void context_destroy_module(void);

  int regen_optix_pipeline;
  void context_create_pipeline(void);
  void context_destroy_pipeline(void);

  /// Shader binding table management routines
  int regen_optix_sbt;
  void context_create_SBT(void);
  void context_destroy_SBT(void);

  /// Geometry AS builder methods
  OptixTraversableHandle build_custprims_GAS(void);
  void free_custprims_GAS(void);

  OptixTraversableHandle build_trimeshes_GAS(void);
  void free_trimeshes_GAS(void);

  void build_IAS(void);
  void free_IAS(void);

public:
  TachyonOptiX();
  ~TachyonOptiX(void);

  /// static methods for querying OptiX-supported GPU hardware independent
  /// of whether we actually have an active context.
  static int device_list(int **, char ***);
  static int device_count(void);
  static unsigned int optix_version(void);

  void log_callback(unsigned int level, const char *tag, const char *msg);

  /// diagnostic info routines
  void print_internal_struct_info(void);

  /// programmatically set verbosity
  void set_verbose_mode(TachyonOptiX::Verbosity mode) {
    verbose = mode;
  } 

  /// check environment variables that modify verbose output
  void check_verbose_env();

  /// initialize the OptiX context 
  void create_context(void);

  /// poorly named routine to initialize key RT user settings...
  void setup_context(int width, int height);
  
  /// report various context statistics for memory leak debugging, etc.
  void report_context_stats(void);

  /// shadows
  void shadows_on(int onoff) { shadows_enabled = (onoff != 0); }

  /// antialiasing (samples > 1 == on)
  void set_aa_samples(int cnt) { aa_samples = cnt; }

  /// set the camera projection mode
  void set_camera_type(CameraType m) { 
    if (camera_type != m) {
      camera_type = m; 
      regen_optix_pipeline=1; // this requires changing the raygen program
    }
  }

  /// set the camera position
  void set_camera_pos(const float *pos) { 
    memcpy(cam_pos, pos, sizeof(cam_pos)); 
  }

  /// set the camera ONB vector orientation frame
  void set_camera_ONB(const float *U, const float *V, const float *W) { 
    memcpy(cam_U, U, sizeof(cam_U)); 
    memcpy(cam_V, V, sizeof(cam_V)); 
    memcpy(cam_W, W, sizeof(cam_W)); 
  }

  /// set camera orientation to look "at" a point in space, with a given "up"
  /// direction (camera ONB "V" vector), the remaining ONB vectors are computed
  /// from the camera position and "at" point in space.
  void set_camera_lookat(const float *at, const float *V);

  /// set camera zoom factor
  void set_camera_zoom(float zoomfactor) { cam_zoom = zoomfactor; }

  /// set stereo eye separation
  void set_camera_stereo_eyesep(float eyesep) { cam_stereo_eyesep = eyesep; }
  
  /// set stereo convergence distance
  void set_camera_stereo_convergence_dist(float dist) {
    cam_stereo_convergence_dist = dist;
  }

  /// depth of field on/off
  void dof_on(int onoff) { 
    if (cam_dof_enabled != (onoff != 0)) {
      cam_dof_enabled = (onoff != 0); 
      regen_optix_pipeline=1; // this requires changing the raygen program
    }
  }

  /// set depth of field focal plane distance
  void set_camera_dof_focal_dist(float d) { cam_dof_focal_dist = d; }

  /// set depth of field f/stop number
  void set_camera_dof_fnumber(float n) { cam_dof_fnumber = n; }

  /// ambient occlusion (samples > 1 == on)
  void set_ao_samples(int cnt) { ao_samples = cnt; }

  /// set AO ambient lighting factor
  void set_ao_ambient(float aoa) { ao_ambient = aoa; }

  /// set AO direct lighting factor
  void set_ao_direct(float aod) { ao_direct = aod; }

  /// set AO maximum occlusion distance
  void set_ao_maxdist(float dist) { ao_maxdist = dist; }

  void set_bg_mode(BGMode m) {
    if (scene_background_mode != m) {
      scene_background_mode = m;
      regen_optix_pipeline=1; // this requires changing the miss program
    }
  }
  void set_bg_color(float *rgb) { memcpy(scene_bg_color, rgb, sizeof(scene_bg_color)); }
  void set_bg_color_grad_top(float *rgb) { memcpy(scene_bg_grad_top, rgb, sizeof(scene_bg_grad_top)); }
  void set_bg_color_grad_bot(float *rgb) { memcpy(scene_bg_grad_bot, rgb, sizeof(scene_bg_grad_bot)); }
  void set_bg_gradient(float *vec) { memcpy(scene_gradient, vec, sizeof(scene_gradient)); }
  void set_bg_gradient_topval(float v) { scene_gradient_topval = v; }
  void set_bg_gradient_botval(float v) { scene_gradient_botval = v; }

  /// set camera clipping plane/sphere mode and parameters
  void set_clip_sphere(ViewClipMode mode, float start, float end) {
    clip_mode = mode;
    clip_start = start;
    clip_end = end;
  }

  /// set depth cueing mode and parameters
  void set_cue_mode(FogMode mode, float start, float end, float density) {
    fog_mode = mode;
    fog_start = start;
    fog_end = end;
    fog_density = density;
  }

  void init_materials();
  void add_material(int matindex, float ambient, float diffuse,
                    float specular, float shininess, float reflectivity,
                    float opacity, float outline, float outlinewidth, 
                    int transmode);
#if 0
  void set_material(RTgeometryinstance instance, int matindex, 
                    const float *uniform_color, int hwtri=0);
#endif

  void set_clipview_mode(int mode) { 
    clipview_mode = mode;
  };
  void set_headlight_onoff(int onoff) { 
    headlight_mode = (onoff==1) ? RT_HEADLIGHT_ON : RT_HEADLIGHT_OFF; 
  };
  void add_directional_light(const float *dir, const float *color);
  void add_positional_light(const float *pos, const float *color);
  void clear_all_lights() { 
    directional_lights.clear(); 
    positional_lights.clear(); 
    regen_optix_lights=1;  
  }

  void update_rendering_state(int interactive);

  void framebuffer_config(int fbwidth, int fbheight, int interactive);
  void framebuffer_resize(int fbwidth, int fbheight);
  void framebuffer_get_size(int &fbwidth, int &fbheight) { 
    fbwidth=width; 
    fbheight=height;
  }
  void framebuffer_download_rgb4u(unsigned char *imgrgb4u);
  void framebuffer_destroy(void);

  void render_compile_and_validate(void);
  void render(); 

  void destroy_scene(void);
  void destroy_context(void);



  //
  // Geometric primitive APIs
  // 

  /// Create geometry instance group
  int create_geom_instance_group();
  int finalize_geom_instance_group(int idx);
  int destroy_geom_instance_group(int idx);
//  int set_geom_instance_group_xforms(int idx, int n, float [][16]);


  //
  // XXX short-term host API hacks to facilitate early bring-up and testing
  //
  void add_conearray(int idx, ConeArray & model, int matidx);
  void add_cylarray(int idx, CylinderArray & model, int matidx);
  void add_ringarray(int idx, RingArray & model, int matidx);
  void add_spherearray(int idx, SphereArray & model, int matidx);
  void add_trimesh(int idx, TriangleMesh & model, int matidx);


  //
  // XXX short-term host API hacks to facilitate early bring-up and testing
  //
  void add_conearray(ConeArray & model, int matidx);
  void add_cylarray(CylinderArray & model, int matidx);
  void add_ringarray(RingArray & model, int matidx);
  void add_spherearray(SphereArray & model, int matidx);
  void add_trimesh(TriangleMesh & model, int matidx);

#if 0
  // old APIs that assumed pre-flattening of the scene graph prior to 
  // rendering, and very very early triangle soup oriented RTX triangle APIs
  void cylinder_array(Matrix4 *wtrans, float rscale, const float *uniform_color,
                      int cylnum, const float *points, int matindex);

  void cylinder_array_color(Matrix4 *wtrans, float rscale, int cylnum, 
                            const float *points, const float *radii, 
                            const float *colors, int matindex);

  void ring_array_color(Matrix4 & wtrans, float rscale, int rnum, 
                        const float *centers, const float *norms, 
                        const float *radii, const float *colors, int matindex);

  void sphere_array(Matrix4 *wtrans, float rscale, const float *uniform_color,
                    int spnum, const float *centers, const float *radii, 
                    int matindex);

  void sphere_array_color(Matrix4 & wtrans, float rscale, int spnum, 
                          const float *centers, const float *radii, 
                          const float *colors, int matindex);

  void tricolor_list(Matrix4 & wtrans, 
                     int numtris, const float *vnc, int matindex);


  void trimesh_c4n3v3(Matrix4 & wtrans, 
                      int numverts, const float *cnv, 
                      int numfacets, const int * facets, 
                      int matindex);


  void trimesh_c4u_n3b_v3f(Matrix4 & wtrans, const unsigned char *c, 
                           const signed char *n, const float *v, 
                           int numfacets, int matindex);


  void trimesh_c4u_n3f_v3f(Matrix4 & wtrans, const unsigned char *c, 
                           const float *n, const float *v, 
                           int numfacets, int matindex);


  void trimesh_n3b_v3f(Matrix4 & wtrans, const float *uniform_color, 
                       const signed char *n, const float *v, 
                       int numfacets, int matindex);


  void trimesh_n3f_v3f(Matrix4 & wtrans, const float *uniform_color, 
                       const float *n, const float *v, 
                       int numfacets, int matindex);


  void trimesh_v3f(Matrix4 & wtrans, const float *uniform_color, 
                   const float *v, int numfacets, int matindex);


  void tristrip(Matrix4 & wtrans, 
                int numverts, const float * cnv,
                int numstrips, const int *vertsperstrip,
                const int *facets, int matindex);
#endif
}; 



#endif

