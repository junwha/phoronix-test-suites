/*
 * TachyonOptiX.cu - OptiX host-side RT engine implementation
 *
 *  $Id: TachyonOptiX.cu,v 1.15 2021/12/11 03:37:20 johns Exp $
 */
/**
 *  \file TachyonOptiX.cu
 *  \brief Tachyon ray tracing host side routines and internal APIs that 
 *         provide the core ray OptiX-based RTX-accelerated tracing engine.
 *         The major responsibilities of the core engine are to manage
 *         runtime RT pipeline construction and JIT-linked shaders
 *         to build complete ray tracing pipelines, management of 
 *         RT engine state, and managing associated GPU hardware.
 *         Written for NVIDIA OptiX 7 and later.
 */

#include "TachyonOptiX.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "TachyonOptiXShaders.h"

#include "ProfileHooks.h"

#if 0
#define DBG()
#else
#define DBG() if (verbose == RT_VERB_DEBUG) { printf("TachyonOptiX) %s\n", __func__); }
#endif


// constructor...
TachyonOptiX::TachyonOptiX(void) {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::TachyonOptiX()", 0);
  rt_timer = wkf_timer_create(); // create and initialize timer
  wkf_timer_start(rt_timer);

  lasterr = OPTIX_SUCCESS;      // begin with no error state set

  context_created = 0;         // no context yet
  cudactx = 0;                 // take over current CUDA context, if not set
  stream = 0;                  // stream 0
  ctx = nullptr;               // no valid context yet
  pipe = nullptr;              // no valid pipeline
  scene_created = 0;           // scene has not been created
  strcpy(shaderpath, "TachyonOptiXShaders.ptx"); // set default shader path

  // clear timers
  time_ctx_setup = 0.0;
  time_ctx_validate = 0.0;
  time_ctx_AS_build = 0.0;
  time_ray_tracing = 0.0;
  time_image_io = 0.0;

  // set default scene background state
  scene_background_mode = RT_BACKGROUND_TEXTURE_SOLID;
  memset(scene_bg_color,    0, sizeof(scene_bg_color));
  memset(scene_bg_grad_top, 0, sizeof(scene_bg_grad_top));
  memset(scene_bg_grad_bot, 0, sizeof(scene_bg_grad_bot));
  memset(scene_gradient,    0, sizeof(scene_gradient));
  scene_gradient_topval = 1.0f;
  scene_gradient_botval = -scene_gradient_topval;
  // this has to be recomputed prior to rendering when topval/botval change
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);

  camera_type = RT_PERSPECTIVE;
  float tmp_pos[3] = { 0.0f, 0.0f, -1.0f };
  float tmp_U[3] = { 1.0f, 0.0f, 0.0f };
  float tmp_V[3] = { 0.0f, 1.0f, 0.0f };
  float tmp_W[3] = { 0.0f, 0.0f, 1.0f };
  memcpy(cam_pos, tmp_pos, sizeof(cam_pos));
  memcpy(cam_U, tmp_U, sizeof(cam_U));
  memcpy(cam_V, tmp_V, sizeof(cam_V));
  memcpy(cam_W, tmp_W, sizeof(cam_W));
  cam_zoom = 1.0f;
  cam_stereo_eyesep = 0.06f;
  cam_stereo_convergence_dist = 2.0f;
  cam_dof_enabled = 0;               // disable DoF by default
  cam_dof_focal_dist = 2.0f;
  cam_dof_fnumber = 64.0f;

  clipview_mode = RT_VIEWCLIP_NONE;  // VR HMD fade+clipping plane/sphere
  clipview_start = 1.0f;             // VR HMD fade+clipping radial start dist
  clipview_end = 0.2f;               // VR HMD fade+clipping radial end dist

  headlight_mode = RT_HEADLIGHT_OFF; // VR HMD headlight disabled by default

  shadows_enabled = RT_SHADOWS_OFF;  // disable shadows by default
  aa_samples = 0;                    // no AA samples by default

  ao_samples = 0;                    // no AO samples by default
  ao_direct = 0.3f;                  // AO direct contribution is 30%
  ao_ambient = 0.7f;                 // AO ambient contribution is 70%
  ao_maxdist = RT_DEFAULT_MAX;       // default is no max occlusion distance

  fog_mode = RT_FOG_NONE;            // fog/cueing disabled by default
  fog_start = 0.0f;
  fog_end = 10.0f;
  fog_density = 0.32f;

  regen_optix_pipeline=1;
  regen_optix_sbt=1;
  regen_optix_lights=1;

  verbose = RT_VERB_MIN;   // quiet console except perf/debugging cases
  check_verbose_env();     // see if the user has overridden verbose flag

  create_context();
  destroy_scene();         // zero object counters, prepare for rendering

  PROFILE_POP_RANGE();
}


// destructor...
TachyonOptiX::~TachyonOptiX(void) {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::~TachyonOptiX()", 0);

  if (context_created)
    destroy_context();

#if 0
  // XXX this is only for use with memory debugging tools!
  cudaDeviceReset();
#endif

  wkf_timer_destroy(rt_timer);

  PROFILE_POP_RANGE();
}


// Global OptiX logging callback
static void TachyonOptixLogCallback(unsigned int level,
                                    const char* tag,
                                    const char* message,
                                    void* cbdata) {
  if (cbdata != NULL) {
    TachyonOptiX *tcy = (TachyonOptiX *) cbdata;
    tcy->log_callback(level, tag, message);
  }
}


void TachyonOptiX::log_callback(unsigned int level, 
                                const char *tag, const char *msg) {
  // Log callback levels:
  //  1: fatal non-recoverable error, context needs to be destroyed
  //  2: recoverable error, invalid call params, etc.
  //  3: warning hints about slow perf, etc.
  //  4: print status or progress messages
  if ((verbose == RT_VERB_DEBUG) || (level < 4))
    printf("TachyonOptiX) [%s]: %s\n", tag, msg);
}


// check environment for verbose timing/debugging output flags
static TachyonOptiX::Verbosity get_verbose_flag(int inform=0) {
  TachyonOptiX::Verbosity myverbosity = TachyonOptiX::RT_VERB_MIN;
  char *verbstr = getenv("TACHYONOPTIXVERBOSE");
  if (verbstr != NULL) {
//    printf("TachyonOptiX) verbosity config request: '%s'\n", verbstr);
    if (!strcasecmp(verbstr, "MIN")) {
      myverbosity = TachyonOptiX::RT_VERB_MIN;
      if (inform)
        printf("TachyonOptiX) verbose setting: minimum\n");
    } else if (!strcasecmp(verbstr, "TIMING")) {
      myverbosity = TachyonOptiX::RT_VERB_TIMING;
      if (inform)
        printf("TachyonOptiX) verbose setting: timing data\n");
    } else if (!strcasecmp(verbstr, "DEBUG")) {
      myverbosity = TachyonOptiX::RT_VERB_DEBUG;
      if (inform)
        printf("TachyonOptiX) verbose setting: full debugging data\n");
    }
  }
  return myverbosity; 
}


int TachyonOptiX::device_list(int **devlist, char ***devnames) {
  TachyonOptiX::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("TachyonOptiX::device_list()\n");

  int devcount = 0;
  cudaGetDeviceCount(&devcount);
  return devcount;
}


int TachyonOptiX::device_count(void) {
  TachyonOptiX::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("TachyonOptiX::device_count()\n");
  
  return device_list(NULL, NULL);
}


unsigned int TachyonOptiX::optix_version(void) {
  TachyonOptiX::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("TachyonOptiX::optix_version()\n");

  /// The OptiX version.
  /// - major =  OPTIX_VERSION/10000
  /// - minor = (OPTIX_VERSION%10000)/100
  /// - micro =  OPTIX_VERSION%100

  unsigned int version=OPTIX_VERSION;

  return version;
}


void TachyonOptiX::check_verbose_env() {
  verbose = get_verbose_flag(1);
}


void TachyonOptiX::create_context() {
  DBG();
  time_ctx_create = 0;
  if (context_created)
    return;

  double starttime = wkf_timer_timenow(rt_timer);

  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) creating context...\n");

//  cudaFree(0); // initialize CUDA

  // initialize CUDA for this thread if not already
  cudaSetDevice(1);
 
  lasterr = optixInit();
  cudaStreamCreate(&stream);
  
  OptixDeviceContextOptions options = {};
  optixDeviceContextCreate(cudactx, &options, &ctx);

  lasterr = optixDeviceContextSetLogCallback(ctx,
                                             TachyonOptixLogCallback,
                                             this,
                                             4); // enable all levels 
  
  if (lasterr == OPTIX_SUCCESS)
    read_ptx_src(shaderpath, &rt_ptx_code_string);

  if (lasterr == OPTIX_SUCCESS)
    context_create_module();

  double time_ptxsrc = wkf_timer_timenow(rt_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("TachyonOptiX) load PTX shader src %.1f secs\n", time_ptxsrc - starttime);
  }

  if (lasterr == OPTIX_SUCCESS)
    context_create_pipeline();

  launchParamsBuffer.set_size(sizeof(rtLaunch));

  double time_pipeline = wkf_timer_timenow(rt_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("TachyonOptiX) create RT pipeline %.1f secs\n", time_pipeline - time_ptxsrc);
  }

  time_ctx_create = wkf_timer_timenow(rt_timer) - starttime;

  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) context creation time: %.2f\n", time_ctx_create);
  }

  context_created = 1;
}


int TachyonOptiX::read_ptx_src(const char *ptxfilename, char **ptxstring) {
  DBG();
  FILE *ptxfp = fopen(ptxfilename, "r");
  if (ptxfp == NULL) {
    return 0;
  } 

  // find size and load RT PTX source
  fseek(ptxfp, 0, SEEK_END);
  long ptxsize = ftell(ptxfp);
  fseek(ptxfp, 0, SEEK_SET);
  *ptxstring = (char *) calloc(1, ptxsize + 1);
  if (fread(*ptxstring, ptxsize, 1, ptxfp) != 1) {
    return 0;
  }
  
  return 1; 
}


void TachyonOptiX::context_create_exception_pgms() {
  DBG();
  exceptionPGs.resize(1);

  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgDesc.raygen.module            = module;           

  pgDesc.raygen.entryFunctionName="__exception__all";

  char log[2048];
  size_t sizeof_log = sizeof(log);
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts,
                                    log, &sizeof_log, &exceptionPGs[0]);

  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX exception construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_exception_pgms() {
  DBG();
  for (auto pg : exceptionPGs)
    optixProgramGroupDestroy(pg);
  exceptionPGs.clear();
}


void TachyonOptiX::context_create_raygen_pgms() {
  DBG();
  raygenPGs.resize(1);
      
  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module            = module;           

  // Assign the raygen program according to the active camera
  const char *raygenfctn=nullptr;
  switch (camera_type) {
    case RT_PERSPECTIVE:
      if (cam_dof_enabled)
        raygenfctn = "__raygen__camera_perspective_dof";
      else
        raygenfctn = "__raygen__camera_perspective";
      break;

    case RT_ORTHOGRAPHIC:
      if (cam_dof_enabled)
        raygenfctn = "__raygen__camera_orthographic_dof";
      else
        raygenfctn = "__raygen__camera_orthographic";
      break;

    case RT_CUBEMAP:
      if (cam_dof_enabled)
        raygenfctn = "__raygen__camera_cubemap_dof";
      else
        raygenfctn = "__raygen__camera_cubemap";
      break;

    case RT_DOME_MASTER:
      if (cam_dof_enabled)
        raygenfctn = "__raygen__camera_dome_master_dof";
      else
        raygenfctn = "__raygen__camera_dome_master";
      break;

    case RT_EQUIRECTANGULAR:
      if (cam_dof_enabled)
        raygenfctn = "__raygen__camera_equirectangular_dof";
      else
        raygenfctn = "__raygen__camera_equirectangular";
      break;

    case RT_OCULUS_RIFT:
      if (cam_dof_enabled)
        raygenfctn = "__raygen__camera_oculus_rift_dof";
      else
        raygenfctn = "__raygen__camera_oculus_rift";
      break;
  }
  pgDesc.raygen.entryFunctionName=raygenfctn;
  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) raygen: '%s'\n", raygenfctn);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts,
                                    log, &sizeof_log, &raygenPGs[0]);

  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX raygen construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_raygen_pgms() {
  DBG();
  for (auto pg : raygenPGs)
    optixProgramGroupDestroy(pg);
  raygenPGs.clear();
}


void TachyonOptiX::context_create_miss_pgms() {
  DBG();
  missPGs.resize(RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);
      
  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module              = module;           

  //
  // radiance rays
  //
 
  // Assign the miss program according to the active background mode 
  const char *missfctn=nullptr;
  switch (scene_background_mode) {
    case RT_BACKGROUND_TEXTURE_SKY_SPHERE:
      missfctn = "__miss__radiance_gradient_bg_sky_sphere";
      break;

    case RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE:
      missfctn = "__miss__radiance_gradient_bg_sky_plane";
      break;

    case RT_BACKGROUND_TEXTURE_SOLID:
    default:
      missfctn = "__miss__radiance_solid_bg";
      break;
  }
  pgDesc.miss.entryFunctionName=missfctn;
  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) miss: '%s'\n", missfctn);

  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log, 
                                    &missPGs[RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX miss radiance construction log:\n %s\n", log);
  }

  // shadow rays
  pgDesc.miss.entryFunctionName   = "__miss__shadow_nop";
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log, 
                                    &missPGs[RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX miss shadow construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_miss_pgms() {
  DBG();
  for (auto pg : missPGs)
    optixProgramGroupDestroy(pg);
  missPGs.clear();
}


void TachyonOptiX::context_create_hitgroup_pgms() {
  DBG();
  trimeshPGs.resize(RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof( log );
      
  OptixProgramGroupOptions pgOpts     = {};
  OptixProgramGroupDesc pgDesc        = {};
  pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  auto &hg = pgDesc.hitgroup;

  hg.moduleCH            = module;           
  hg.moduleAH            = module;           

  // radiance rays
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log, 
                                    &trimeshPGs[RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX hitgroup radiance construction log:\n %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  // XXX if we ever want to do two-pass shadows, we might use ray masks
  // and intersect opaque geometry first and do transmissive geom last
  //   hg.entryFunctionNameAH = "__anyhit__shadow_opaque";

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log, 
                                    &trimeshPGs[RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX hitgroup shadow construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_hitgroup_pgms() {
  DBG();
  for (auto pg : trimeshPGs)
    optixProgramGroupDestroy(pg);
  trimeshPGs.clear();
}


void TachyonOptiX::context_create_intersection_pgms() {
  DBG();
  custprimPGs.resize(RT_PRIM_COUNT * RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  auto &hg = pgDesc.hitgroup;
  hg.moduleIS = module;
  hg.moduleCH = module;
  hg.moduleAH = module;

  //
  // Cones 
  //
  const int conePG = RT_PRIM_CONE * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__cone_array_color";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[conePG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere radiance intersection construction log:\n %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__cone_array_color";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[conePG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere shadow intersection construction log:\n %s\n", log);
  }


  //
  // Cylinders
  //
  const int cylPG = RT_PRIM_CYLINDER * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__cylinder_array_color";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[cylPG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere radiance intersection construction log:\n %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__cylinder_array_color";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[cylPG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere shadow intersection construction log:\n %s\n", log);
  }


  //
  // Rings
  // 
  const int ringPG = RT_PRIM_RING * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__ring_array";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[ringPG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere radiance intersection construction log:\n %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__ring_array";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[ringPG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere shadow intersection construction log:\n %s\n", log);
  }


  //
  // Spheres
  //
  const int spherePG = RT_PRIM_SPHERE * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__sphere_array";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[spherePG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere radiance intersection construction log:\n %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__sphere_array";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts, log, &sizeof_log,
                                    &custprimPGs[spherePG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX sphere shadow intersection construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_intersection_pgms() {
  DBG();
  for (auto pg : custprimPGs)
    optixProgramGroupDestroy(pg);
  custprimPGs.clear();
}


void TachyonOptiX::context_create_module() {
  DBG();

  OptixModuleCompileOptions moduleCompOpts = {};
  moduleCompOpts.maxRegisterCount    = 50;
  moduleCompOpts.optLevel            = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
 
  // NOTE: lineinfo is required for profiling tools like nsight compute.
  // OptiX RT PTX must also be compiled using the "--generate-line-info" flag.
  moduleCompOpts.debugLevel          = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

  pipeCompOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipeCompOpts.usesMotionBlur        = false;
  pipeCompOpts.numPayloadValues      = 3;
  pipeCompOpts.numAttributeValues    = 2;

  // XXX enable exceptions full-time during development/testing
  if (1 || (getenv("TACHYONOPTIXDEBUG") != NULL)) {
    pipeCompOpts.exceptionFlags        = OPTIX_EXCEPTION_FLAG_DEBUG |
                                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                         OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  } else {
    pipeCompOpts.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
  }
  pipeCompOpts.pipelineLaunchParamsVariableName = "rtLaunch";

  char log[2048];
  size_t sizeof_log = sizeof(log);
  lasterr = optixModuleCreateFromPTX(ctx, &moduleCompOpts, &pipeCompOpts,
                                     rt_ptx_code_string,
                                     strlen(rt_ptx_code_string),
                                     log, &sizeof_log, &module);

  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX module construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_module() {
  DBG();
  optixModuleDestroy(module);
}


void TachyonOptiX::context_create_pipeline() {
  DBG();

  if (lasterr == OPTIX_SUCCESS)
    context_create_exception_pgms();

  if (lasterr == OPTIX_SUCCESS)
    context_create_raygen_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_miss_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_hitgroup_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_intersection_pgms();

  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : exceptionPGs)
    programGroups.push_back(pg);
  for (auto pg : raygenPGs)
    programGroups.push_back(pg);
  for (auto pg : missPGs)
    programGroups.push_back(pg);
  for (auto pg : trimeshPGs)
    programGroups.push_back(pg);
  for (auto pg : custprimPGs)
    programGroups.push_back(pg);

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) creating complete pipeline...\n");
  }
  
  char log[2048];
  size_t sizeof_log = sizeof(log);
  OptixPipelineLinkOptions pipeLinkOpts = {};
  pipeLinkOpts.maxTraceDepth            = 21; // OptiX recursion limit is 31
  lasterr = optixPipelineCreate(ctx, &pipeCompOpts, &pipeLinkOpts,
                                programGroups.data(), (int)programGroups.size(),
                                log, &sizeof_log, &pipe);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("OptiX pipeline construction log:\n %s\n", log);
  }

  // max allowed stack sz appears to be 64kB per category
  optixPipelineSetStackSize(pipe, 
                            8*1024, ///< direct IS/AH callable stack sz
                            8*1024, ///< direct RG/MS/CH callable stack sz
                            8*1024, ///< continuation stack sz
                            1);     ///< max traversal depth

  regen_optix_pipeline=0;
  regen_optix_sbt=1;
}


void TachyonOptiX::context_destroy_pipeline() {
  DBG();
  cudaDeviceSynchronize();
  context_destroy_SBT();

  if (pipe != nullptr) {
    if (verbose == RT_VERB_DEBUG)
      printf("TachyonOptiX) destroying existing pipeline...\n");

    optixPipelineDestroy(pipe);
    pipe=nullptr;
  }

  context_destroy_raygen_pgms();
  context_destroy_miss_pgms();
  context_destroy_hitgroup_pgms();
  context_destroy_intersection_pgms();
  context_destroy_exception_pgms();

  regen_optix_pipeline=1;
  regen_optix_sbt=1;
}


void TachyonOptiX::context_create_SBT() {
  DBG();

  memset((void *) &sbt, 0, sizeof(sbt));

  // build exception records
  std::vector<ExceptionRecord> exceptionRecords;
  for (int i=0; i<exceptionPGs.size(); i++) {
    ExceptionRecord rec = {};
    optixSbtRecordPackHeader(exceptionPGs[i], &rec);
    rec.data = nullptr;
    exceptionRecords.push_back(rec);
  }
  exceptionRecordsBuffer.resize_upload(exceptionRecords);
  sbt.exceptionRecord = exceptionRecordsBuffer.cu_dptr();

  // build raygen records
  std::vector<RaygenRecord> raygenRecords;
  for (int i=0; i<raygenPGs.size(); i++) {
    RaygenRecord rec = {};
    optixSbtRecordPackHeader(raygenPGs[i], &rec);
    rec.data = nullptr;
    raygenRecords.push_back(rec);
  }
  raygenRecordsBuffer.resize_upload(raygenRecords);
  sbt.raygenRecord = raygenRecordsBuffer.cu_dptr();

  // build miss records
  std::vector<MissRecord> missRecords;
  for (int i=0; i<missPGs.size(); i++) {
    MissRecord rec = {};
    optixSbtRecordPackHeader(missPGs[i], &rec);
    rec.data = nullptr;
    missRecords.push_back(rec);
  }
  missRecordsBuffer.resize_upload(missRecords);
  sbt.missRecordBase          = missRecordsBuffer.cu_dptr();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount         = (int) missRecords.size();

  // build hitgroup records
  //   Note: SBT must not contain any NULLs, stubs must exist at least
  std::vector<HGRecordGroup> HGRecGroups;

  // Add cone arrays to SBT
  const int conePG = RT_PRIM_CONE * RT_RAY_TYPE_COUNT;
  int numCones = (int) conearrays.size();
  for (int objID=0; objID<numCones; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.cone.base     = (float3 *) coneBaseBuffers[objID].cu_dptr();
    p.cone.apex     = (float3 *) coneApexBuffers[objID].cu_dptr();
    p.cone.baserad  = (float *) coneBaseRadBuffers[objID].cu_dptr();
    p.cone.apexrad  = (float *) coneApexRadBuffers[objID].cu_dptr();

    auto &d = rec.radiance.data;
    d.prim_color = (float3 *) conePrimColorBuffers[objID].cu_dptr();
    d.uniform_color = conearrays[objID].uniform_color;
    d.materialindex = conearrays[objID].materialindex;

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[conePG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[conePG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }

  // Add cylinder arrays to SBT
  const int cylPG = RT_PRIM_CYLINDER * RT_RAY_TYPE_COUNT;
  int numCyls = (int) cyarrays.size();
  for (int objID=0; objID<numCyls; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.cyl.start  = (float3 *) cyStartBuffers[objID].cu_dptr();
    p.cyl.end    = (float3 *) cyEndBuffers[objID].cu_dptr();
    p.cyl.radius = (float *) cyRadiusBuffers[objID].cu_dptr();

    auto &d = rec.radiance.data;
    d.prim_color = (float3 *) cyPrimColorBuffers[objID].cu_dptr();
    d.uniform_color = cyarrays[objID].uniform_color;
    d.materialindex = cyarrays[objID].materialindex;

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[cylPG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[cylPG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }

  // Add ring arrays to SBT
  const int ringPG = RT_PRIM_RING * RT_RAY_TYPE_COUNT;
  int numRings = (int) riarrays.size();
  for (int objID=0; objID<numRings; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.ring.center = (float3 *) riCenterBuffers[objID].cu_dptr();
    p.ring.norm   = (float3 *) riNormalBuffers[objID].cu_dptr();
    p.ring.inrad  = (float *) riInRadiusBuffers[objID].cu_dptr();
    p.ring.outrad = (float *) riOutRadiusBuffers[objID].cu_dptr();

    auto &d = rec.radiance.data;
    d.prim_color = (float3 *) riPrimColorBuffers[objID].cu_dptr();
    d.uniform_color = riarrays[objID].uniform_color;
    d.materialindex = riarrays[objID].materialindex;

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[ringPG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[ringPG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }

  // Add sphere arrays to SBT
  const int spherePG = RT_PRIM_SPHERE * RT_RAY_TYPE_COUNT;
  int numSpheres = (int) sparrays.size();
  for (int objID=0; objID<numSpheres; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.sphere.center = (float3 *) spCenterBuffers[objID].cu_dptr();
    p.sphere.radius = (float *) spRadiusBuffers[objID].cu_dptr();

    auto &d = rec.radiance.data;
    d.prim_color = (float3 *) spPrimColorBuffers[objID].cu_dptr();
    d.uniform_color = sparrays[objID].uniform_color;
    d.materialindex = sparrays[objID].materialindex;

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[spherePG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[spherePG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


  // Add triangle meshes to SBT
  int numTrimeshes = (int) meshes.size();
  for (int objID=0; objID<numTrimeshes; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &t = rec.radiance.data.trimesh;
    t.vertex = (float3 *) triMeshVertBuffers[objID].cu_dptr();
    t.index  = (int3 *) triMeshIdxBuffers[objID].cu_dptr();
    t.normals = (float3 *) triMeshVertNormalBuffers[objID].cu_dptr();
    t.packednormals = (uint4 *) triMeshVertPackedNormalBuffers[objID].cu_dptr();
    t.vertexcolors3f = (float3 *) triMeshVertColor3fBuffers[objID].cu_dptr();
    t.vertexcolors4u = (uchar4 *) triMeshVertColor4uBuffers[objID].cu_dptr();

    auto &d = rec.radiance.data;
    d.prim_color = (float3 *) triMeshPrimColorBuffers[objID].cu_dptr();
    d.uniform_color = meshes[objID].uniform_color;
    d.materialindex = meshes[objID].materialindex;

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(trimeshPGs[RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(trimeshPGs[RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


#if 0
  hitgroupRecordsBuffer.resize_upload(HGRecGroups);

  sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.cu_dptr();
  sbt.hitgroupRecordStrideInBytes = sizeof(HGRecord);

  // Each HGRecordGroup contains RT_RAY_TYPE_COUNT HGRecords, so we multiply
  // the vector size by RT_RAY_TYPE_COUNT to get the total HG record count
  sbt.hitgroupRecordCount         = (int) HGRecGroups.size()*RT_RAY_TYPE_COUNT;
#else
  // upload and set the final SBT hitgroup array
  int hgrgsz = HGRecGroups.size();
  if (hgrgsz > 0) {
    // temporarily append the contents of HGRecGroups to hitgroupRecordGroups
    // so they are also included in the SBT
    auto &h = hitgroupRecordGroups;
    int hgsz = h.size();

    // pre-grow hitgroupRecordGroups to final size prior to append loop...
    if (h.capacity() < (hgsz+hgrgsz))
      h.reserve(hgsz+hgrgsz);

    // append HGRecGroups and upload the final HG record list to the GPU
    for (auto r: HGRecGroups) {
      h.push_back(r);
    }
    hitgroupRecordsBuffer.resize_upload(h);

    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.cu_dptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(HGRecord);

    // Each HGRecordGroup contains RT_RAY_TYPE_COUNT HGRecords, so we multiply
    // the vector size by RT_RAY_TYPE_COUNT to get the total HG record count
    sbt.hitgroupRecordCount = (int) hitgroupRecordGroups.size()*RT_RAY_TYPE_COUNT;

    // delete temporarily appended HGRecGroups records 
    h.erase(h.begin()+hgsz, h.end());
  } else {
    hitgroupRecordsBuffer.resize_upload(hitgroupRecordGroups);

    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.cu_dptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(HGRecord);

    // Each HGRecordGroup contains RT_RAY_TYPE_COUNT HGRecords, so we multiply
    // the vector size by RT_RAY_TYPE_COUNT to get the total HG record count
    sbt.hitgroupRecordCount = (int) hitgroupRecordGroups.size()*RT_RAY_TYPE_COUNT;
  } 
#endif

  regen_optix_sbt=0;
}


void TachyonOptiX::context_destroy_SBT() {
  DBG();

  exceptionRecordsBuffer.free();
  raygenRecordsBuffer.free();
  missRecordsBuffer.free();
  hitgroupRecordsBuffer.free(); 

  memset((void *) &sbt, 0, sizeof(sbt));
  regen_optix_sbt=1;
}


OptixTraversableHandle TachyonOptiX::build_trimeshes_GAS() {
  DBG();

  const int arrayCount = meshes.size();

  // RTX triangle inputs, preset vector sizes
  // AS build will consume device pointers, so when these
  // are freed, we should be destroying the associated AS
  triMeshVertBuffers.resize(arrayCount);
  triMeshIdxBuffers.resize(arrayCount);
  triMeshVertNormalBuffers.resize(arrayCount);
  triMeshVertPackedNormalBuffers.resize(arrayCount);
  triMeshVertColor3fBuffers.resize(arrayCount);
  triMeshVertColor4uBuffers.resize(arrayCount);
  triMeshPrimColorBuffers.resize(arrayCount);

  std::vector<OptixBuildInput> asTriInp(arrayCount);
  std::vector<CUdeviceptr> d_vertices(arrayCount);
  std::vector<uint32_t> asTriInpFlags(arrayCount);

  // loop over geom buffers and incorp into AS build...
  //   Uploads each mesh to the GPU before building AS,
  //   stores resulting device pointers in lists, and 
  //   prepares OptixBuildInput records containing the
  //   resulting device pointers, primitive counts, and flags.
  for (int i=0; i<arrayCount; i++) {
    TriangleMesh &model = meshes[i];
    triMeshVertBuffers[i].resize_upload(model.vertex);
    triMeshIdxBuffers[i].resize_upload(model.index);

    // optional buffers 
    triMeshVertNormalBuffers[i].free();
    triMeshVertPackedNormalBuffers[i].free();
    triMeshVertColor3fBuffers[i].free();
    triMeshVertColor4uBuffers[i].free();
    triMeshPrimColorBuffers[i].free();

    asTriInp[i] = {};
    asTriInp[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    d_vertices[i] = triMeshVertBuffers[i].cu_dptr(); // host array of dev ptrs...

    // device triangle mesh buffers
    auto &triArray = asTriInp[i].triangleArray;
   
    // device trimesh vertex buffer 
    triArray.vertexBuffers               = &d_vertices[i];
    triArray.numVertices                 = (int)model.vertex.size();
    triArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triArray.vertexStrideInBytes         = sizeof(float3);
    
    // device trimesh index buffer 
    triArray.indexBuffer                 = triMeshIdxBuffers[i].cu_dptr();
    triArray.numIndexTriplets            = (int)model.index.size();
    triArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triArray.indexStrideInBytes          = sizeof(int3);
    triArray.preTransform                = 0; // no xform matrix
   
    // Ensure that anyhit is called only once for transparency handling 
    asTriInpFlags[i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    triArray.flags                       = &asTriInpFlags[i];
 
    triArray.numSbtRecords               = 1;
    triArray.sbtIndexOffsetBuffer        = 0; 
    triArray.sbtIndexOffsetSizeInBytes   = 0; 
    triArray.sbtIndexOffsetStrideInBytes = 0; 
    triArray.primitiveIndexOffset        = 0;
  }
    
  // BLAS setup
  OptixAccelBuildOptions asOpts          = {};
  asOpts.motionOptions.numKeys           = 1;
  asOpts.buildFlags                      = OPTIX_BUILD_FLAG_NONE | 
                                           OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  asOpts.operation                       = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufSizes = {};
  optixAccelComputeMemoryUsage(ctx, &asOpts, asTriInp.data(),
                               arrayCount, &blasBufSizes);

  // prepare compaction
  CUMemBuf compactedSizeBuffer;
  compactedSizeBuffer.set_size(sizeof(uint64_t));
  OptixAccelEmitDesc emitDesc = {};
  emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.cu_dptr();
    
  // execute build (main stage)
  CUMemBuf tempBuffer, outputBuffer;
  tempBuffer.set_size(blasBufSizes.tempSizeInBytes);
  outputBuffer.set_size(blasBufSizes.outputSizeInBytes);
  OptixTraversableHandle asHandle { 0 };
  optixAccelBuild(ctx, /* stream */0, &asOpts,
                  asTriInp.data(), arrayCount,
                  tempBuffer.cu_dptr(), tempBuffer.get_size(),
                  outputBuffer.cu_dptr(), outputBuffer.get_size(),
                  &asHandle, &emitDesc, 1);

  cudaDeviceSynchronize(); // XXX device-wide sync
    
  // perform compaction
  uint64_t compactedSize = 0;
  compactedSizeBuffer.download(&compactedSize, 1);
  
  trimeshesGASBuffer.set_size(compactedSize);
  optixAccelCompact(ctx, /*stream:*/0, asHandle,
                    trimeshesGASBuffer.cu_dptr(), 
                    trimeshesGASBuffer.get_size(), &asHandle);

  cudaDeviceSynchronize(); // XXX device-wide sync

  // at this point, the final compacted AS is stored in the final buffer 
  // and the returned asHandle--ephemeral data can be destroyed...
  outputBuffer.free();
  tempBuffer.free();
  compactedSizeBuffer.free();

  return asHandle;
}


void TachyonOptiX::free_trimeshes_GAS() {
  DBG();

  trimeshesGASBuffer.free();
}


OptixTraversableHandle TachyonOptiX::build_custprims_GAS() {
  DBG();

  // RTX custom primitive inputs
  // AS build will consume device pointers, so when these
  // are freed, we should be destroying the associated AS

  const int coneCount = conearrays.size();
  coneBaseBuffers.resize(coneCount);
  coneApexBuffers.resize(coneCount);
  coneBaseRadBuffers.resize(coneCount);
  coneApexRadBuffers.resize(coneCount);
  conePrimColorBuffers.resize(coneCount);
  coneAabbBuffers.resize(coneCount);

  const int cyCount = cyarrays.size();
  cyStartBuffers.resize(cyCount);
  cyEndBuffers.resize(cyCount);
  cyRadiusBuffers.resize(cyCount);
  cyPrimColorBuffers.resize(cyCount);
  cyAabbBuffers.resize(cyCount);

  const int riCount = riarrays.size();
  riCenterBuffers.resize(riCount);
  riNormalBuffers.resize(riCount);
  riInRadiusBuffers.resize(riCount);
  riOutRadiusBuffers.resize(riCount);
  riPrimColorBuffers.resize(riCount);
  riAabbBuffers.resize(riCount);

  const int spCount = sparrays.size();
  spCenterBuffers.resize(spCount);
  spRadiusBuffers.resize(spCount);
  spPrimColorBuffers.resize(spCount);
  spAabbBuffers.resize(spCount);

  const int arrayCount = coneCount + cyCount + riCount + spCount;

  std::vector<OptixBuildInput> asCustInp(arrayCount);
  std::vector<CUdeviceptr> d_aabb(arrayCount);
  std::vector<uint32_t> asCustInpFlags(arrayCount);

  std::vector<OptixAabb> hostAabb; // temp array for aabb generation

  // loop over geom buffers and incorp into AS build...
  //   Uploads each mesh to the GPU before building AS,
  //   stores resulting device pointers in lists, and 
  //   prepares OptixBuildInput records containing the
  //   resulting device pointers, primitive counts, and flags.
  int bufIdx = 0;

  // Cones...
  for (int i=0; i<coneCount; i++) {
    ConeArray &model = conearrays[i];
    coneBaseBuffers[i].resize_upload(model.base);
    coneApexBuffers[i].resize_upload(model.apex);
    coneBaseRadBuffers[i].resize_upload(model.baserad);
    coneApexRadBuffers[i].resize_upload(model.apexrad);
//    conePrimColorBuffers[i].resize();

    // XXX AABB calcs should be done in CUDA on the GPU...
    hostAabb.resize(model.base.size());
    for (int j=0; j<model.base.size(); j++) {
      auto &base = model.base[j];
      auto &apex = model.apex[j];
      float baserad = model.baserad[j];
      float apexrad = model.apexrad[j];

      hostAabb[j].minX = fminf(base.x - baserad, apex.x - apexrad);
      hostAabb[j].minY = fminf(base.y - baserad, apex.y - apexrad);
      hostAabb[j].minZ = fminf(base.z - baserad, apex.z - apexrad);
      hostAabb[j].maxX = fmaxf(base.x + baserad, apex.x + apexrad);
      hostAabb[j].maxY = fmaxf(base.y + baserad, apex.y + apexrad);
      hostAabb[j].maxZ = fmaxf(base.z + baserad, apex.z + apexrad);
    } 
    coneAabbBuffers[i].resize_upload(hostAabb);

    asCustInp[bufIdx + i] = {};
    asCustInp[bufIdx + i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    // host array of device ptrs...
    d_aabb[bufIdx + i] = coneAabbBuffers[i].cu_dptr();
   
    // device custom primitive buffers
    auto &primArray = asCustInp[bufIdx + i].customPrimitiveArray;

    primArray.aabbBuffers                 = &d_aabb[bufIdx + i];
    primArray.numPrimitives               = (int) model.base.size();
    primArray.strideInBytes               = sizeof(HGRecord);

    // Ensure that anyhit is called only once for transparency handling
    asCustInpFlags[bufIdx + i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    primArray.flags                       = &asCustInpFlags[bufIdx + i];

    primArray.numSbtRecords               = 1;
    primArray.sbtIndexOffsetBuffer        = 0;   // No per-primitive record
    primArray.sbtIndexOffsetSizeInBytes   = 0;
    primArray.sbtIndexOffsetStrideInBytes = 0;
    primArray.primitiveIndexOffset        = 0;
  }
  bufIdx += coneCount;

  // Cylinders...
  for (int i=0; i<cyCount; i++) {
    CylinderArray &model = cyarrays[i];
    cyStartBuffers[i].resize_upload(model.start);
    cyEndBuffers[i].resize_upload(model.end);
    cyRadiusBuffers[i].resize_upload(model.radius);
//    cyPrimColorBuffers[i].resize();

    // XXX AABB calcs should be done in CUDA on the GPU...
    hostAabb.resize(model.start.size());
    for (int j=0; j<model.start.size(); j++) {
      auto &base = model.start[j];
      auto &apex = model.end[j];
      float rad = model.radius[j];

      hostAabb[j].minX = fminf(base.x - rad, apex.x - rad);
      hostAabb[j].minY = fminf(base.y - rad, apex.y - rad);
      hostAabb[j].minZ = fminf(base.z - rad, apex.z - rad);
      hostAabb[j].maxX = fmaxf(base.x + rad, apex.x + rad);
      hostAabb[j].maxY = fmaxf(base.y + rad, apex.y + rad);
      hostAabb[j].maxZ = fmaxf(base.z + rad, apex.z + rad);
    } 
    cyAabbBuffers[i].resize_upload(hostAabb);

    asCustInp[bufIdx + i] = {};
    asCustInp[bufIdx + i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    // host array of device ptrs...
    d_aabb[bufIdx + i] = cyAabbBuffers[i].cu_dptr();
   
    // device custom primitive buffers
    auto &primArray = asCustInp[bufIdx + i].customPrimitiveArray;

    primArray.aabbBuffers                 = &d_aabb[bufIdx + i];
    primArray.numPrimitives               = (int) model.start.size();
    primArray.strideInBytes               = sizeof(HGRecord);

    // Ensure that anyhit is called only once for transparency handling
    asCustInpFlags[bufIdx + i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    primArray.flags                       = &asCustInpFlags[bufIdx + i];

    primArray.numSbtRecords               = 1;
    primArray.sbtIndexOffsetBuffer        = 0;   // No per-primitive record
    primArray.sbtIndexOffsetSizeInBytes   = 0;
    primArray.sbtIndexOffsetStrideInBytes = 0;
    primArray.primitiveIndexOffset        = 0;
  }
  bufIdx += cyCount;

  // Rings...
  for (int i=0; i<riCount; i++) {
    RingArray &model = riarrays[i];
    riCenterBuffers[i].resize_upload(model.center);
    riNormalBuffers[i].resize_upload(model.normal);
    riInRadiusBuffers[i].resize_upload(model.inrad);
    riOutRadiusBuffers[i].resize_upload(model.outrad);
//    riPrimColorBuffers[i].resize();

    // XXX AABB calcs should be done in CUDA on the GPU...
    hostAabb.resize(model.center.size());
    for (int j=0; j<model.center.size(); j++) {
      float rad = model.outrad[j];
      hostAabb[j].minX = model.center[j].x - rad;
      hostAabb[j].minY = model.center[j].y - rad;
      hostAabb[j].minZ = model.center[j].z - rad;
      hostAabb[j].maxX = model.center[j].x + rad;
      hostAabb[j].maxY = model.center[j].y + rad;
      hostAabb[j].maxZ = model.center[j].z + rad;
    } 
    riAabbBuffers[i].resize_upload(hostAabb);

    asCustInp[bufIdx + i] = {};
    asCustInp[bufIdx + i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    // host array of device ptrs...
    d_aabb[bufIdx + i] = riAabbBuffers[i].cu_dptr();
   
    // device custom primitive buffers
    auto &primArray = asCustInp[bufIdx + i].customPrimitiveArray;

    primArray.aabbBuffers                 = &d_aabb[bufIdx + i];
    primArray.numPrimitives               = (int) model.center.size();
    primArray.strideInBytes               = sizeof(HGRecord);

    // Ensure that anyhit is called only once for transparency handling
    asCustInpFlags[bufIdx + i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    primArray.flags                       = &asCustInpFlags[bufIdx + i];

    primArray.numSbtRecords               = 1;
    primArray.sbtIndexOffsetBuffer        = 0;   // No per-primitive record
    primArray.sbtIndexOffsetSizeInBytes   = 0;
    primArray.sbtIndexOffsetStrideInBytes = 0;
    primArray.primitiveIndexOffset        = 0;
  }
  bufIdx += riCount;

  // Spheres...
  for (int i=0; i<spCount; i++) {
    SphereArray &model = sparrays[i];
    spCenterBuffers[i].resize_upload(model.center);
    spRadiusBuffers[i].resize_upload(model.radius);
//    spPrimColorBuffers[i].resize();

    // XXX AABB calcs should be done in CUDA on the GPU...
    hostAabb.resize(model.radius.size());
    for (int j=0; j<model.radius.size(); j++) {
      float rad = model.radius[j];
      hostAabb[j].minX = model.center[j].x - rad;
      hostAabb[j].minY = model.center[j].y - rad;
      hostAabb[j].minZ = model.center[j].z - rad;
      hostAabb[j].maxX = model.center[j].x + rad;
      hostAabb[j].maxY = model.center[j].y + rad;
      hostAabb[j].maxZ = model.center[j].z + rad;
    } 
    spAabbBuffers[i].resize_upload(hostAabb);

    asCustInp[bufIdx + i] = {};
    asCustInp[bufIdx + i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    // host array of device ptrs...
    d_aabb[bufIdx + i] = spAabbBuffers[i].cu_dptr();
   
    // device custom primitive buffers
    auto &primArray = asCustInp[bufIdx + i].customPrimitiveArray;

    primArray.aabbBuffers                 = &d_aabb[bufIdx + i];
    primArray.numPrimitives               = (int) model.radius.size();
    primArray.strideInBytes               = sizeof(HGRecord);

    // Ensure that anyhit is called only once for transparency handling
    asCustInpFlags[bufIdx + i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    primArray.flags                       = &asCustInpFlags[bufIdx + i];

    primArray.numSbtRecords               = 1;
    primArray.sbtIndexOffsetBuffer        = 0;   // No per-primitive record
    primArray.sbtIndexOffsetSizeInBytes   = 0;
    primArray.sbtIndexOffsetStrideInBytes = 0;
    primArray.primitiveIndexOffset        = 0;
  }
  bufIdx += spCount;
    
  // BLAS setup
  OptixAccelBuildOptions asOpts           = {};
  asOpts.motionOptions.numKeys            = 1;
  asOpts.buildFlags                       = OPTIX_BUILD_FLAG_NONE | 
                                            OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  asOpts.operation                        = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufSizes = {};
  optixAccelComputeMemoryUsage(ctx, &asOpts, asCustInp.data(),
                               arrayCount, &blasBufSizes);

  // prepare compaction
  CUMemBuf compactedSizeBuffer;
  compactedSizeBuffer.set_size(sizeof(uint64_t));
  OptixAccelEmitDesc emitDesc = {};
  emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.cu_dptr();
    
  // execute build (main stage)
  CUMemBuf tempBuffer, outputBuffer;
  tempBuffer.set_size(blasBufSizes.tempSizeInBytes);
  outputBuffer.set_size(blasBufSizes.outputSizeInBytes);
  OptixTraversableHandle asHandle { 0 };
  optixAccelBuild(ctx, /* stream */0, &asOpts,
                  asCustInp.data(), arrayCount,
                  tempBuffer.cu_dptr(), tempBuffer.get_size(),
                  outputBuffer.cu_dptr(), outputBuffer.get_size(),
                  &asHandle, &emitDesc, 1);

  cudaDeviceSynchronize(); // XXX device-wide sync
    
  // perform compaction
  uint64_t compactedSize = 0;
  compactedSizeBuffer.download(&compactedSize, 1);
  
  custprimsGASBuffer.set_size(compactedSize);
  optixAccelCompact(ctx, /*stream:*/0, asHandle,
                    custprimsGASBuffer.cu_dptr(), 
                    custprimsGASBuffer.get_size(), &asHandle);

  cudaDeviceSynchronize(); // XXX device-wide sync

  // at this point, the final compacted AS is stored in the final Buffer 
  // and the returned asHandle--ephemeral data can be destroyed...
  outputBuffer.free();
  tempBuffer.free();
  compactedSizeBuffer.free();

  return asHandle;
}


void TachyonOptiX::free_custprims_GAS() {
  DBG();

  custprimsGASBuffer.free();
}


void TachyonOptiX::build_IAS() {
  DBG();

  OptixTraversableHandle trimeshesGAS = {};
  OptixTraversableHandle custprimsGAS = {};

  // (re)build GASes for each geometry class
  if (sparrays.size() > 0) {
    free_custprims_GAS();
    custprimsGAS = build_custprims_GAS();
  }
  rtLaunch.traversable = custprimsGAS;

  if (meshes.size() > 0) {
    free_trimeshes_GAS();
    trimeshesGAS = build_trimeshes_GAS();
  }
  rtLaunch.traversable = trimeshesGAS;

#if 1
  int sbtOffset = 0;
  std::vector<OptixInstance> instances;

  OptixInstance tmpInst = {};
  auto &i = tmpInst;
  float identity_xform3x4[12] = {
    1.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  1.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  1.0f,  0.0f
  };

  // populate instance 
  memcpy(i.transform, identity_xform3x4, sizeof(identity_xform3x4));
  i.instanceId = 0;
  i.sbtOffset = 0;
  i.visibilityMask = 0xFF;
  i.flags = OPTIX_INSTANCE_FLAG_NONE;

  if (custprimsGAS && sparrays.size() > 0) {
    i.traversableHandle = custprimsGAS;
    i.sbtOffset = sbtOffset;
    instances.push_back(i);

    sbtOffset += RT_RAY_TYPE_COUNT * (conearrays.size() + cyarrays.size() + riarrays.size() + sparrays.size());
  }

  if (trimeshesGAS && meshes.size() > 0) {
    i.traversableHandle = trimeshesGAS;
    i.sbtOffset = sbtOffset;
    instances.push_back(i);

    sbtOffset += RT_RAY_TYPE_COUNT * meshes.size();
  }

#if 0
  printf("custprimsGAS: %p\n", custprimsGAS);
  printf("trimeshesGAS: %p\n", trimeshesGAS);
  printf("i.traversable: %p\n", i.traversableHandle);
  printf("instance[0].traversable: %p\n", instances[0].traversableHandle);
#endif

  cudaDeviceSynchronize(); // XXX device-wide sync

  CUMemBuf devinstances;
  devinstances.resize_upload(instances);

  OptixBuildInput buildInput = {};
  buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  buildInput.instanceArray.instances    = devinstances.cu_dptr();
  buildInput.instanceArray.numInstances = (int) instances.size();

  OptixAccelBuildOptions asOpts         = {};
  asOpts.buildFlags                     = OPTIX_BUILD_FLAG_NONE;
  asOpts.operation                      = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes tlasBufSizes = {};
  optixAccelComputeMemoryUsage(ctx, &asOpts, &buildInput,
                               1 /* build inputs */, &tlasBufSizes);

  // execute build (main stage)
  CUMemBuf tempBuffer;
  tempBuffer.set_size(tlasBufSizes.tempSizeInBytes);
  IASBuffer.set_size(tlasBufSizes.outputSizeInBytes);
  OptixTraversableHandle asHandle { 0 };
  optixAccelBuild(ctx, /* stream */0, &asOpts,
                  &buildInput, 1,  // num build inputs
                  tempBuffer.cu_dptr(), tempBuffer.get_size(),
                  IASBuffer.cu_dptr(), IASBuffer.get_size(),
                  &asHandle, nullptr, 0);

  cudaDeviceSynchronize(); // XXX device-wide sync

  devinstances.free();
  tempBuffer.free();

  rtLaunch.traversable = asHandle;
#endif

}


void TachyonOptiX::free_IAS() {
  DBG();
  IASBuffer.free();
}


void TachyonOptiX::setup_context(int w, int h) {
  DBG();
  double starttime = wkf_timer_timenow(rt_timer);
  time_ctx_setup = 0;

  lasterr = OPTIX_SUCCESS; // clear any error state
  width = w;
  height = h;

  if (!context_created)
    return;

  check_verbose_env(); // update verbose flag if changed since last run

  // set default global ray tracing recursion depth and/or allow
  // runtime user override here.
  scene_max_depth = 10;
  scene_max_trans = scene_max_depth;

  // set default scene epsilon
  scene_epsilon = 5.e-5f * 50;

  // zero out the array of material usage counts for the scene
//  memset(material_special_counts, 0, sizeof(material_special_counts));

  time_ctx_setup = wkf_timer_timenow(rt_timer) - starttime;
}


void TachyonOptiX::destroy_context() {
  DBG();
  if (!context_created)
    return;

  destroy_scene();
  context_destroy_pipeline();
  context_destroy_module();
  framebuffer_destroy();

  // launch params buffer refers to materials/lights buffers 
  // so we destroy it first...
  launchParamsBuffer.free();
  materialsBuffer.free();
  directionalLightsBuffer.free(); 
  positionalLightsBuffer.free(); 
 
  optixDeviceContextDestroy(ctx);

  regen_optix_pipeline=1;
  regen_optix_sbt=1;
  regen_optix_lights=1;

#if 0
  if ((lasterr = optixContextDestroy(ctx)) != OPTIX_SUCCESS) {
    msgErr << "TachyonOptiX) An error occured while destroying the OptiX context" << sendmsg;
  }
#endif
}


void TachyonOptiX::add_material(int matindex,
                                float ambient, float diffuse, float specular,
                                float shininess, float reflectivity,
                                float opacity,
                                float outline, float outlinewidth,
                                int transmode) {
  DBG();

  int oldmatcount = materialcache.size();
  if (oldmatcount <= matindex) {
    rt_material m;

    // XXX do something noticable so we see that we got a bad entry...
    m.ambient = 0.5f;
    m.diffuse = 0.7f;
    m.specular = 0.0f;
    m.shininess = 10.0f;
    m.reflectivity = 0.0f;
    m.opacity = 1.0f;
    m.transmode = 0;

    materialcache.resize(matindex+1);
    materialvalid.resize(matindex+1);
    for (int i=oldmatcount; i<=matindex; i++) {
      materialcache[i]=m;
      materialvalid[i]=0;
    }
  }

  if (materialvalid[matindex]) {
    return;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("TachyonOptiX) Adding material[%d]\n", matindex);

    materialcache[matindex].ambient      = ambient;
    materialcache[matindex].diffuse      = diffuse;
    materialcache[matindex].specular     = specular;
    materialcache[matindex].shininess    = shininess;
    materialcache[matindex].reflectivity = reflectivity;
    materialcache[matindex].opacity      = opacity;
    materialcache[matindex].outline      = outline;
    materialcache[matindex].outlinewidth = outlinewidth;
    materialcache[matindex].transmode    = transmode;

    materialvalid[matindex]=1;
    regen_optix_materials=1; // force a fresh material table upload to the GPU
  }
}


void TachyonOptiX::init_materials() {
  if (verbose == RT_VERB_DEBUG) printf("TachyonOptiX) init_materials()\n");
}


void TachyonOptiX::add_directional_light(const float *dir, const float *color) {
  rt_directional_light l;
  l.dir = normalize(make_float3(dir[0], dir[1], dir[2]));
//  l.color = make_float3(color[0], color[1], color[2]);
  directional_lights.push_back(l);
  regen_optix_lights=1;
}


void TachyonOptiX::add_positional_light(const float *pos, const float *color) {
  rt_positional_light l;
  l.pos = make_float3(pos[0], pos[1], pos[2]);
//  l.color = make_float3(color[0], color[1], color[2]);
  positional_lights.push_back(l);
  regen_optix_lights=1;
}


void TachyonOptiX::destroy_scene() {
  DBG();
  double starttime = wkf_timer_timenow(rt_timer);
  time_ctx_destroy_scene = 0;

  // zero out all object counters
  cylinder_array_cnt = 0;
  cylinder_array_color_cnt = 0;
  ring_array_color_cnt = 0;
  sphere_array_cnt = 0;
  sphere_array_color_cnt = 0;
  tricolor_cnt = 0;
  trimesh_c4u_n3b_v3f_cnt = 0;
  trimesh_n3b_v3f_cnt = 0;
  trimesh_n3f_v3f_cnt = 0;
  trimesh_v3f_cnt = 0;

  if (!context_created)
    return;

  // XXX this renderer class isn't tracking scene state yet
  scene_created = 1;
  if (scene_created) {
    materialcache.clear();
    materialvalid.clear();

    clear_all_lights();

    triMeshVertBuffers.clear();
    triMeshIdxBuffers.clear();
    triMeshVertNormalBuffers.clear();
    triMeshVertPackedNormalBuffers.clear();
    triMeshVertColor3fBuffers.clear();
    triMeshVertColor4uBuffers.clear();
    triMeshPrimColorBuffers.clear();
    meshes.clear();

    spCenterBuffers.clear();
    spRadiusBuffers.clear();
    sparrays.clear();

    context_destroy_SBT();
    free_trimeshes_GAS();
  }

  double endtime = wkf_timer_timenow(rt_timer);
  time_ctx_destroy_scene = endtime - starttime;

  scene_created = 0; // scene has been destroyed
}


void TachyonOptiX::set_camera_lookat(const float *at, const float *upV) {
  // force position update to be committed to the rtLaunch struct too...
  rtLaunch.cam.pos = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
  float3 lookat = make_float3(at[0], at[1], at[2]);
  float3 V = make_float3(upV[0], upV[1], upV[2]);
  rtLaunch.cam.W = normalize(lookat - rtLaunch.cam.pos);
  rtLaunch.cam.U = normalize(cross(rtLaunch.cam.W, V));
  rtLaunch.cam.V = normalize(cross(rtLaunch.cam.U, rtLaunch.cam.W));

  // copy new ONB vectors back to top level data structure
  cam_U[0] = rtLaunch.cam.U.x;
  cam_U[1] = rtLaunch.cam.U.y;
  cam_U[2] = rtLaunch.cam.U.z;

  cam_V[0] = rtLaunch.cam.V.x;
  cam_V[1] = rtLaunch.cam.V.y;
  cam_V[2] = rtLaunch.cam.V.z;

  cam_W[0] = rtLaunch.cam.W.x;
  cam_W[1] = rtLaunch.cam.W.y;
  cam_W[2] = rtLaunch.cam.W.z;
}


void TachyonOptiX::framebuffer_config(int fbwidth, int fbheight,
                                      int interactive) {
  DBG();
  if (!context_created)
    return;

  width = fbwidth;
  height = fbheight;

  framebuffer.set_size(width * height * sizeof(int));
  accumulation_buffer.set_size(width * height * sizeof(float4));
}


void TachyonOptiX::framebuffer_resize(int fbwidth, int fbheight) {
  DBG();
  if (!context_created)
    return;

  width = fbwidth;
  height = fbheight;

  framebuffer.set_size(width * height * sizeof(int));
  accumulation_buffer.set_size(width * height * sizeof(float4));

  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) framebuffer_resize(%d x %d)\n", width, height);
}


void TachyonOptiX::framebuffer_download_rgb4u(unsigned char *imgrgb4u) {
  DBG();
  framebuffer.download(imgrgb4u, width * height * sizeof(int));
}


void TachyonOptiX::framebuffer_destroy() {
  DBG();
  if (!context_created)
    return;

  framebuffer.free();
  accumulation_buffer.free();
}


void TachyonOptiX::render_compile_and_validate(void) {
  DBG();
  if (!context_created)
    return;

  //
  // finalize context validation, compilation, and AS generation
  //
  double startctxtime = wkf_timer_timenow(rt_timer);

  // (re)build OptiX raygen/hitgroup/miss program pipeline
  if (regen_optix_pipeline) {
    if (pipe != nullptr)
      context_destroy_pipeline();
    context_create_pipeline();
  }

  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during pipeline regen!\n"); 

  double start_AS_build = wkf_timer_timenow(rt_timer);

  free_IAS();
  build_IAS();

  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during AS regen!\n"); 

  // (re)build SBT 
  if (regen_optix_sbt) {
    context_destroy_SBT();
    context_create_SBT();
  }
  time_ctx_AS_build = wkf_timer_timenow(rt_timer) - start_AS_build;

  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during SBT regen!\n"); 

  // upload current materials
  if (regen_optix_materials) {
    materialsBuffer.resize_upload(materialcache);
    regen_optix_materials=0; // no need to re-upload until a change occurs
  }

  // upload current lights
  if (regen_optix_lights) {
    directionalLightsBuffer.resize_upload(directional_lights);
    positionalLightsBuffer.resize_upload(positional_lights);
    regen_optix_lights=0; // no need to re-upload until a change occurs
  }

  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during materials/lights regen!\n"); 

  // XXX 
  // update the launch parameters that we'll pass to the optix launch:
  rtLaunch.frame.size = make_int2(width, height);
  rtLaunch.frame.subframe_index = 1;
  rtLaunch.frame.framebuffer = (uchar4*) framebuffer.cu_dptr();
  rtLaunch.frame.accumulation_buffer = (float4*) accumulation_buffer.cu_dptr();
#if defined(TACHYON_RAYSTATS)
//  rtLaunch.frame.raystats1_buffer = (uint4*) raystats1_buffer.cu_dptr();
//  rtLaunch.frame.raystats2_buffer = (uint4*) raystats2_buffer.cu_dptr();
#endif

  // update material table pointer
  rtLaunch.materials = (rt_material *) materialsBuffer.cu_dptr();

  // finalize camera parms
  rtLaunch.cam.pos = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
  rtLaunch.cam.U   = make_float3(cam_U[0], cam_U[1], cam_U[2]);
  rtLaunch.cam.V   = make_float3(cam_V[0], cam_V[1], cam_V[2]);
  rtLaunch.cam.W   = make_float3(cam_W[0], cam_W[1], cam_W[2]);
  rtLaunch.cam.zoom = cam_zoom;
  rtLaunch.cam.stereo_eyesep = cam_stereo_eyesep;
  rtLaunch.cam.stereo_convergence_dist = cam_stereo_convergence_dist;
  rtLaunch.cam.dof_enabled = cam_dof_enabled;
  rtLaunch.cam.dof_focal_dist = cam_dof_focal_dist;
  rtLaunch.cam.dof_aperture_rad = cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber);

  // populate rtLaunch scene data
  rtLaunch.scene.bg_color = make_float3(scene_bg_color[0],
                                        scene_bg_color[1],
                                        scene_bg_color[2]);
  rtLaunch.scene.bg_color_grad_top = make_float3(scene_bg_grad_top[0],
                                                 scene_bg_grad_top[1],
                                                 scene_bg_grad_top[2]);
  rtLaunch.scene.bg_color_grad_bot = make_float3(scene_bg_grad_bot[0],
                                                 scene_bg_grad_bot[1],
                                                 scene_bg_grad_bot[2]);
  rtLaunch.scene.gradient = make_float3(scene_gradient[0],
                                        scene_gradient[1],
                                        scene_gradient[2]);
  rtLaunch.scene.gradient_topval = scene_gradient_topval;
  rtLaunch.scene.gradient_botval = scene_gradient_botval;

  // this has to be recomputed prior to rendering when topval/botval change
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);
  rtLaunch.scene.gradient_invrange = scene_gradient_invrange;

  rtLaunch.scene.fog_mode = fog_mode;
  rtLaunch.scene.fog_start = fog_start;
  rtLaunch.scene.fog_end = fog_end;
  rtLaunch.scene.fog_density = fog_density;

  rtLaunch.scene.epsilon = scene_epsilon;

  rtLaunch.max_depth = scene_max_depth;
  rtLaunch.max_trans = scene_max_trans;

  rtLaunch.aa_samples = aa_samples;
  rtLaunch.accumulation_normalization_factor = 1.0f / float(aa_samples + 1.0f);
  rtLaunch.accum_count = 0;

  rtLaunch.lights.shadows_enabled = shadows_enabled;
  rtLaunch.lights.ao_samples = ao_samples;
  rtLaunch.lights.ao_ambient = ao_ambient;
  rtLaunch.lights.ao_direct  = ao_direct;
  rtLaunch.lights.ao_maxdist = ao_maxdist;
  rtLaunch.lights.headlight_mode = headlight_mode;

  rtLaunch.lights.num_dir_lights = directional_lights.size();
  rtLaunch.lights.dir_lights = (float3 *) directionalLightsBuffer.cu_dptr();
  rtLaunch.lights.num_pos_lights = positional_lights.size();
  rtLaunch.lights.pos_lights = (float3 *) positionalLightsBuffer.cu_dptr();

  time_ctx_validate = wkf_timer_timenow(rt_timer) - startctxtime;

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) launching render: %d x %d\n", width, height);
  }
}


void TachyonOptiX::update_rendering_state(int interactive) {
  DBG();
  if (!context_created)
    return;

  wkf_timer_start(rt_timer);
}


void TachyonOptiX::render() {
  DBG();
  if (!context_created)
    return;

  update_rendering_state(0);
  render_compile_and_validate();
  double starttime = wkf_timer_timenow(rt_timer);

  //
  // run the renderer
  //
  if (lasterr == OPTIX_SUCCESS) {
    // clear the accumulation buffer

PROFILE_START();
PROFILE_PUSH_RANGE("RenderFrame...", 1);
    // Render to the accumulation buffer for the required number of passes
    launchParamsBuffer.upload(&rtLaunch, 1);

    lasterr = optixLaunch(pipe, stream,
                          launchParamsBuffer.cu_dptr(),
                          launchParamsBuffer.get_size(),
                          &sbt,
                          rtLaunch.frame.size.x,
                          rtLaunch.frame.size.y,
                          1);
PROFILE_POP_RANGE();
PROFILE_STOP();

    // copy the accumulation buffer image data to the framebuffer and perform
    // type conversion and normaliztion on the image data...

    cudaDeviceSynchronize(); // XXX device-wide sync

    rtLaunch.frame.subframe_index++; // advance to next subrame index

    double rtendtime = wkf_timer_timenow(rt_timer);
    time_ray_tracing = rtendtime - starttime;

    if (lasterr != OPTIX_SUCCESS) {
      printf("TachyonOptiX) Error during rendering.  Rendering aborted.\n");
    }

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("TachyonOptiX) ctx setup %.2f  valid %.2f  AS %.2f  RT %.2f io %.2f\n", time_ctx_setup, time_ctx_validate, time_ctx_AS_build, time_ray_tracing, time_image_io);
    }
  } else {
    printf("TachyonOptiX) An error occured prior to rendering. Rendering aborted.\n");
  }
}



//
// A few structure padding/alignment/size diagnostic helper routines
//
void TachyonOptiX::print_internal_struct_info() { 
  printf("TachyonOptiX) internal data structure information\n"); 

  printf("Hitgroup SBT record info:\n");
  printf("SBT record alignment size: %d b\n", OPTIX_SBT_RECORD_ALIGNMENT);
  printf("  total size: %d bytes\n", sizeof(HGRecord));   
  printf("     header size: %d b\n", sizeof(((HGRecord*)0)->header));
  printf("     data offset: %d b\n", offsetof(HGRecord, data));
  printf("       data size: %d b\n", sizeof(((HGRecord*)0)->data));
  printf("     material size: %d b\n", offsetof(HGRecord, data.cone) - offsetof(HGRecord, data.prim_color));
  printf("     geometry size: %d b\n", sizeof(HGRecord) - offsetof(HGRecord, data.trimesh));
  printf("\n");
  printf("   prim_color offset: %d b\n", offsetof(HGRecord, data.prim_color));
  printf("uniform_color offset: %d b\n", offsetof(HGRecord, data.uniform_color));
  printf("materialindex offset: %d b\n", offsetof(HGRecord, data.materialindex));
  printf("     geometry offset: %d b\n", offsetof(HGRecord, data.cone));

  printf("\n");
  printf("geometry union size: %d b\n", sizeof(HGRecord) - offsetof(HGRecord, data.trimesh));
  printf("            cone sz: %d b\n", sizeof(((HGRecord*)0)->data.cone   ));
  printf("             cyl sz: %d b\n", sizeof(((HGRecord*)0)->data.cyl    ));
  printf("            ring sz: %d b\n", sizeof(((HGRecord*)0)->data.ring   ));
  printf("          sphere sz: %d b\n", sizeof(((HGRecord*)0)->data.sphere ));
  printf("         trimesh sz: %d b\n", sizeof(((HGRecord*)0)->data.trimesh));
  printf(" WASTED hitgroup sz: %d b\n", sizeof(HGRecord) - (sizeof(((HGRecord*)0)->header) + sizeof(((HGRecord*)0)->data)));
  printf("\n");
}



//
// geometry instance group management
//
int TachyonOptiX::create_geom_instance_group() {
  TachyonInstanceGroup g = {};
  sceneinstancegroups.push_back(g);
  return int(sceneinstancegroups.size()) - 1;
}

int TachyonOptiX::finalize_geom_instance_group(int idx) {
  TachyonInstanceGroup &g = sceneinstancegroups[idx];
  return 0;
}


int TachyonOptiX::destroy_geom_instance_group(int idx) {
  return 0;
}


#if 0
int TachyonOptiX::set_geom_instance_group_xforms(int idx, int n, float [][16]) {
  return 0;
}
#endif



//
// XXX short-term host API hacks to facilitate early bring-up and testing
//
void TachyonOptiX::add_conearray(int geomidx, ConeArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...
  sceneinstancegroups[geomidx].conearrays.push_back(newmodel);
  regen_optix_sbt=1;
}

void TachyonOptiX::add_cylarray(int geomidx, CylinderArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...
  sceneinstancegroups[geomidx].cyarrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_ringarray(int geomidx, RingArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...
  sceneinstancegroups[geomidx].riarrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_spherearray(int geomidx, SphereArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...
  sceneinstancegroups[geomidx].sparrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_trimesh(int geomidx, TriangleMesh & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...
  sceneinstancegroups[geomidx].meshes.push_back(newmodel);
  regen_optix_sbt=1;
}



//
// XXX short-term host API hacks to facilitate early bring-up and testing
//
void TachyonOptiX::add_conearray(ConeArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  conearrays.push_back(newmodel);
  regen_optix_sbt=1;
}

void TachyonOptiX::add_cylarray(CylinderArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  cyarrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_ringarray(RingArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  riarrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_spherearray(SphereArray & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  sparrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_trimesh(TriangleMesh & newmodel, int materialidx) {
  DBG();
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  meshes.push_back(newmodel);
  regen_optix_sbt=1;
}
