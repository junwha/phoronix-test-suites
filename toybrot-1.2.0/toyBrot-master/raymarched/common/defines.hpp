#ifndef _TOYBROT_DEFINES_HPP_DEFINED_
#define _TOYBROT_DEFINES_HPP_DEFINED_

#include <cstddef>
#ifndef TB_SINGLETHREAD
    #include <thread>

    namespace toyBrot
    {
    #ifdef TOYBROT_MAX_THREADS
        constexpr const size_t MaxThreads = TOYBROT_MAX_THREADS;
    #else
        constexpr const size_t MaxThreads = 0;
    #endif

    }
#endif

#ifdef TOYBROT_USE_DOUBLES
    using tbFPType = double;
#else
    using tbFPType = float;
#endif


/**
  * Some languages such as HIP and CUDA
  * end up needing some massaging in the form of
  * function decorations and whatnot. In order to avoid
  * duplicating code (as was in earlier versions) and
  * to not reproduce the specific checks wherever we
  * might need them, I'm centralising those here
  *
  */

#if defined(__HIP_PLATFORM_HCC__) || defined (__HIP_PLATFORM_NVCC__) || defined(__CUDACC__)
    #define _TB_DUAL_ __device__ __host__
    #define _TB_HOST_ __host__
    #define _TB_DEV_  __device__
    #define _TB_CUDA_HIP_
#else
    #define _TB_DUAL_
    #define _TB_HOST_
#endif


#endif //_TOYBROT_DEFINES_HPP_DEFINED_
