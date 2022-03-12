cmake_minimum_required(VERSION 3.9)


set(TB_COMMON_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/common)

if(TB_WASM)
    add_subdirectory(STDTHREADS-WASM)
    add_subdirectory(WebGL)
else(TB_WASM)
    add_subdirectory(STDTASKS)
    add_subdirectory(STDTHREADS)
    add_subdirectory(STDTHREADS-ptr)
    add_subdirectory(STDTHREADS-member)

    if(TB_CUDA_FOUND)
        #add_subdirectory(CUDA)
    endif()

    if(TB_HIP_FOUND)
         add_subdirectory(HIP)
    endif()

    if(TB_ISPC_FOUND)
         option(ISPC_MULTI_TARGET "Build ISPC for separate targets" OFF)
         add_subdirectory(ISPC)
         if(TB_TBB_FOUND)
             add_subdirectory(ISPC-TBB)
         endif()
    endif()

    if(TB_SYCL_FOUND)
#        add_subdirectory(SYCL)
    endif()

    if(TB_OMP_FOUND)
         add_subdirectory(OMP)
    endif()
    #add_subdirectory(MPI)

    if(TB_OCL_FOUND)
         add_subdirectory(OpenCL)
    endif()

    if(TB_GL_FOUND)
        add_subdirectory(OpenGL)
        add_subdirectory(OpenGL-frag)
    endif()

    if(TB_VK_FOUND)
         add_subdirectory(Vulkan)
    endif()

    if(TB_TBB_FOUND)
         add_subdirectory(TBB)
    endif()

    #add_subdirectory(MPICLI)
endif(TB_WASM)
