cmake_minimum_required(VERSION 3.9)

if(TB_MPI_FOUND)
    add_subdirectory(MPICLI)
endif()

if(TB_SDL_FOUND)

    add_subdirectory(STDTASKS)
    add_subdirectory(STDTHREADS)


    if(TB_CUDA_FOUND)
        add_subdirectory(CUDA)
    endif()

    if(TB_HIP_FOUND)
        add_subdirectory(HIP)
    endif()

    if(TB_SYCL_FOUND)
        add_subdirectory(SYCL)
    endif()

    if(TB_MPI_FOUND)
        add_subdirectory(MPI)
    endif()

    if(TB_OMP_FOUND)
        add_subdirectory(OMP)
    endif()

    if(TB_OCL_FOUND)
        add_subdirectory(OpenCL)
    endif()

    if(TB_VK_FOUND)
        add_subdirectory(Vulkan)
    endif()
endif()



