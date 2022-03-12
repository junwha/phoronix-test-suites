cmake_minimum_required(VERSION 3.9)
project(Toybrot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/redist)

###############################################################################
#
# This messy bit is a bunch of find_package calls and work around that
#
###############################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

set(TB_WASM FALSE)

if( CMAKE_CXX_COMPILER MATCHES "emcc" OR
CMAKE_CXX_COMPILER MATCHES "em\\+\\+")
    set(TB_WASM TRUE)
else()

    find_package(SDL2)

    find_package(OpenGL COMPONENTS OpenGL EGL)

    include(CheckLanguage)
    check_language(CUDA)

    list(APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake)
    list(APPEND CMAKE_MODULE_PATH /opt/rocm)
    list(APPEND CMAKE_MODULE_PATH /home/lasagna/rocm/aomp/cmake/)


    set(HIP_PLATFORM "clang" CACHE STRING "Platform to deploy HIP to (if found)")
    set_property(CACHE HIP_PLATFORM PROPERTY STRINGS "hcc;clang;nvcc")

    set(ENV{HIP_PLATFORM} ${HIP_PLATFORM})

    find_package(HIP MODULE)
    find_package(OpenCL)
    find_package(Vulkan)
    list(APPEND CMAKE_PREFIX_PATH /opt/hipsycl/lib)
    list(APPEND CMAKE_PREFIX_PATH /opt/hipSYCL/lib)
    #find_package(hipsycl CONFIG)
    find_package(PNG)

    if(NOT CMAKE_CXX_COMPILER MATCHES "aomp")
        # The standard packages don't deal too well with the
        # AOMP compiler
        find_package(MPI)
        find_package(OpenMP)
    endif()

    find_package(TBB)

    set(ISPC_HINT "" CACHE PATH "Path to your ISPC installation if it isn't on your PATH")


    find_program(ISPC_COMPILER "ispc" DOC "Intel ISPC compiler" HINTS ${ISPC_HINT})
    if(ISPC_COMPILER)
        execute_process(COMMAND ${ISPC_COMPILER} --version
                        OUTPUT_VARIABLE ISPC_VERSION)
        string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]" ISPC_VERSION ${ISPC_VERSION})
    endif()

endif()


if(NOT WIN32)
    set(TB_STD_LIBS pthread)
else()
    set(TB_STD_LIBS "")
endif()

set(TB_NUM_THREADS "0" CACHE STRING "Number of threads to use (0 for auto)")

###############################################################################
#
# Support for std::filesystem is still iffy but I still want to use it so...
#
###############################################################################

try_compile(TB_STDFS ${CMAKE_CURRENT_BINARY_DIR} SOURCES "${CMAKE_SOURCE_DIR}/stdfilesystemtest.cpp"
                     CXX_STANDARD ${CMAKE_CXX_STANDARD}
                     CXX_STANDARD_REQUIRED ON
                     CXX_EXTENSIONS OFF)


try_compile(TB_EXPFS ${CMAKE_CURRENT_BINARY_DIR} SOURCES "${CMAKE_SOURCE_DIR}/stdfilesystemtest.cpp"
                     COMPILE_DEFINITIONS -DEXPERIMENTAL
                     LINK_LIBRARIES stdc++fs
                     CXX_STANDARD ${CMAKE_CXX_STANDARD}
                     CXX_EXTENSIONS OFF)

###############################################################################
#
# Actual management of the found stuff begins here
#
###############################################################################

set(TB_STDFS_FOUND FALSE)
set(TB_CUDA_FOUND  FALSE)
set(TB_GL_FOUND    FALSE)
set(TB_HIP_FOUND   FALSE)
set(TB_ISPC_FOUND  FALSE)
set(TB_MPI_FOUND   FALSE)
set(TB_OCL_FOUND   FALSE)
set(TB_OMP_FOUND   FALSE)
set(TB_PNG_FOUND   FALSE)
set(TB_SDL_FOUND   FALSE)
set(TB_SYCL_FOUND  FALSE)
set(TB_TBB_FOUND   FALSE)
set(TB_VK_FOUND    FALSE)

message(" ")
message("*********************************")
message("Toybrot dependency probe")
message(" ")

if(TB_STDFS)
    message("STD::FILESYSTEM: OK")
    set(TB_FS_DEFINE "TB_USE_STDFS")
    set(TB_STDFS_FOUND TRUE)
elseif(TB_EXPFS)
    message("STD::FILESYSTEM: OK(EXPERIMENTAL)")
    list(APPEND TB_STD_LIBS stdc++fs)
    set(TB_FS_DEFINE "TB_USE_EXPFS")
    set(TB_STDFS_FOUND TRUE)
else()
    message("STD::FILESYSTEM: NOT FOUND")
    # The functionality we lose without std::filesystem is having the applications create the
    # directory for the config files. So I put in this cmake script that creates it itself;
    # It won't get recreated if the directory gets deleted but at least everything is pretty
    # much fully functional from a fresh deploy
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/bin/toybrot.conf.d)")
endif()


if(TB_WASM)
    message("emscripten detected, building for Web Assembly")
    message("(this is a different set of projects)")


    # Please see common/toybrot.html for info on this option
    # you probably don't need to worry about it
    option(TB_ALTERNATIVE_SHELL_FILE OFF)

    ###############################################################################
    #
    # Emscripten requires some specific tweaks
    #
    ###############################################################################

    set(CMAKE_C_FLAGS_DEBUG "-g4 -sDEMANGLE_SUPPORT=1 -sSAFE_HEAP=1 --source-map-base http://localhost/")
    set(CMAKE_CXX_FLAGS_DEBUG "-g4 -sDEMANGLE_SUPPORT=1 -sSAFE_HEAP=1 --source-map-base http://localhost/")

    set(CMAKE_EXECUTABLE_SUFFIX ".html")

    if(TB_NUM_THREADS EQUAL 0)
        message("Web Assembly requires a predefined number of threads")
        message("Defaulting to 120")
        set(TB_NUM_THREADS "120" CACHE STRING "Number of threads to use (0 for auto)" FORCE)
    endif()

    # PROXY_TO_PTHREAD should(?) be used but getting "screen is undefined"
    set(TB_EMS_THREADS -sUSE_PTHREADS=1 -sPTHREAD_POOL_SIZE=${TB_NUM_THREADS} -sALLOW_BLOCKING_ON_MAIN_THREAD=1 )
    set(TB_EMS_OPTIONS -sEXIT_RUNTIME=1 -sUSE_SDL=2 -sEXTRA_EXPORTED_RUNTIME_METHODS=['FS'] -sDISABLE_EXCEPTION_CATCHING=0 -sTOTAL_STACK=100mb -sINITIAL_MEMORY=700mb --use-preload-plugins --bind -sUSE_LIBPNG=1 -sASSERTIONS=1 )
    set(TB_EMS_DEFINES "TOYBROT_MAX_THREADS=${TB_NUM_THREADS}")

    set(TB_EMS_ROOT "/usr/lib" CACHE PATH "Path to your Emscripten installation")
    set(TB_EMS_PORTS_INCLUDE "$ENV{HOME}/.emscripten_cache/wasm/include" CACHE PATH "Path to your Emscripten ports cache installation")
    # Emscripten should find these but your IDE might not, so this is convenience
    # I know Qt Creator doesn't, at the very least
    set(TB_EMS_INCLUDES "${TB_EMS_ROOT}/emscripten/system/include/libc"
                        "${TB_EMS_ROOT}/emscripten/system/include/libcxx"
                        "${TB_EMS_ROOT}/emscripten/system/include"
                        "${TB_EMS_PORTS_INCLUDE}" )

else()

    if(CMAKE_CUDA_COMPILER)
        message("CUDA: OK")
        set(TB_CUDA_FOUND TRUE)
    else()
        message("CUDA: NOT FOUND")
    endif()

    if(HIP_FOUND)
        set(TB_HIP_FOUND  TRUE)
        message("HIP: OK (${HIP_VERSION})")
    else()
        message("HIP: NOT FOUND")
    endif()

    if(OPENGL_FOUND)
        set(TB_GL_FOUND  TRUE)
        message("OpenGL: OK (EGL)")
    else()
        message("OpenGL: NOT FOUND")
    endif()

    if(ISPC_COMPILER)
        set(TB_ISPC_FOUND  TRUE)
        message("ISPC: OK (${ISPC_VERSION})")
    else()
        message("ISPC: NOT FOUND")
    endif()

    if(TARGET MPI::MPI_CXX)
        set(TB_MPI_FOUND  TRUE)
        message("MPI: OK")
    else()
        message("MPI: NOT FOUND")
    endif()

    if(TARGET OpenCL::OpenCL)
        set(TB_OCL_FOUND  TRUE)
        message("OpenCL: OK")
    else()
        message("OpenCL: NOT FOUND")
    endif()

    if(TARGET OpenMP::OpenMP_CXX OR CMAKE_CXX_COMPILER MATCHES "aomp")
        set(TB_OMP_FOUND  TRUE)
        message("OpenMP: OK")
    else()
        message("OpenMP: NOT FOUND")
    endif()

    if(TARGET PNG::PNG)
        set(TB_PNG_FOUND TRUE)
        message("PNG: OK")
    else()
        message("PNG: NOT FOUND")
    endif()

    if(TARGET SDL2::SDL2)
        set(TB_SDL_FOUND TRUE)
        message("SDL2: OK")
    else()
        message("SDL2: NOT FOUND")
    endif()

    if(HIPSYCL_SYCLCC)
        set(TB_SYCL_FOUND TRUE)
        message("SYCL: OK (hipSYCL)")
    else()
        message("SYCL: NOT FOUND")
    endif()

    if(TARGET TBB::tbb)
        set(TB_TBB_FOUND  TRUE)
        message("TBB: OK")
    else()
        message("TBB: NOT FOUND")
    endif()

    if(TARGET Vulkan::Vulkan)
        find_program(SLANGCOMPILER "glslangValidator" DOC "spir-v shader compiler")

        if(NOT SLANGCOMPILER)
            message("Vulkan: NOT FOUND")
            message("Could not find Vulkan shader compiler (glslangValidator)" )
        else()
            message("Vulkan: OK")
            set(TB_VK_FOUND TRUE)
        endif()
    else()
        message("Vulkan: NOT FOUND")
    endif()
endif()

message(" ")
message("*********************************")
message(" ")


if(CMAKE_CXX_COMPILER MATCHES "hcc")
    execute_process(
        COMMAND hcc-config --cxxflags
        OUTPUT_VARIABLE HCC_CXX_OPTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

    execute_process(
        COMMAND hcc-config  --ldflags
        OUTPUT_VARIABLE HCC_LD_OPTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
endif()

set(AMD_GPU_ARCH "gfx906" CACHE STRING "ROCm GPU arch string for compilers")
set(NV_GPU_ARCH  "sm_52"  CACHE STRING "CUDA GPU arch string for compilers")

if(CMAKE_CUDA_COMPILER)
    set(CUDA_PATH $ENV{CUDA_PATH})
    if(NOT CUDA_PATH)
        set(CUDA_PATH "/opt/cuda")
    endif()
    set(CUDA_CLANG_DIR        "${CUDA_PATH}" CACHE PATH "Path to your CUDA toolkit for Clang/hipSYCL")
    set(NVCC_CXX_INCLUDES_DIR "/usr/lib/gcc/x86_64-pc-linux-gnu/8.4.0/include/c++" CACHE PATH "Path to alternative C++ std headers if yours are too new")
endif()


set(ENABLE2D OFF CACHE BOOL "Enable legacy mandelbrot projects")

add_subdirectory(raymarched)
if(ENABLE2D)
    add_subdirectory(mandelbrot)
endif()
