project(rmHIP LANGUAGES CXX)

set(tbHIP_CXX_SRCS ${tbHIP_CXX_SRCS}
    "main.cpp"
    "FracGen.hpp"
     )

 set(tbHIP_HIP_HDRS ${tbHIP_HIP_HDRS}
     "${TB_COMMON_SRC_DIR}/Vec.hxx"
     "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
     "${TB_COMMON_SRC_DIR}/defines.hpp"
    )

set(tbHIP_HIP_SRCS ${tbHIP_HIP_SRCS}
    "FracGen.cxx"
    )

#set(tbHIP_HIPCC_OPTS "")
#set(tbHIP_HCC_OPTS   "")
#set(tbHIP_NVCC_OPTS  "")

message("HIP_HIPCONFIG_EXECUTABLE: ${HIP_HIPCONFIG_EXECUTABLE}")
execute_process(
    COMMAND  bash -c "HIP_PLATFORM=hcc ${HIP_HIPCONFIG_EXECUTABLE} -C"
    OUTPUT_VARIABLE "tbHIP_HCC_OPTS"    ERROR_VARIABLE tbHIP_HCC_OPTS
    COMMAND_ECHO STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

if(CMAKE_CUDA_COMPILER)

    execute_process(
        COMMAND bash -c "HIP_PLATFORM=nvcc ${HIP_HIPCONFIG_EXECUTABLE} -C"
        OUTPUT_VARIABLE tbHIP_NVCC_OPTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

    #set(tbHIP_NVCC_OPTS "${tbHIP_NVCC_OPTS} -ccbin=/usr/bin/clang++-5.0 --std=c++14 -Xcompiler -Wno-ignored-attributes -x cu --cuda-gpu-arch=${NV_GPU_ARCH}sm_52")
    #set_source_files_properties(${tbHIP_HIP_SRCS} PROPERTIES COMPILE_FLAGS "-ccbin=/usr/bin/clang++-5.0 --std=c++14 -Xcompiler -Wno-ignored-attributes -x cu")


endif()

set(tbHIPCC_OPTS "${tbHIPCC_OPTS} -std=c++${CMAKE_CXX_STANDARD} -Wno-unused-result")

message("tbHIP HIPCC_OPTIONS: ${tbHIPCC_OPTS}")
message("tbHIP HCC_OPTIONS:   ${tbHIP_HCC_OPTS}")
message("tbHIP NVCC_OPTS:     ${tbHIP_NVCC_OPTS}")


set(tbHIPCC_OPTS-db "${tbHIPCC_OPTS}    -DTOYBROT_USE_DOUBLES")
set(tbHCC_OPTS-db   "${tbHIP_HCC_OPTS}  -DTOYBROT_USE_DOUBLES")
set(tbNVCC_OPTS-db  "${tbHIP_NVCC_OPTS} -DTOYBROT_USE_DOUBLES")

if(TB_FS_DEFINE)
    set(tbHIPCC_OPTS-db "${tbHIPCC_OPTS}    -D${TB_FS_DEFINE}")
    set(tbHCC_OPTS-db   "${tbHIP_HCC_OPTS}  -D${TB_FS_DEFINE}")
    set(tbNVCC_OPTS-db  "${tbHIP_NVCC_OPTS} -D${TB_FS_DEFINE}")
endif()

if(TB_SDL_FOUND)
    set(tbHIP_CXX_SRCS ${tbHIP_CXX_SRCS} "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")
endif()

if(TB_PNG_FOUND)
    set(tbHIP_CXX_SRCS ${tbHIP_CXX_SRCS} "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
endif()

set_source_files_properties(${tbHIP_HIP_SRCS} PROPERTIES COMPILE_OPTIONS "-std=c++17")


set(EXE_NAME ${PROJECT_NAME})
set_source_files_properties(${tbHIP_HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

if(HIP_PLATFORM MATCHES nvcc)

    #set(tbHIP_HIPCC_OPTS "-D__STRICT_ANSI__")
    set_source_files_properties(${tbHIP_CXX_SRCS} PROPERTIES COMPILE_FLAGS "${tbHIP_NVCC_OPTS}")


    set(EXE_NAME ${PROJECT_NAME}-nvcc)

    hip_add_executable(${EXE_NAME} ${tbHIP_CXX_SRCS} ${tbHIP_HIP_SRCS} ${tbHIP_HIP_HDRS}
                       #HIPCC_OPTIONS ${tbHIPCC_OPTS}
                       #HCC_OPTIONS   ${tbHIP_HCC_OPTS}
                       NVCC_OPTIONS  ${tbHIP_NVCC_OPTS}
                       )
else()

    set(EXE_NAME ${PROJECT_NAME}-hcc)

    set_source_files_properties(${tbHIP_CXX_SRCS} PROPERTIES COMPILE_DEFINITIONS "__HIP_PLATFORM_HCC__")


    hip_add_executable(${EXE_NAME} ${tbHIP_CXX_SRCS} ${tbHIP_HIP_SRCS} ${tbHIP_HIP_HDRS}
                       HIPCC_OPTIONS ${tbHIPCC_OPTS}
                       HCC_OPTIONS   ${tbHIP_HCC_OPTS}
                       #NVCC_OPTIONS  ${tbHIP_NVCC_OPTS}
                      )
    hip_add_executable(${EXE_NAME}-db ${tbHIP_CXX_SRCS} ${tbHIP_HIP_SRCS} ${tbHIP_HIP_HDRS}
                       HIPCC_OPTIONS  ${tbHIPCC_OPTS-db}
                       HCC_OPTIONS    ${tbHIP_HCC_OPTS-db}
                       #NVCC_OPTIONS  ${tbHIP_NVCC_OPTS}
                      )
endif()

target_compile_options(${EXE_NAME} PRIVATE "-Wno-unused-result")
target_compile_options(${EXE_NAME}-db PRIVATE "-Wno-unused-result")

target_link_libraries(${EXE_NAME} PRIVATE ${TB_STD_LIBS})
target_link_libraries(${EXE_NAME}-db PRIVATE ${TB_STD_LIBS})

target_compile_definitions(${EXE_NAME}-db PRIVATE "TOYBROT_USE_DOUBLES")


if(TB_SDL_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE SDL2::SDL2)
    target_compile_definitions(${EXE_NAME} PRIVATE "TOYBROT_ENABLE_GUI")

    target_link_libraries(${EXE_NAME}-db PRIVATE SDL2::SDL2)
    target_compile_definitions(${EXE_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
endif()

if(TB_PNG_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${EXE_NAME} PRIVATE "TOYBROT_ENABLE_PNG")

    target_link_libraries(${EXE_NAME}-db PRIVATE PNG::PNG)
    target_compile_definitions(${EXE_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
endif()

install(TARGETS ${EXE_NAME} RUNTIME DESTINATION bin)

install(TARGETS ${EXE_NAME}-db RUNTIME DESTINATION bin)

#install(FILES   ${tbHIP_CONF} DESTINATION bin/toybrot.conf.d)

target_include_directories(${EXE_NAME} PRIVATE  "/home/lasagna/rocm/aomp/include"
                                                ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${TB_COMMON_SRC_DIR}        )


target_include_directories(${EXE_NAME}-db PRIVATE "/home/lasagna/rocm/aomp/include"
                                                  ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${TB_COMMON_SRC_DIR}        )


add_custom_target(${PROJECT_NAME}HIP SOURCES ${tbHIP_HIP_SRCS})


