project(tbHIP LANGUAGES CXX)

set(tbHIP_CXX_SRCS ${tbHIP_CXX_SRCS}
    "main.cpp"
    "FracGenWindow.hpp"
    "FracGenWindow.cpp"
    "FracGen.hpp"
     )


set(tbHIP_HIP_SRCS ${tbHIP_HIP_SRCS}
    "FracGen.cxx"
    )

set(tbHIP_HIPCC_OPTS "")
set(tbHIP_HCC_OPTS   "")
set(tbHIP_NVCC_OPTS  "")

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

    set(tbHIP_NVCC_OPTS "${tbHIP_NVCC_OPTS} -ccbin=/usr/bin/clang++-5.0 --std=c++14 -Xcompiler -Wno-ignored-attributes -x cu --cuda-gpu-arch=sm_52")
    #set_source_files_properties(${tbHIP_HIP_SRCS} PROPERTIES COMPILE_FLAGS "-ccbin=/usr/bin/clang++-5.0 --std=c++14 -Xcompiler -Wno-ignored-attributes -x cu")


endif()

if(HIP_PLATFORM MATCHES cuda)

    set(tbHIP_HIPCC_OPTS "-D__STRICT_ANSI__")

endif()

set_source_files_properties(${tbHIP_HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

message("tbHIP HIPCC_OPTIONS: ${tbHIPCC_OPTS}")
message("tbHIP HCC_OPTIONS:   ${tbHIP_HCC_OPTS}")
message("tbHIP NVCC_OPTS:     ${tbHIP_NVCC_OPTS}")

hip_add_executable(${PROJECT_NAME}-hcc ${tbHIP_CXX_SRCS} ${tbHIP_HIP_SRCS}
                   HIPCC_OPTIONS ${tbHIPCC_OPTS}
                   HCC_OPTIONS   ${tbHIP_HCC_OPTS}
                   #NVCC_OPTIONS  ${tbHIP_NVCC_OPTS}
                   )
target_link_libraries(${PROJECT_NAME}-hcc SDL2::SDL2)
install(TARGETS ${PROJECT_NAME}-hcc RUNTIME DESTINATION bin)

#if(CMAKE_CUDA_COMPILER)
#    hip_add_executable(${PROJECT_NAME}-nvcc ${tbHIP_CXX_SRCS} ${tbHIP_HIP_SRCS}
#                      HIPCC_OPTIONS ${tbHIPCC_OPTS}
#                      #HCC_OPTIONS   ${tbHIP_HCC_OPTS}
#                      NVCC_OPTIONS  ${tbHIP_NVCC_OPTS}
#                      )
#    target_link_libraries(${PROJECT_NAME}-nvcc SDL2::SDL2)
#    install(TARGETS ${PROJECT_NAME}-nvcc RUNTIME DESTINATION bin)
#endif()

add_custom_target(${PROJECT_NAME}HIP SOURCES ${tbHIP_HIP_SRCS})

#if(NOT WIN32)
#    target_link_libraries(${PROJECT_NAME} pthread)
#endif()

#target_compile_definitions(${PROJECT_NAME} PRIVATE "__STRICT_ANSI__")
#target_compile_options(${PROJECT_NAME} PRIVATE "-ccbin=/usr/bin/clang++-5.0 --std=c++14 -Xcompiler -Wno-ignored-attributes -x cu")


