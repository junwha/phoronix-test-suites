project(tbSYCL LANGUAGES CXX)

set(tbSYCL_SRCS ${HIPSDL_CXX_SRCS}
                "main.cpp"
                "FracGenWindow.cpp"
                "FracGen.cpp"
    )

set(tbSYCL_HDRS ${tbSYCL_HDRS}
                "FracGenWindow.hpp"
                "FracGen.hpp"
   )


add_executable(${PROJECT_NAME} ${tbSYCL_SRCS} ${tbSYCL_HDRS})

add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${tbSYCL_SRCS})

target_compile_definitions(${PROJECT_NAME} PRIVATE "__STRICT_ANSI__" "SDL_DISABLE_IMMINTRIN_H")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_compile_options(${PROJECT_NAME} PRIVATE -fcxx-exceptions -std=c++14)
#target_compile_options(${PROJECT_NAME} PRIVATE -fcxx-exceptions -std=c++14 --hipsycl-cuda-path=/opt/cuda -nocudalib --hipsycl-gpu-arch=sm_53)
#target_link_options(${PROJECT_NAME} PRIVATE --hipsycl-cuda-path=/opt/cuda -nocudalib --hipsycl-gpu-arch=gfx906)


#if(HIPSYCL_PLATFORM MATCHES "cuda")
#    target_link_directories(${PROJECT_NAME} PUBLIC /opt/hipsycl/lib/ /opt/cuda-9.2/lib64)

#    target_link_libraries(${PROJECT_NAME} PRIVATE cudart)
#endif()

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)

set_target_properties(${PROJECT_NAME} PROPERTIES    INSTALL_RPATH_USE_LINK_PATH ON
                                                    POSITION_INDEPENDENT_CODE ON    )

#target_include_directories(${PROJECT_NAME} PUBLIC /opt/hipsycl/include/hipSYCL /opt/hipsycl/include /opt/rocm/hsa/include /opt/rocm/hcc/include)


install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
