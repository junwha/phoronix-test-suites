project(rmSYCL LANGUAGES SYCL)

set(rmSYCL_SRCS ${rmSYCL_SRCS}
                "main.cpp"
                "FracGen.cpp"
    )

set(rmSYCL_HDRS ${rmSYCL_HDRS}
                "FracGen.hpp"
                "Vec.hxx"
   )


if(TB_SDL_FOUND)
    set(rmSYCL_SRCS ${rmSYCL_SRCS} "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" )
    set(rmSYCL_HDRS ${rmSYCL_HDRS} "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp" )

endif()

if(TB_PNG_FOUND)
    set(rmSYCL_SRCS ${rmSYCL_SRCS} "${TB_COMMON_SRC_DIR}/pngWriter.cpp" )
    set(rmSYCL_HDRS ${rmSYCL_HDRS} "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
endif()


# The code in the else is dependent on changes I haven't discussed with the hipSYCL
# maintainer and haven't contributed to upstream at this moment
if (FALSE)

    add_executable(${PROJECT_NAME} ${rmSYCL_SRCS} ${rmSYCL_HDRS})

    target_compile_definitions(${PROJECT_NAME} PRIVATE "SDL_DISABLE_IMMINTRIN_H")
    target_compile_options(${PROJECT_NAME} PRIVATE -fcxx-exceptions)

    if(TB_SDL_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)
        target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")
    endif()

    if(TB_PNG_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
        target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")
    endif()

    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
    add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${rmSYCL_SRCS})

    set_target_properties(${PROJECT_NAME} PROPERTIES   INSTALL_RPATH_USE_LINK_PATH ON
                                                       POSITION_INDEPENDENT_CODE ON    )

    target_include_directories(${PROJECT_NAME}  PRIVATE
                                                ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${TB_COMMON_SRC_DIR})

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

else()
    add_executable(${PROJECT_NAME}-rocm ${rmSYCL_SRCS} ${rmSYCL_HDRS})
    add_executable(${PROJECT_NAME}-cuda ${rmSYCL_SRCS} ${rmSYCL_HDRS})
    add_executable(${PROJECT_NAME}-cpu  ${rmSYCL_SRCS} ${rmSYCL_HDRS})

    set(CMAKE_SYCL_FLAGS "")
    set_target_properties(${PROJECT_NAME}-cuda PROPERTIES HIPSYCL_PLATFORM "cuda"
                                                          COMPILE_FLAGS "--hipsycl-platform=cuda --hipsycl-gpu-arch=${NV_GPU_ARCH} --cuda-path=${CUDA_CLANG_DIR} -fPIE"
                                                          LINK_FLAGS    "--hipsycl-cuda-link-line=-L${CUDA_CLANG_DIR}/lib64 -lcudart")

    set_target_properties(${PROJECT_NAME}-rocm PROPERTIES HIPSYCL_PLATFORM "rocm"
                                                          COMPILE_FLAGS "--hipsycl-platform=rocm --hipsycl-rocm-path=${HIPSYCL_ROCM_PATH}")

    set_target_properties(${PROJECT_NAME}-cpu PROPERTIES  HIPSYCL_PLATFORM "cpu"
                                                          COMPILE_FLAGS "--hipsycl-platform=cpu")

    add_sycl_to_target(TARGET ${PROJECT_NAME}-rocm SOURCES ${rmSYCL_SRCS})
    add_sycl_to_target(TARGET ${PROJECT_NAME}-cuda SOURCES ${rmSYCL_SRCS})
    add_sycl_to_target(TARGET ${PROJECT_NAME}-cpu  SOURCES ${rmSYCL_SRCS})


    target_compile_definitions(${PROJECT_NAME}-rocm PRIVATE "SDL_DISABLE_IMMINTRIN_H" "TOYBROT_HIPSYCL_FLAVOUR=\"hipSYCL-ROCM\"" )
    target_compile_definitions(${PROJECT_NAME}-cuda PRIVATE "SDL_DISABLE_IMMINTRIN_H" "TOYBROT_HIPSYCL_FLAVOUR=\"hipSYCL-CUDA\"" )
    target_compile_definitions(${PROJECT_NAME}-cpu  PRIVATE "SDL_DISABLE_IMMINTRIN_H" "TOYBROT_HIPSYCL_FLAVOUR=\"hipSYCL-CPU\"" )
    target_compile_options(${PROJECT_NAME}-rocm PRIVATE -fcxx-exceptions)
    target_compile_options(${PROJECT_NAME}-cuda PRIVATE -fcxx-exceptions)
    target_compile_options(${PROJECT_NAME}-cpu  PRIVATE -fcxx-exceptions)
    target_link_libraries(${PROJECT_NAME}-rocm PRIVATE pthread)
    target_link_libraries(${PROJECT_NAME}-cuda PRIVATE pthread)
    target_link_libraries(${PROJECT_NAME}-cpu  PRIVATE pthread)
    target_include_directories(${PROJECT_NAME}-rocm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TB_COMMON_SRC_DIR})
    target_include_directories(${PROJECT_NAME}-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TB_COMMON_SRC_DIR})
    target_include_directories(${PROJECT_NAME}-cpu  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TB_COMMON_SRC_DIR})

    if(TB_SDL_FOUND)
        target_link_libraries(${PROJECT_NAME}-rocm PRIVATE SDL2::SDL2)
        target_link_libraries(${PROJECT_NAME}-cuda PRIVATE SDL2::SDL2)
        target_link_libraries(${PROJECT_NAME}-cpu  PRIVATE SDL2::SDL2)
        target_compile_definitions(${PROJECT_NAME}-rocm PRIVATE "TOYBROT_ENABLE_GUI")
        target_compile_definitions(${PROJECT_NAME}-cuda PRIVATE "TOYBROT_ENABLE_GUI")
        target_compile_definitions(${PROJECT_NAME}-cpu  PRIVATE "TOYBROT_ENABLE_GUI")
    endif()

    if(TB_PNG_FOUND)
        target_link_libraries(${PROJECT_NAME}-rocm PRIVATE PNG::PNG)
        target_link_libraries(${PROJECT_NAME}-cuda PRIVATE PNG::PNG)
        target_link_libraries(${PROJECT_NAME}-cpu  PRIVATE PNG::PNG)
        target_compile_definitions(${PROJECT_NAME}-rocm PRIVATE "TOYBROT_ENABLE_PNG")
        target_compile_definitions(${PROJECT_NAME}-cuda PRIVATE "TOYBROT_ENABLE_PNG")
        target_compile_definitions(${PROJECT_NAME}-cpu  PRIVATE "TOYBROT_ENABLE_PNG")
    endif()


    set_target_properties(${PROJECT_NAME}-rocm PROPERTIES   INSTALL_RPATH_USE_LINK_PATH ON
                                                            POSITION_INDEPENDENT_CODE ON    )
    set_target_properties(${PROJECT_NAME}-cuda PROPERTIES   INSTALL_RPATH_USE_LINK_PATH ON
                                                            POSITION_INDEPENDENT_CODE ON    )
    set_target_properties(${PROJECT_NAME}-cpu PROPERTIES    INSTALL_RPATH_USE_LINK_PATH ON
                                                            POSITION_INDEPENDENT_CODE ON    )


    install(TARGETS ${PROJECT_NAME}-rocm RUNTIME DESTINATION bin)
    install(TARGETS ${PROJECT_NAME}-cuda RUNTIME DESTINATION bin)
    install(TARGETS ${PROJECT_NAME}-cpu  RUNTIME DESTINATION bin)
endif()
