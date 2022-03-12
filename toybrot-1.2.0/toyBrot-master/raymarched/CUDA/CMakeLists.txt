project(rmCUDA-nvcc LANGUAGES CXX CUDA)


set(NVCC_CXX_SRCS   "main.cpp"
                    "FracGen.hpp"
                    "${TB_COMMON_SRC_DIR}/Vec.hxx"
                    "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                    "${TB_COMMON_SRC_DIR}/defines.hpp"
                    )

set(NVCC_CUDA_SRCS  "FracGen.cu" )


add_executable(${PROJECT_NAME}  ${NVCC_CXX_SRCS} ${NVCC_CUDA_SRCS} )
add_executable(${PROJECT_NAME}-db  ${NVCC_CXX_SRCS} ${NVCC_CUDA_SRCS} )

# More compat problems with nvcc, this lovely tool which never give me issues
# It freaks out if we have too new a version of c++stdlib and because of needing
# an older version, also needs the old std::experimental::filesystem
# It's a great tool, guise

target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<BOOL:TB_STDFS_FOUND>:TB_USE_EXPFS> "TOYBROT_CUDA_NVCC")
target_compile_definitions(${PROJECT_NAME}-db PRIVATE $<$<BOOL:TB_STDFS_FOUND>:TB_USE_EXPFS> "TOYBROT_CUDA_NVCC" "TOYBROT_USE_DOUBLES")

if(NVCC_CXX_INCLUDES_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${NVCC_CXX_INCLUDES_DIR})
    target_include_directories(${PROJECT_NAME}-db PRIVATE ${NVCC_CXX_INCLUDES_DIR})
endif()

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread $<$<BOOL:TB_STDFS_FOUND>:stdc++fs>)
    target_link_libraries(${PROJECT_NAME}-db PRIVATE pthread $<$<BOOL:TB_STDFS_FOUND>:stdc++fs>)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                                   ${CMAKE_CURRENT_SOURCE_DIR}
                                                   ${TB_COMMON_SRC_DIR})

target_include_directories(${PROJECT_NAME}-db PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                                  ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${TB_COMMON_SRC_DIR})



set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME}-db PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TB_SDL_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")
    target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

    target_link_libraries(${PROJECT_NAME}-db PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
    target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

endif()

if(TB_PNG_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

    target_link_libraries(${PROJECT_NAME}-db PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)


if(CMAKE_CXX_COMPILER MATCHES "clang")
    option(BUILD_CUDA_CLANG "Build the CUDA project using clang in addition to the NVCC version" ON)
endif()

if(BUILD_CUDA_CLANG)
    project(rmCUDA-clang LANGUAGES CXX)

    set(CLANG_CXX_SRCS  "main.cpp"
                        "FracGen.hpp"
                        "${TB_COMMON_SRC_DIR}/Vec.hxx"
                        "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                        "${TB_COMMON_SRC_DIR}/defines.hpp"
                        )

    set(CLANG_CUDA_SRCS "${CMAKE_CURRENT_BINARY_DIR}/FracGenClang.cu" )

    # in order to avoid CMake calling nvcc I copy the file over
    # This is because source-file properties are not tied to projects

    add_custom_command( OUTPUT ${CLANG_CUDA_SRCS}
                        COMMAND ${CMAKE_COMMAND} -E remove -f ${CLANG_CUDA_SRCS}
                        COMMAND ${CMAKE_COMMAND} -E copy ${NVCC_CUDA_SRCS} ${CLANG_CUDA_SRCS}
                        COMMENT "Copying FracGen.cu for clang build"
                        DEPENDS ${NVCC_CUDA_SRCS}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    set_source_files_properties(${CLANG_CUDA_SRCS} PROPERTIES
                    GENERATED TRUE
                    LANGUAGE CXX
                    COMPILE_FLAGS "--cuda-path=${CUDA_CLANG_DIR} --cuda-gpu-arch=${NV_GPU_ARCH} -I${CMAKE_CURRENT_SOURCE_DIR}" )


    add_executable(${PROJECT_NAME} ${CLANG_CXX_SRCS} ${CLANG_CUDA_SRCS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<BOOL:TB_STDFS_FOUND>:${TB_FS_DEFINE}> "TOYBROT_CUDA_CLANG")

    add_executable(${PROJECT_NAME}-db ${CLANG_CXX_SRCS} ${CLANG_CUDA_SRCS})
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE $<$<BOOL:TB_STDFS_FOUND>:${TB_FS_DEFINE}> "TOYBROT_CUDA_CLANG" "TOYBROT_USE_DOUBLES")


    target_link_libraries(${PROJECT_NAME} PRIVATE ${TB_STD_LIBS})
    target_link_libraries(${PROJECT_NAME}-db PRIVATE ${TB_STD_LIBS})

    target_include_directories(${PROJECT_NAME} PRIVATE  ${CUDA_CLANG_DIR}/include
                                                        ${CMAKE_CURRENT_SOURCE_DIR}
                                                        ${TB_COMMON_SRC_DIR})
    target_include_directories(${PROJECT_NAME}-db PRIVATE  ${CUDA_CLANG_DIR}/include
                                                           ${CMAKE_CURRENT_SOURCE_DIR}
                                                           ${TB_COMMON_SRC_DIR})

    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${PROJECT_NAME}-db PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(${PROJECT_NAME} PRIVATE dl rt cudart -L${CUDA_CLANG_DIR}/lib64)
    target_link_libraries(${PROJECT_NAME}-db PRIVATE dl rt cudart -L${CUDA_CLANG_DIR}/lib64)

    if(TB_SDL_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)
        target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")
        target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

        target_link_libraries(${PROJECT_NAME}-db PRIVATE SDL2::SDL2)
        target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
        target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")
    endif()

    if(TB_PNG_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
        target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")
        target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

        target_link_libraries(${PROJECT_NAME}-db PRIVATE PNG::PNG)
        target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
        target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
    endif()

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
    install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)


endif()
