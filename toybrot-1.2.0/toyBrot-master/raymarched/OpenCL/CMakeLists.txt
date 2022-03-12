project(rmOpenCL LANGUAGES CXX)

set(rm_OpenCL_SRCS  "main.cpp"
                    "FracGen.cpp" )

set(rm_OpenCL_HDRS  "FracGen.hpp"
                    "${TB_COMMON_SRC_DIR}/Vec.hxx"
                    "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                    "${TB_COMMON_SRC_DIR}/defines.hpp" )

set(rm_OpenCL_CLSRCS "FracGen.cl" )


add_executable(${PROJECT_NAME} ${rm_OpenCL_SRCS} ${rm_OpenCL_HDRS} )
add_executable(${PROJECT_NAME}-db ${rm_OpenCL_SRCS} ${rm_OpenCL_HDRS} )
add_custom_target(${PROJECT_NAME}_CL_SRCS SOURCES ${rm_OpenCL_CLSRCS} )

target_compile_definitions(${PROJECT_NAME} PRIVATE "TB_VEC_GL_ALIGN" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_USE_DOUBLES" "TB_VEC_GL_ALIGN" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)

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


target_link_libraries(${PROJECT_NAME}    PRIVATE ${TB_STD_LIBS} OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME}-db PRIVATE ${TB_STD_LIBS} OpenCL::OpenCL)

target_include_directories(${PROJECT_NAME}  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR})

target_include_directories(${PROJECT_NAME}-db  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR})

# Copy the cl file to the binary dir as well so we can debug more easily
add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${rm_OpenCL_CLSRCS} ${CMAKE_CURRENT_BINARY_DIR}/${rm_OpenCL_CLSRCS}
                    DEPENDS ${rm_OpenCL_CLSRCS}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT WIN32)
    if(CMAKE_CXX_COMPILER MATCHES clang)
        target_compile_options(${PROJECT_NAME} PRIVATE "-fcxx-exceptions")
        target_compile_options(${PROJECT_NAME}-db PRIVATE "-fcxx-exceptions")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE "-fexceptions")
        target_compile_options(${PROJECT_NAME}-db PRIVATE "-fexceptions")
    endif()
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)
install(FILES ${rm_OpenCL_CLSRCS} DESTINATION bin)
