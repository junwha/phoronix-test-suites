project(rmWebGL LANGUAGES CXX)

set(rmWebGL_SRCS  "main.cpp"
                    "FracGen.cpp" )

set(rmWebGL_HDRS  "FracGen.hpp"
                    "${TB_COMMON_SRC_DIR}/Vec.hxx"
                    "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                    "${TB_COMMON_SRC_DIR}/defines.hpp" )

set(rmWebGL_GLSRCS "FracGen.frag" )

set(rmWebGL_JSSRCS "${TB_COMMON_SRC_DIR}/download.js")

set(rmWebGL_HTML "${TB_COMMON_SRC_DIR}/toybrot.html" )

if(TB_ALTERNATIVE_SHELL_FILE)
    set(rmWebGL_EMCCEXTRA --shell-file ${rmWebGL_HTML})
endif()

list(APPEND rmWebGL_EMCCEXTRA --extern-pre-js ${rmWebGL_JSSRCS})

add_executable(${PROJECT_NAME} ${rmWebGL_SRCS} ${rmWebGL_HDRS} )
add_executable(${PROJECT_NAME}-mt ${rmWebGL_SRCS} ${rmWebGL_HDRS} )
#add_executable(${PROJECT_NAME}-db ${rmWebGL_SRCS} ${rmWebGL_HDRS} )
add_custom_target(${PROJECT_NAME}_EXTRA_SRCS SOURCES ${rmWebGL_GLSRCS} ${rmWebGL_JSSRCS} ${rmWebGL_HTML})

target_compile_definitions(${PROJECT_NAME} PRIVATE ${TB_EMS_DEFINES} "TB_SINGLETHREAD" "TB_WEBGL_COMPAT" "TB_OPENGL" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-mt PRIVATE ${TB_EMS_DEFINES} "TB_WEBGL_COMPAT" "TB_OPENGL" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
#target_compile_definitions(${PROJECT_NAME}-db PRIVATE ${TB_EMS_DEFINES} "TOYBROT_USE_DOUBLES" "TB_VEC_GL_ALIGN" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)

# "-sOFFSCREENCANVAS_SUPPORT=1" "-sOFFSCREEN_FRAMEBUFFER=1"
target_compile_options(${PROJECT_NAME} PRIVATE --preload-file ${rmWebGL_GLSRCS} ${TB_EMS_OPTIONS})
target_compile_options(${PROJECT_NAME}-mt PRIVATE "-sOFFSCREENCANVAS_SUPPORT=1" --preload-file ${rmWebGL_GLSRCS} ${TB_EMS_OPTIONS} ${TB_EMS_THREADS})
#target_compile_options(${PROJECT_NAME}-db PRIVATE ${TB_EMS_OPTIONS})

target_link_options(${PROJECT_NAME} PRIVATE "-sDISABLE_EXCEPTION_CATCHING=0" "-sMAX_WEBGL_VERSION=2" "-sMIN_WEBGL_VERSION=2" ${rmWebGL_EMCCEXTRA} --preload-file ${rmWebGL_GLSRCS} ${TB_EMS_OPTIONS})
target_link_options(${PROJECT_NAME}-mt PRIVATE "-sOFFSCREENCANVAS_SUPPORT=1" "-sDISABLE_EXCEPTION_CATCHING=0" "-sMAX_WEBGL_VERSION=2" "-sMIN_WEBGL_VERSION=2" ${rmWebGL_EMCCEXTRA} --preload-file ${rmWebGL_GLSRCS} ${TB_EMS_OPTIONS} ${TB_EMS_THREADS})
#target_link_options(${PROJECT_NAME}-db PRIVATE "-sMAX_WEBGL_VERSION=2" "-sMIN_WEBGL_VERSION=2" ${TB_EMS_OPTIONS})

target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")
target_compile_definitions(${PROJECT_NAME}-mt PRIVATE "TOYBROT_ENABLE_GUI")
target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")
target_sources(${PROJECT_NAME}-mt PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

#target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
#target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

#if(TB_PNG_FOUND)
#if(FALSE)
 #   target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
 target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")
 #target_compile_definitions(${PROJECT_NAME}-mt PRIVATE "TOYBROT_ENABLE_PNG")
 target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
 #target_sources(${PROJECT_NAME}-mt PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

 #   target_link_libraries(${PROJECT_NAME}-db PRIVATE PNG::PNG)
    #target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
    #target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
#endif()


# target_link_libraries(${PROJECT_NAME}    PRIVATE ${TB_STD_LIBS} OpenGL::OpenGL OpenGL::EGL)
# target_link_libraries(${PROJECT_NAME}-db PRIVATE ${TB_STD_LIBS} OpenGL::OpenGL OpenGL::EGL)

target_include_directories(${PROJECT_NAME}  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR}
                                            ${TB_EMS_INCLUDES} )

target_include_directories(${PROJECT_NAME}-mt  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR}
                                            ${TB_EMS_INCLUDES} )

#target_include_directories(${PROJECT_NAME}-db  PRIVATE
#                                            ${CMAKE_CURRENT_SOURCE_DIR}
#                                            ${TB_COMMON_SRC_DIR}
#                                            ${TB_EMS_INCLUDES} )

add_custom_command( TARGET ${PROJECT_NAME} PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${rmWebGL_GLSRCS} ${CMAKE_CURRENT_BINARY_DIR}/
                    COMMAND ${CMAKE_COMMAND} -E copy ${rmWebGL_JSSRCS} ${CMAKE_CURRENT_BINARY_DIR}/
                    DEPENDS ${rmWebGL_GLSRCS} ${rmWebGL_JSSRCS}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_command( TARGET ${PROJECT_NAME}-mt PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${rmWebGL_GLSRCS} ${CMAKE_CURRENT_BINARY_DIR}/
                    COMMAND ${CMAKE_COMMAND} -E copy ${rmWebGL_JSSRCS} ${CMAKE_CURRENT_BINARY_DIR}/
                    DEPENDS ${rmWebGL_GLSRCS} ${rmWebGL_JSSRCS}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT WIN32)
    if(CMAKE_CXX_COMPILER MATCHES clang)
        target_compile_options(${PROJECT_NAME} PRIVATE "-fcxx-exceptions")
        #target_compile_options(${PROJECT_NAME}-db PRIVATE "-fcxx-exceptions")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE "-fexceptions")
        #target_compile_options(${PROJECT_NAME}-db PRIVATE "-fexceptions")
    endif()
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-mt RUNTIME DESTINATION bin)
#install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)
install(FILES ${rmWebGL_GLSRCS} DESTINATION bin)
