project(rmSTD_THREADS-WASM LANGUAGES CXX)


set(rmSTD_THREADS-WASM_SRCS "main.cpp"
                            "FracGen.cpp" )

set(rmSTD_THREADS-WASM_HDRS "FracGen.hpp"
                            "${TB_COMMON_SRC_DIR}/Vec.hxx"
                            "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                            "${TB_COMMON_SRC_DIR}/defines.hpp" )


set(rmSTD_THREADS-WASM_JSSRCS "${TB_COMMON_SRC_DIR}/download.js")


set(rmSTD_THREADS-WASM_HTML "${TB_COMMON_SRC_DIR}/toybrot.html" )

if(TB_ALTERNATIVE_SHELL_FILE)
    set(rmSTD_THREADS_EMCCEXTRA --shell-file ${rmSTD_THREADS-WASM_HTML})
endif()
list(APPEND rmSTD_THREADS_EMCCEXTRA --extern-pre-js ${rmSTD_THREADS-WASM_JSSRCS})

add_executable(${PROJECT_NAME} ${rmSTD_THREADS-WASM_SRCS} ${rmSTD_THREADS-WASM_HDRS} )
add_executable(${PROJECT_NAME}-db ${rmSTD_THREADS-WASM_SRCS} ${rmSTD_THREADS-WASM_HDRS} )

add_custom_target(${PROJECT_NAME}_EXTRA_SRCS SOURCES ${rmSTD_THREADS-WASM_JSSRCS} ${rmSTD_THREADS-WASM_HTML})

target_compile_definitions(${PROJECT_NAME} PRIVATE ${TB_EMS_DEFINES}  "TB_WEBGL_COMPAT" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-db PRIVATE ${TB_EMS_DEFINES} "TB_WEBGL_COMPAT" "TOYBROT_USE_DOUBLES" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TB_STD_LIBS})
target_link_libraries(${PROJECT_NAME}-db PRIVATE ${TB_STD_LIBS})

target_compile_options(${PROJECT_NAME} PRIVATE ${rmSTD_THREADS_EMCCEXTRA} ${TB_EMS_OPTIONS} ${TB_EMS_THREADS})
target_compile_options(${PROJECT_NAME}-db PRIVATE ${rmSTD_THREADS_EMCCEXTRA} ${TB_EMS_OPTIONS} ${TB_EMS_THREADS})

target_link_options(${PROJECT_NAME} PRIVATE ${rmSTD_THREADS_EMCCEXTRA}  ${TB_EMS_OPTIONS} ${TB_EMS_THREADS})
target_link_options(${PROJECT_NAME}-db PRIVATE ${rmSTD_THREADS_EMCCEXTRA} ${TB_EMS_OPTIONS} ${TB_EMS_THREADS})

target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")
target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")


#if(TB_PNG_FOUND)
#if(FALSE)
    #target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

    #target_link_libraries(${PROJECT_NAME}-db PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
#endif()

#message("TB_EMS_INCLUDES: ${TB_EMS_INCLUDES}")

add_custom_command( TARGET ${PROJECT_NAME} PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${rmSTD_THREADS-WASM_JSSRCS} ${CMAKE_CURRENT_BINARY_DIR}/
                    DEPENDS ${rmSTD_THREADS-WASM_JSSRCS} )

target_include_directories(${PROJECT_NAME}  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR}
                                            ${TB_EMS_INCLUDES} )

#get_target_property(PROJ_INCLUDES ${PROJECT_NAME} INCLUDE_DIRECTORIES)

#message("PROJ_INCLUDES: ${PROJ_INCLUDES}")

target_include_directories(${PROJECT_NAME}-db PRIVATE
                                              ${CMAKE_CURRENT_SOURCE_DIR}
                                              ${TB_COMMON_SRC_DIR}
                                              ${TB_EMS_INCLUDES} )

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)
