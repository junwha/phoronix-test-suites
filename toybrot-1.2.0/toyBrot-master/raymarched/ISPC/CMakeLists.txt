project(rmISPC LANGUAGES CXX)

set(rmISPC_CXX_SRCS "main.cpp"
                    "FracGen.hpp"
                    "${TB_COMMON_SRC_DIR}/Vec.hxx"
                    "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                    "${TB_COMMON_SRC_DIR}/defines.hpp" )

set(rmISPC_ISPC_SRCS  "FracGen.ispc")

set(rmISPC_OBJS      "FracGen.ispc.o" )
set(rmISPC_OBJS_DB   "FracGen.ispc.db.o" )

set(ISPC_TARGETS    "host"
                    "sse2-i32x4"
                    "sse2-i32x8"
                    "sse4-i32x4"
                    "sse4-i32x8"
                    "sse4-i16x8"
                    "sse4-i8x16"
                    "avx1-i32x4"
                    "avx1-i32x8"
                    "avx1-i32x16"
                    "avx1-i64x4"
                    "avx2-i32x4"
                    "avx2-i32x8"
                    "avx2-i32x16"
                    "avx2-i64x4"
                    "avx512knl-i32x16"
                    "avx512skx-i32x16"
                    "avx512skx-i32x8"
                    # These are for ARM
                    #"neon-i8x16"
                    #"neon-i16x8"
                    #"neon-i32x4"
                    #"neon-i32x8"
                    )

# CMake trickery to avoid recompiling all the regular source files for
# every different flavour target when only the ISPC bit is different

add_custom_command( OUTPUT FracGen.ispc.h
                    COMMAND ${ISPC_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.ispc
                                       --target=host
                                       -h ${CMAKE_CURRENT_BINARY_DIR}/FracGen_ispc.h
                    MAIN_DEPENDENCY FracGen.ispc  )


add_custom_command( OUTPUT FracGen.ispc.db.h
                    COMMAND ${ISPC_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.ispc
                                       --target=host
                                       -DTOYBROT_USE_DOUBLES
                                       -h ${CMAKE_CURRENT_BINARY_DIR}/FracGen_ispc.db.h
                    MAIN_DEPENDENCY FracGen.ispc  )

add_library(ISPC_CXX_OBJS OBJECT ${rmISPC_CXX_SRCS} FracGen.ispc.h)
target_include_directories(ISPC_CXX_OBJS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${TB_COMMON_SRC_DIR}/)
target_compile_definitions(ISPC_CXX_OBJS PUBLIC $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)


add_library(ISPC_CXX_OBJS_DB OBJECT ${rmISPC_CXX_SRCS} FracGen.ispc.db.h)
target_compile_definitions(ISPC_CXX_OBJS_DB PUBLIC "TOYBROT_USE_DOUBLES" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_include_directories(ISPC_CXX_OBJS_DB PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${TB_COMMON_SRC_DIR}/)



target_link_libraries( ISPC_CXX_OBJS PUBLIC ${TB_STD_LIBS} )
target_link_libraries( ISPC_CXX_OBJS_DB PUBLIC ${TB_STD_LIBS} )


if(TB_SDL_FOUND)
    target_link_libraries( ISPC_CXX_OBJS PUBLIC SDL2::SDL2 )
    target_compile_definitions(ISPC_CXX_OBJS PUBLIC "TOYBROT_ENABLE_GUI")
    target_sources(ISPC_CXX_OBJS PRIVATE  "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

    target_link_libraries( ISPC_CXX_OBJS_DB PUBLIC SDL2::SDL2 )
    target_compile_definitions(ISPC_CXX_OBJS_DB PUBLIC "TOYBROT_ENABLE_GUI")
    target_sources(ISPC_CXX_OBJS_DB PRIVATE  "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")
endif()

if(TB_PNG_FOUND)
    target_link_libraries(ISPC_CXX_OBJS PUBLIC PNG::PNG)
    target_compile_definitions(ISPC_CXX_OBJS PUBLIC "TOYBROT_ENABLE_PNG")
    target_sources(ISPC_CXX_OBJS PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

    target_link_libraries(ISPC_CXX_OBJS_DB PUBLIC PNG::PNG)
    target_compile_definitions(ISPC_CXX_OBJS_DB PUBLIC "TOYBROT_ENABLE_PNG")
    target_sources(ISPC_CXX_OBJS_DB PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
endif()

if(ISPC_MULTI_TARGET)
    foreach(ISPC_TGT IN LISTS ISPC_TARGETS)

        set(ISPC_FLAGS "--target=${ISPC_TGT} --pic")
        set(ISPC_FLAGS_DB "--target=${ISPC_TGT} --pic -DTOYBROT_USE_DOUBLES")
        separate_arguments(ISPC_FLAGS UNIX_COMMAND ${ISPC_FLAGS})
        separate_arguments(ISPC_FLAGS_DB UNIX_COMMAND ${ISPC_FLAGS_DB})

        add_custom_command( OUTPUT FracGen.ispc.${ISPC_TGT}.o
                            COMMAND ${ISPC_COMPILER} ${ISPC_FLAGS}
                                                $<IF:$<CONFIG:Debug>,-O0,-O3>
                                                $<IF:$<CONFIG:Release>,,-g>
                                                ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.ispc
                                               -o ${CMAKE_CURRENT_BINARY_DIR}/FracGen.ispc.${ISPC_TGT}.o
                            MAIN_DEPENDENCY FracGen.ispc  )

        add_custom_command( OUTPUT FracGen.ispc.${ISPC_TGT}.db.o
                            COMMAND ${ISPC_COMPILER} ${ISPC_FLAGS_DB}
                                                $<IF:$<CONFIG:Debug>,-O0,-O3>
                                                $<IF:$<CONFIG:Release>,,-g>
                                                ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.ispc
                                               -o ${CMAKE_CURRENT_BINARY_DIR}/FracGen.ispc.${ISPC_TGT}.db.o
                            MAIN_DEPENDENCY FracGen.ispc  )

        add_executable(${PROJECT_NAME}-${ISPC_TGT}    FracGen.ispc.${ISPC_TGT}.o    "FracGen.cpp" )
        add_executable(${PROJECT_NAME}-${ISPC_TGT}-db FracGen.ispc.${ISPC_TGT}.db.o "FracGen.cpp" )
        target_link_libraries(${PROJECT_NAME}-${ISPC_TGT}    PUBLIC ISPC_CXX_OBJS)
        target_link_libraries(${PROJECT_NAME}-${ISPC_TGT}-db PUBLIC ISPC_CXX_OBJS_DB)
        target_compile_definitions(${PROJECT_NAME}-${ISPC_TGT} PRIVATE _TOYBROT_ISPC_ARCH_=\"${ISPC_TGT}\")
        target_compile_definitions(${PROJECT_NAME}-${ISPC_TGT}-db PRIVATE _TOYBROT_ISPC_ARCH_=\"${ISPC_TGT}\" "TOYBROT_USE_DOUBLES")

        install(TARGETS ${PROJECT_NAME}-${ISPC_TGT}    RUNTIME DESTINATION bin)
        install(TARGETS ${PROJECT_NAME}-${ISPC_TGT}-db RUNTIME DESTINATION bin)

    endforeach()
else()
    set(ISPC_ARCH "host" )
    set(ISPC_FLAGS "--target=${ISPC_ARCH} --pic")
    set(ISPC_FLAGS_DB "--target=${ISPC_ARCH} --pic -DTOYBROT_USE_DOUBLES")
    separate_arguments(ISPC_FLAGS UNIX_COMMAND ${ISPC_FLAGS})
    separate_arguments(ISPC_FLAGS_DB UNIX_COMMAND ${ISPC_FLAGS_DB})
    add_custom_command( OUTPUT FracGen.ispc.o
                        COMMAND ${ISPC_COMPILER} ${ISPC_FLAGS}
                                            $<IF:$<CONFIG:Debug>,-O0,-O3>
                                            $<IF:$<CONFIG:Release>,,-g>
                                            ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.ispc
                                           -o ${CMAKE_CURRENT_BINARY_DIR}/FracGen.ispc.o
                        MAIN_DEPENDENCY FracGen.ispc  )

    add_custom_command( OUTPUT FracGen.ispc.db.o
                        COMMAND ${ISPC_COMPILER} ${ISPC_FLAGS_DB}
                                            $<IF:$<CONFIG:Debug>,-O0,-O3>
                                            $<IF:$<CONFIG:Release>,,-g>
                                            ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.ispc
                                           -o ${CMAKE_CURRENT_BINARY_DIR}/FracGen.ispc.db.o
                        MAIN_DEPENDENCY FracGen.ispc  )

    add_executable(${PROJECT_NAME} ${rmISPC_OBJS} "FracGen.cpp" )
    target_link_libraries(${PROJECT_NAME} PUBLIC ISPC_CXX_OBJS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _TOYBROT_ISPC_ARCH_=\"${ISPC_ARCH}\")

    add_executable(${PROJECT_NAME}-db ${rmISPC_OBJS_DB} "FracGen.cpp" )
    target_link_libraries(${PROJECT_NAME}-db PUBLIC ISPC_CXX_OBJS_DB)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE _TOYBROT_ISPC_ARCH_=\"${ISPC_ARCH}\" "TOYBROT_USE_DOUBLES")

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
    install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)

endif()
