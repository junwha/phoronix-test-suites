project(rmSTD_TASKS LANGUAGES CXX)

add_executable(${PROJECT_NAME}  "main.cpp"
                                "FracGen.hpp"
                                "FracGen.cpp"
                                "${TB_COMMON_SRC_DIR}/Vec.hxx"
                                "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                                "${TB_COMMON_SRC_DIR}/defines.hpp" )

add_executable(${PROJECT_NAME}-db "main.cpp"
                                  "FracGen.hpp"
                                  "FracGen.cpp"
                                  "${TB_COMMON_SRC_DIR}/Vec.hxx"
                                  "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                                  "${TB_COMMON_SRC_DIR}/defines.hpp" )

target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_USE_DOUBLES" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TB_STD_LIBS})
target_link_libraries(${PROJECT_NAME}-db PRIVATE ${TB_STD_LIBS})

#target_compile_options(${PROJECT_NAME} PRIVATE -fitodcalls -mllvm -vectorize-memory-aggressively -mllvm -enable-strided-vectorization)

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

target_include_directories(${PROJECT_NAME}  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR}        )

target_include_directories(${PROJECT_NAME}-db PRIVATE
                                              ${CMAKE_CURRENT_SOURCE_DIR}
                                              ${TB_COMMON_SRC_DIR}        )

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)
