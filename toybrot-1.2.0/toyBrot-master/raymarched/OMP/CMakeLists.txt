project(rmOpenMP LANGUAGES CXX)

set(OMP_CXX_SRCS   "main.cpp"
                   "FracGen.hpp"
                   "${TB_COMMON_SRC_DIR}/Vec.hxx"
                   "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                   "${TB_COMMON_SRC_DIR}/defines.hpp" )

set(OMP_OMP_SRCS  "FracGen.cpp" )
set(OMP_OMP_PSRCS  "FracGen-parallel.cpp" )

set(OMP_OMP_OMPSRCS  "FracGen-parallel-omp.cpp" )


set(OMP_OBJS    "main.cpp.o"
                "FracGen.cpp.o")


set(OMP_OMP_OBJS    "FracGen-parallel-omp.cpp.o")
set(OMP_OMP_OBJS_DB "FracGen-parallel-omp-db.cpp.o")

add_custom_target(${PROJECT_NAME}-dist-omp SOURCES ${OMP_OMP_OMPSRCS})

set(OMP_FLAGS  "")
set(OMP_LINK_FLAGS "")

if(CMAKE_CXX_COMPILER MATCHES "aomp")

    set(OMP_FLAGS "${OMP_FLAGS} -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${AMD_GPU_ARCH} -fPIC")
    set(OMP_LINK_FLAGS "-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${AMD_GPU_ARCH}")

#    get_target_property(SDL_INCLUDES SDL2::SDL2 INTERFACE_INCLUDE_DIRECTORIES)
#    set(AOMP_FLAGS "-O3 -std=c++14 -target x86_64-pc-linux-gnu -w -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${AMD_GPU_ARCH} -I${SDL_INCLUDES}")
#    separate_arguments(AOMP_FLAGS UNIX_COMMAND ${AOMP_FLAGS})
#    add_custom_command( OUTPUT main.cpp.o
#                        COMMAND ${CMAKE_CXX_COMPILER}
#                        ARGS ${AOMP_FLAGS}   -c ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#                                             -o ${CMAKE_CURRENT_BINARY_DIR}/main.cpp.o
#                        MAIN_DEPENDENCY main.cpp  )
#    add_custom_command( OUTPUT FracGenWindow.cpp.o
#                        COMMAND ${CMAKE_CXX_COMPILER}
#                        ARGS ${AOMP_FLAGS}   -c ${CMAKE_CURRENT_SOURCE_DIR}/FracGenWindow.cpp
#                                             -o ${CMAKE_CURRENT_BINARY_DIR}/FracGenWindow.cpp.o
#                        MAIN_DEPENDENCY FracGenWindow.cpp  )
#    add_custom_command( OUTPUT FracGen.cpp.o
#                        COMMAND ${CMAKE_CXX_COMPILER}
#                        ARGS ${AOMP_FLAGS}   -c ${CMAKE_CURRENT_SOURCE_DIR}/FracGen.cpp
#                                             -o ${CMAKE_CURRENT_BINARY_DIR}/FracGen.cpp.o
#                        MAIN_DEPENDENCY FracGen.cpp  )

#    add_executable(${PROJECT_NAME}  ${OMP_OBJS} )

##    target_compile_options(${PROJECT_NAME} PRIVATE "-w")
##    set_source_files_properties(${OMP_OMP_SRCS} PROPERTIES
##                                                COMPILE_FLAGS "-O3 -target x86_64-pc-linux-gnu -w -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${AMD_GPU_ARCH}")
else()
    set(OMP_FLAGS "${OMP_FLAGS} -fopenmp -march=native")
endif()

separate_arguments(OMP_FLAGS UNIX_COMMAND ${OMP_FLAGS})


add_custom_command( OUTPUT ${OMP_OMP_OBJS}
                    COMMAND ${CMAKE_CXX_COMPILER} -c
                    $<IF:$<CONFIG:Debug>,-O0,-O3>
                    $<IF:$<CONFIG:Release>,,-g>
                    $<IF:$<CONFIG:Release>,,-DNDEBUG>
                    -std=c++17 ${OMP_FLAGS}
                    -I${TB_COMMON_SRC_DIR}
                    -fPIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/${OMP_OMP_OMPSRCS}
                                        -o ${CMAKE_CURRENT_BINARY_DIR}/${OMP_OMP_OBJS})


add_custom_command( OUTPUT ${OMP_OMP_OBJS_DB}
                    COMMAND ${CMAKE_CXX_COMPILER} -c
                    $<IF:$<CONFIG:Debug>,-O0,-O3>
                    $<IF:$<CONFIG:Release>,,-g>
                    $<IF:$<CONFIG:Release>,,-DNDEBUG>
                    -std=c++17 ${OMP_FLAGS}
                    -I${TB_COMMON_SRC_DIR}
                    -DTOYBROT_USE_DOUBLES
                    -fPIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/${OMP_OMP_OMPSRCS}
                                        -o ${CMAKE_CURRENT_BINARY_DIR}/${OMP_OMP_OBJS_DB})

add_executable(${PROJECT_NAME}    ${OMP_CXX_SRCS} ${OMP_OMP_SRCS} )
add_executable(${PROJECT_NAME}-db ${OMP_CXX_SRCS} ${OMP_OMP_SRCS} )

add_executable(${PROJECT_NAME}-dist  ${OMP_CXX_SRCS} ${OMP_OMP_PSRCS} ${OMP_OMP_OBJS})
add_executable(${PROJECT_NAME}-dist-db  ${OMP_CXX_SRCS} ${OMP_OMP_PSRCS} ${OMP_OMP_OBJS_DB})

target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_USE_DOUBLES" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-dist PRIVATE "TOYBROT_OPENMP_DIST" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)
target_compile_definitions(${PROJECT_NAME}-dist-db PRIVATE "TOYBROT_OPENMP_DIST" "TOYBROT_USE_DOUBLES" $<$<BOOL:TB_STDFS_FOUND>:"${TB_FS_DEFINE}">)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME}-db PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME}-dist PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME}-dist-db PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(${PROJECT_NAME}-dist PRIVATE "-fopenmp")
target_compile_options(${PROJECT_NAME}-dist-db PRIVATE "-fopenmp")


target_link_libraries(${PROJECT_NAME}-dist PRIVATE ${OMP_LINK_FLAGS})
target_link_libraries(${PROJECT_NAME}-dist-db PRIVATE ${OMP_LINK_FLAGS})
if(CMAKE_CXX_COMPILER MATCHES "clang")
    target_link_libraries(${PROJECT_NAME}    PRIVATE "-fopenmp=libomp")
    target_link_libraries(${PROJECT_NAME}-db PRIVATE "-fopenmp=libomp")
endif()


if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME}         PRIVATE ${TB_STD_LIBS})
    target_link_libraries(${PROJECT_NAME}-db      PRIVATE ${TB_STD_LIBS})
    target_link_libraries(${PROJECT_NAME}-dist    PRIVATE ${TB_STD_LIBS} m gcc gcc_s c)
    target_link_libraries(${PROJECT_NAME}-dist-db PRIVATE ${TB_STD_LIBS} m gcc gcc_s c)
endif()

if(TB_SDL_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")
    target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

    target_link_libraries(${PROJECT_NAME}-db PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
    target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

    target_link_libraries(${PROJECT_NAME}-dist PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME}-dist PRIVATE "TOYBROT_ENABLE_GUI")
    target_sources(${PROJECT_NAME}-dist PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")

    target_link_libraries(${PROJECT_NAME}-dist-db PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME}-dist-db PRIVATE "TOYBROT_ENABLE_GUI")
    target_sources(${PROJECT_NAME}-dist-db PRIVATE "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp")
endif()

if(TB_PNG_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME} PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

    target_link_libraries(${PROJECT_NAME}-db PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME}-db PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

    target_link_libraries(${PROJECT_NAME}-dist PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME}-dist PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME}-dist PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )

    target_link_libraries(${PROJECT_NAME}-dist-db PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME}-dist-db PRIVATE "TOYBROT_ENABLE_PNG")
    target_sources(${PROJECT_NAME}-dist-db PRIVATE "${TB_COMMON_SRC_DIR}/pngWriter.cpp" "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
endif()

target_include_directories(${PROJECT_NAME}  PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${TB_COMMON_SRC_DIR})

target_include_directories(${PROJECT_NAME}-db PRIVATE
                                              ${CMAKE_CURRENT_SOURCE_DIR}
                                              ${TB_COMMON_SRC_DIR})

target_include_directories(${PROJECT_NAME}-dist PRIVATE
                                                ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${TB_COMMON_SRC_DIR})

target_include_directories(${PROJECT_NAME}-dist-db  PRIVATE
                                                    ${CMAKE_CURRENT_SOURCE_DIR}
                                                    ${TB_COMMON_SRC_DIR})

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME}    PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME}-db PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME}-dist PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME}-dist-db PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-db RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-dist RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}-dist-db RUNTIME DESTINATION bin)
