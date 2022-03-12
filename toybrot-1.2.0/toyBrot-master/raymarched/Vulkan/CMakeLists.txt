project(rmVulkan LANGUAGES CXX)

set(rmVK_SRCS   "main.cpp"
                "FracGen.cpp"        )

set(rmVK_HDRS   "FracGen.hpp"
                "${TB_COMMON_SRC_DIR}/Vec.hxx"
                "${TB_COMMON_SRC_DIR}/dataTypes.hxx"
                "${TB_COMMON_SRC_DIR}/defines.hpp" )

set(rmVK_GLSL "${CMAKE_CURRENT_SOURCE_DIR}/../OpenGL/FracGen.comp.glsl" )

set(rmVK_HLSL "FracGen.comp.hlsl" )

set(rmVK_CL   "${CMAKE_CURRENT_SOURCE_DIR}/../OpenCL/FracGen.cl" )

set(rmVK_DBG  "vk_layer_settings.txt")

#set(rmVK_CL   "${CMAKE_CURRENT_BINARY_DIR}/../OpenCL/FracGen.cl")

find_program(CLSPV_COMPILER "clspv" DOC "OpenCL to Spir-V compiler")

set(rmVK_SPIRV_GLSL   "${CMAKE_CURRENT_BINARY_DIR}/FracGen.glsl.spv")
set(rmVK_SPIRV_HLSL   "${CMAKE_CURRENT_BINARY_DIR}/FracGen.hlsl.spv")
set(rmVK_SPIRV_CL     "${CMAKE_CURRENT_BINARY_DIR}/FracGen.cl.spv"  )

set(rmVK_SPIRV_GLSL_DB   "${CMAKE_CURRENT_BINARY_DIR}/FracGen.glsl.db.spv")
set(rmVK_SPIRV_HLSL_DB   "${CMAKE_CURRENT_BINARY_DIR}/FracGen.hlsl.db.spv")
set(rmVK_SPIRV_CL_DB     "${CMAKE_CURRENT_BINARY_DIR}/FracGen.cl.db.spv"  )

set(rmVK_SPIRV    ${rmVK_SPIRV_GLSL}    ${rmVK_SPIRV_HLSL})
set(rmVK_SPIRV_DB ${rmVK_SPIRV_GLSL_DB} ${rmVK_SPIRV_HLSL_DB})

set_source_files_properties(${rmVK_SPIRV_GLSL} PROPERTIES GENERATED TRUE)
set_source_files_properties(${rmVK_SPIRV_HLSL} PROPERTIES GENERATED TRUE)

set_source_files_properties(${rmVK_SPIRV_GLSL_DB} PROPERTIES GENERATED TRUE)
set_source_files_properties(${rmVK_SPIRV_HLSL_DB} PROPERTIES GENERATED TRUE)

if(CLSPV_COMPILER)
    message("CLSPV found")
    set(rmVK_SPIRV ${rmVK_SPIRV} ${rmVK_SPIRV_CL})
    set_source_files_properties(${rmVK_SPIRV_CL} PROPERTIES GENERATED TRUE)

    set(rmVK_SPIRV_DB ${rmVK_SPIRV_DB} ${rmVK_SPIRV_CL_DB})
    set_source_files_properties(${rmVK_SPIRV_CL_DB} PROPERTIES GENERATED TRUE)

endif()

if(TB_SDL_FOUND)
    set(rmVK_SRCS ${rmVK_SRCS} "${TB_COMMON_SRC_DIR}/FracGenWindow.cpp" )
    set(rmVK_HDRS ${rmVK_HDRS} "${TB_COMMON_SRC_DIR}/FracGenWindow.hpp" )
endif()

if(TB_PNG_FOUND)
    set(rmVK_SRCS ${rmVK_SRCS} "${TB_COMMON_SRC_DIR}/pngWriter.cpp" )
    set(rmVK_HDRS ${rmVK_HDRS} "${TB_COMMON_SRC_DIR}/pngWriter.hpp" )
endif()


set_source_files_properties(${rmVK_SRCS} PROPERTIES LANGUAGE CXX)

#add_custom_target(SPVClean ALL COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV})

#GLSL NEEDS the entry point to be called "main" but OpenCL CANNOT have that be the case
add_custom_command(OUTPUT ${rmVK_SPIRV_GLSL}
                   COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV_GLSL}
                   COMMAND ${SLANGCOMPILER} -e traceRegion $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN --source-entrypoint main -V -o ${rmVK_SPIRV_GLSL} ${rmVK_GLSL}
                   COMMENT "Compiling GLSL shader to SPIR-V"
                   DEPENDS ${rmVK_GLSL}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(OUTPUT ${rmVK_SPIRV_GLSL_DB}
                   COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV_GLSL_DB}
                   COMMAND ${SLANGCOMPILER} -e traceRegion $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN -DTOYBROT_USE_DOUBLES --source-entrypoint main -V -o ${rmVK_SPIRV_GLSL_DB} ${rmVK_GLSL}
                   COMMENT "Compiling GLSL shader to SPIR-V (doubles)"
                   DEPENDS ${rmVK_GLSL}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(OUTPUT ${rmVK_SPIRV_HLSL}
                   COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV_HLSL}
                   COMMAND ${SLANGCOMPILER} -e traceRegion $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN -V -o ${rmVK_SPIRV_HLSL} ${rmVK_HLSL}
                   COMMENT "Compiling HLSL shader to SPIR-V"
                   DEPENDS ${rmVK_HLSL}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(OUTPUT ${rmVK_SPIRV_HLSL_DB}
                  COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV_HLSL_DB}
                  COMMAND ${SLANGCOMPILER} -e traceRegion $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN -DTOYBROT_USE_DOUBLES -V -o ${rmVK_SPIRV_HLSL_DB} ${rmVK_HLSL}
                  COMMENT "Compiling HLSL shader to SPIR-V"
                  DEPENDS ${rmVK_HLSL}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(CLSPV_COMPILER)
    # Sadly the generator expressions weren't behaving because of the spaces
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CLSPV_OPT "-O0" )
    else()
        set(CLSPV_OPT "-O3 --cl-unsafe-math-optimizations --cl-finite-math-only" )
    endif()
    set(CLSPV_ARGS "-constant-args-ubo ${CLSPV_OPT} -descriptormap=FracGen.cl.csv $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN -o ${rmVK_SPIRV_CL} ${rmVK_CL}")
    # descriptor map is just useful for debugging
    add_custom_command(OUTPUT ${rmVK_SPIRV_CL}
                       COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV_CL}
                       COMMAND ${CLSPV_COMPILER} -constant-args-ubo ${CLSPV_OPT} -descriptormap=FracGen.cl.csv $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN -o ${rmVK_SPIRV_CL} ${rmVK_CL}
                       COMMENT "Compiling OpenCL kernel code to SPIR-V"
                       DEPENDS ${rmVK_CL}
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

   add_custom_command(OUTPUT ${rmVK_SPIRV_CL_DB}
                      COMMAND ${CMAKE_COMMAND} -E remove -f ${rmVK_SPIRV_CL_DB}
                      COMMAND ${CLSPV_COMPILER} -constant-args-ubo ${CLSPV_OPT} -descriptormap=FracGen.cl.db.csv $<$<CONFIG:Debug>:-DTOYBROT_DEBUG> -DTOYBROT_VULKAN -DTOYBROT_USE_DOUBLES -o ${rmVK_SPIRV_CL_DB} ${rmVK_CL}
                      COMMENT "Compiling OpenCL kernel code to SPIR-V"
                      DEPENDS ${rmVK_CL}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_executable(${PROJECT_NAME} ${rmVK_SRCS} ${rmVK_HDRS} ${rmVK_SPIRV})
add_executable(${PROJECT_NAME}-db ${rmVK_SRCS} ${rmVK_HDRS} ${rmVK_SPIRV_DB})
target_compile_definitions(${PROJECT_NAME} PRIVATE "TB_VEC_GL_ALIGN")
target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TB_VEC_GL_ALIGN" "TOYBROT_USE_DOUBLES")

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TB_COMMON_SRC_DIR})
target_include_directories(${PROJECT_NAME}-db PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TB_COMMON_SRC_DIR})

if(CLSPV_COMPILER)
    add_custom_target(${PROJECT_NAME}_VK_SRCS SOURCES ${rmVK_GLSL} ${rmVK_HLSL} ${rmVK_DBG} ${rmVK_CL})
else()
    add_custom_target(${PROJECT_NAME}_VK_SRCS SOURCES ${rmVK_GLSL} ${rmVK_HLSL} ${rmVK_DBG})
endif()

#Make sure we're running the clean and recompiling of the shader
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_VK_SRCS)

target_link_libraries(${PROJECT_NAME}    PRIVATE  ${TB_STD_LIBS} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME}-db PRIVATE  ${TB_STD_LIBS} Vulkan::Vulkan)

if(TB_SDL_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_GUI")

    target_link_libraries(${PROJECT_NAME}-db PRIVATE SDL2::SDL2)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_GUI")
endif()

if(TB_PNG_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "TOYBROT_ENABLE_PNG")

    target_link_libraries(${PROJECT_NAME}-db PRIVATE PNG::PNG)
    target_compile_definitions(${PROJECT_NAME}-db PRIVATE "TOYBROT_ENABLE_PNG")
endif()

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
    if(CMAKE_CXX_COMPILER MATCHES clang)
        target_compile_options(${PROJECT_NAME} PRIVATE "-fcxx-exceptions")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE "-fexceptions")
    endif()
endif()

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}-db RUNTIME DESTINATION bin)
install(FILES ${rmVK_GLSL} ${rmVK_HLSL} ${rmVK_SPIRV} ${rmVK_SPIRV_DB} DESTINATION bin)

if(CMAKE_BUILD_TYPE MATCHES "DEBUG")
    install(FILES ${rmVK_DBG} DESTINATION bin)
endif()

if(NOT TARGET rmOpenCL)
    #If OpenCL exists IT will install this file
    install(FILES ${rmVK_CL} DESTINATION bin)
endif()
