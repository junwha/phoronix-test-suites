project(tbVK LANGUAGES CXX)

set(tbVK_SRCS    "main.cpp"
                 "FracGenWindow.cpp"
                 "FracGen.cpp" )

set(tbVK_HDRS    "FracGenWindow.hpp"
                 "FracGen.hpp" )

set(tbVK_VKSRCS  "FracGen.comp")

set(tbVK_SPIRV   "${CMAKE_CURRENT_BINARY_DIR}/FracGen.spv")

set_source_files_properties(${tbVKSPIRV} PROPERTIES GENERATED TRUE)
set_source_files_properties(${tbVK_SRCS} PROPERTIES LANGUAGE CXX)


add_custom_command(OUTPUT ${tbVK_SPIRV}
                   COMMAND ${SLANGCOMPILER} -V -o ${tbVK_SPIRV} ${tbVK_VKSRCS}
                   COMMENT "Compiling shader to SPIR-V"
                   DEPENDS ${tbVK_VKSRCS}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${tbVK_SRCS} ${tbVK_HDRS} ${tbVK_SPIRV})
add_custom_target(${PROJECT_NAME}_VK_SRCS SOURCES ${tbVK_VKSRCS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_VK_SRCS)

target_link_libraries(${PROJECT_NAME} SDL2::SDL2 Vulkan::Vulkan)

#For ease of debugging as it needs to be deployed alongside the kernel source
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} pthread)
    if(CMAKE_CXX_COMPILER MATCHES clang)
        target_compile_options(${PROJECT_NAME} PRIVATE "-fcxx-exceptions")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE "-fexceptions")
    endif()
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES ${tbVK_VKSRCS} ${tbVK_SPIRV} DESTINATION bin)
