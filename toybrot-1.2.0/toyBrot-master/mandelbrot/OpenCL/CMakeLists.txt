project(tbOCL LANGUAGES CXX)

set(tbOCL_SRCS    "main.cpp"
                 "FracGenWindow.cpp"
                 "FracGen.cpp" )

set(tbOCL_HDRS   "FracGenWindow.hpp"
                 "FracGen.hpp" )

set(tbOCL_CLSRCS "FracGen.cl")


add_executable(${PROJECT_NAME} ${tbOCL_SRCS} ${tbOCL_HDRS} )
add_custom_target(${PROJECT_NAME}_CL_SRCS SOURCES ${tbOCL_CLSRCS})

target_link_libraries(${PROJECT_NAME} SDL2::SDL2 OpenCL::OpenCL)

#For ease of debugging as it needs to be deployed alongside the kernel source
#set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} pthread)
    if(CMAKE_CXX_COMPILER MATCHES clang)
        target_compile_options(${PROJECT_NAME} PRIVATE "-fcxx-exceptions")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE "-fexceptions")
    endif()
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES ${tbOCL_CLSRCS} DESTINATION bin)
