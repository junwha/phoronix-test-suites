project(tbHC LANGUAGES CXX)

set(tbHC_SRCS    "main.cpp"
                 "FracGenWindow.cpp"
                 "FracGen.cpp" )

set(tbHC_HDRS   "FracGenWindow.hpp"
                "FracGen.hpp" )

set(tbHC_FILES ${tbHC_SRCS} ${tbHC_HDRS})

if(NOT TARGET hcc::hccrt)
    find_package(hcc)
endif()

set(tbHC_HCC_OPTS)
execute_process(
    COMMAND hcc-config --cxxflags --ldflags
    OUTPUT_VARIABLE HCSDL_HCC_OPTS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
set(tbHIP_NVCC_OPTS)

set_source_files_properties(${tbHC_SRCS} ${tbHC_HDRS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
set_source_files_properties(${tbHC_HDRS} PROPERTIES HEADER_FILE_ONLY 1)

if(  $ENV{HIP_PLATFORM} MATCHES "rocm"
  OR $ENV{HIP_PLATFORM} MATCHES "hip"
  OR $ENV{HIP_PLATFORM} MATCHES "hcc")

    hip_add_executable(${PROJECT_NAME} ${tbHC_FILES} HCC_OPTIONS ${tbHC_HCC_OPTS} "-fPIC")
    add_custom_target(${PROJECT_NAME}_HC_SRCS SOURCES ${tbHC_SRCS})

    target_link_libraries(${PROJECT_NAME} SDL2::SDL2 hcc::hccrt)

    if(NOT WIN32)
        target_link_libraries(${PROJECT_NAME} pthread)
    endif()

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
    endif()
