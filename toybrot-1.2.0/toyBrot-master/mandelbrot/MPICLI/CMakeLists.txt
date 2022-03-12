project(tbMPICLI LANGUAGES CXX)

add_executable(${PROJECT_NAME}  "main.cpp"
                                "pngWriter.hpp"
                                "pngWriter.cpp")


find_package(PNG)

target_link_libraries(${PROJECT_NAME} MPI::MPI_CXX)

if(TARGET PNG::PNG)
    target_link_libraries(${PROJECT_NAME} PNG::PNG)

    if(NOT WIN32)
        target_link_libraries(${PROJECT_NAME} pthread)
    endif()

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

else()
    message("Could not find PNG. Disabling Toybrot ${PROJECT_NAME}")
endif()


