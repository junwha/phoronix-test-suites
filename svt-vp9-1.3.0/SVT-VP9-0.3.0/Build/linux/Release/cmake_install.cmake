# Install script for directory: /var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/svt-vp9/")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/include/svt-vp9" TYPE DIRECTORY FILES "/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Source/API/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/VPX/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/Codec/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/C_DEFAULT/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/ASM_SSE2/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/ASM_SSSE3/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/ASM_SSE4_1/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/Lib/ASM_AVX2/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/Source/App/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/var/lib/phoronix-test-suite/installed-tests/pts/svt-vp9-1.3.0/SVT-VP9-0.3.0/Build/linux/Release/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
