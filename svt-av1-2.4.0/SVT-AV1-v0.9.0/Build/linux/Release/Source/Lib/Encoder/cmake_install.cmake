# Install script for directory: /var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder

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
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so.0.9.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Bin/Release/libSvtAv1Enc.so.0.9.0"
    "/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Bin/Release/libSvtAv1Enc.so.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so.0.9.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/Codec:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/C_DEFAULT:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/Codec:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/Globals:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_SSE2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/C_DEFAULT:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_SSSE3:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_SSE4_1:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_AVX2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_AVX512:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_SSE2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_SSSE3:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_SSE4_1:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_AVX2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_AVX512:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Bin/Release/libSvtAv1Enc.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so"
         OLD_RPATH "/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/Codec:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/C_DEFAULT:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/Codec:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/Globals:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_SSE2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/C_DEFAULT:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_SSSE3:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_SSE4_1:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_AVX2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Common/ASM_AVX512:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_SSE2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_SSSE3:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_SSE4_1:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_AVX2:/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Source/Lib/Encoder/ASM_AVX512:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSvtAv1Enc.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/SvtAv1Enc.pc")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/C_DEFAULT/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/Codec/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/Globals/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/ASM_SSE2/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/ASM_SSSE3/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/ASM_SSE4_1/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/ASM_AVX2/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/svt-av1-2.4.0/SVT-AV1-v0.9.0/Build/linux/Release/Source/Lib/Encoder/ASM_AVX512/cmake_install.cmake")

endif()

