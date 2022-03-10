# Install script for directory: /var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/HelloWorld/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/OpenGL/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/AllBulletDemos/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ConvexDecompositionDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/CcdPhysicsDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/BulletXmlImportDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ConstraintDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/SliderConstraintDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/GenericJointDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/Raytracer/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/RagdollDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ForkLiftDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/BasicDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/RollingFrictionDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/RaytestDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/VoronoiFractureDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/GyroscopicDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/FractureDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/Box2dDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/BspDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/MovingConcaveDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/VehicleDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/UserCollisionAlgorithm/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/CharacterDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/SoftDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/CollisionInterfaceDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ConcaveConvexcastDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/SimplexDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/DynamicControlDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ConvexHullDistance/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/DoublePrecisionDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ConcaveDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/CollisionDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ContinuousConvexCollision/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/ConcaveRaycastDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/GjkConvexCastDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/MultiMaterialDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/SerializeDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/InternalEdgeDemo/cmake_install.cmake")
  include("/var/lib/phoronix-test-suite/installed-tests/pts/bullet-1.2.2/bullet-2.81-rev2613/Demos/Benchmarks/cmake_install.cmake")

endif()

