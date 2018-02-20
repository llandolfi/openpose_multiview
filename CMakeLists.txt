
cmake_minimum_required( VERSION 2.8 )
project( StereoPose )

add_definitions(--std=c++11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules")

find_package(Freenect2 REQUIRED)

include_directories(/usr/include/jsoncpp)

find_package(OpenMP)
if (OPENMP_FOUND) 

    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DHAVE_OMP)
endif()

#######find libusb
if (LIBUSB_1_LIBRARIES AND LIBUSB_1_INCLUDE_DIRS)
  # in cache already
  set(LIBUSB_FOUND TRUE)
else (LIBUSB_1_LIBRARIES AND LIBUSB_1_INCLUDE_DIRS)
  find_path(LIBUSB_1_INCLUDE_DIR
    NAMES
  libusb.h
    PATHS
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include
  PATH_SUFFIXES
    libusb-1.0
  )

  find_library(LIBUSB_1_LIBRARY
    NAMES
      usb-1.0 usb
    PATHS
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
  )

  set(LIBUSB_1_INCLUDE_DIRS
    ${LIBUSB_1_INCLUDE_DIR}
  )
  set(LIBUSB_1_LIBRARIES
    ${LIBUSB_1_LIBRARY}
)

  if (LIBUSB_1_INCLUDE_DIRS AND LIBUSB_1_LIBRARIES)
     set(LIBUSB_1_FOUND TRUE)
  endif (LIBUSB_1_INCLUDE_DIRS AND LIBUSB_1_LIBRARIES)

  if (LIBUSB_1_FOUND)
    if (NOT libusb_1_FIND_QUIETLY)
      message(STATUS "Found libusb-1.0:")
    message(STATUS " - Includes: ${LIBUSB_1_INCLUDE_DIRS}")
    message(STATUS " - Libraries: ${LIBUSB_1_LIBRARIES}")
    endif (NOT libusb_1_FIND_QUIETLY)
  else (LIBUSB_1_FOUND)
    if (libusb_1_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find libusb")
    endif (libusb_1_FIND_REQUIRED)
  endif (LIBUSB_1_FOUND)

  # show the LIBUSB_1_INCLUDE_DIRS and LIBUSB_1_LIBRARIES variables only in the advanced view
  mark_as_advanced(LIBUSB_1_INCLUDE_DIRS LIBUSB_1_LIBRARIES)

endif (LIBUSB_1_LIBRARIES AND LIBUSB_1_INCLUDE_DIRS)
include_directories(${LIBUSB_1_INCLUDE_DIRS} ${LIBUSB_1_INCLUDE_DIRS}/libusb-1.0)

find_library(FREENECT_LIBRARY  NAMES freenect)
include_directories(${CMAKE_PREFIX_PATH}/include/libfreenect )

include_directories(${LIBUSB_1_INCLUDE_DIRS} ${LIBUSB_1_INCLUDE_DIRS}/libusb-1.0)
include_directories(${FREENECT_GRABBER_INCLUDE_DIR} )


find_package( OpenCV REQUIRED )

include_directories(${FREENECT2_INCLUDE_DIRS})

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package( Boost REQUIRED COMPONENTS program_options thread system)

find_package( Eigen3 REQUIRED )
find_package( libuvc REQUIRED )
find_package( CUDA REQUIRED)
if (CUDA_FOUND)
    INCLUDE_DIRECTORIES(
        ${CUDA_INCLUDE_DIRS}
    )
    include_directories( include )
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( EIGEN3_INCLUDE_DIR )


    add_definitions(-DUSE_CAFFE)
    add_definitions(-DUSE_CUDNN)

    file(GLOB STEREO_SOURCES src/*.cpp)

    ADD_LIBRARY( stereoutils  ${STEREO_SOURCES} )

    TARGET_LINK_LIBRARIES( stereoutils
        ${CUDA_LIBS}
        ${libuvc_LIBRARIES} 
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        caffe
        glog
        gflags
        openpose
        jsoncpp
        opencv_cudastereo
    )

    #install library 
    install( TARGETS stereoutils DESTINATION lib/stereoutils)

    #install library headers
    file( GLOB HEADERS inlude/*.h include/*.hpp)
    install( FILES ${HEADERS} DESTINATION include/stereoutils )


    ADD_EXECUTABLE( stereopose_rt examples/videoprocess.cpp )
    ADD_EXECUTABLE( stereopose_of examples/fileprocess.cpp )
    ADD_EXECUTABLE( k1server examples/kinect1server.cpp )
    ADD_EXECUTABLE( k2server examples/kinect2server.cpp )

    TARGET_LINK_LIBRARIES(stereopose_rt
        stereoutils
        rt
    )

    TARGET_LINK_LIBRARIES(stereopose_of
        stereoutils
    )

    TARGET_LINK_LIBRARIES(k1server
      ${FREENECT_LIBRARY}   
      ${OpenCV_LIBS}
      ${Boost_LIBRARIES}
      rt
    )

    TARGET_LINK_LIBRARIES(k2server
      ${FREENECT2_LIBRARY}
      ${OpenCV_LIBS}
      ${Boost_LIBRARIES}
      rt
    )

endif ()
