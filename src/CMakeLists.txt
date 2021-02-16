# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(NOT BUILD_SHARED_LIBS)
    add_definitions(-DVKT_STATIC)
endif()

set(VSNRAY_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/visionaray)
set(VSNRAY_ENABLE_VIEWER OFF)
add_subdirectory(${VSNRAY_DIR})

add_subdirectory(cli)
add_subdirectory(examples)
add_subdirectory(vkt)
