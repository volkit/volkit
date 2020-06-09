# This file is distributed under the MIT license.
# See the LICENSE file for details.

set(VSNRAY_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/visionaray)
set(VSNRAY_ENABLE_VIEWER OFF)
add_subdirectory(${VSNRAY_DIR})

add_subdirectory(examples)
add_subdirectory(vkt)
