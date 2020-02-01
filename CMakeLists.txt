# This file is distributed under the MIT license.
# See the LICENSE file for details.

project(volkit)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake"
    "${PROJECT_SOURCE_DIR}/cmake/modules"
)

include(VktAddTarget)
include(VktUsePackage)

#---------------------------------------------------------------------------------------------------
# Directories containing config headers
#

# Config file location (#include <vkt/config.h>)
set(__VKT_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/config")


#---------------------------------------------------------------------------------------------------
# VKT Options
#

option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" OFF)
option(VKT_ENABLE_WARNINGS "Enable all warnings" ON)
option(VKT_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" ON)
option(VKT_ENABLE_CUDA "Use CUDA, if available" ON)
option(VKT_MACOSX_BUNDLE "Build executables as application bundles on macOS" ON)

#---------------------------------------------------------------------------------------------------
# All VKT_* options have to be specified before this include
#

include(VktHandleOptions)

add_subdirectory(src)
