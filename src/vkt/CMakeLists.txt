# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(CUDA)
find_package(SDL2)
find_package(SWIG)
find_package(Virvo)
find_package(Visionaray)

vkt_use_package(CUDA)
vkt_use_package(SDL2)
vkt_use_package(SWIG)
vkt_use_package(Virvo)
if(VIRVO_FOUND)
    set(__VKT_LINK_LIBRARIES ${__VKT_LINK_LIBRARIES} ${VIRVO_FILEIO_LIBRARY})
endif()
#vkt_use_package(Visionaray COMPONENTS common) # TODO!!
vkt_use_package(Visionaray)

cmake_policy(SET CMP0078 NEW)

if(SWIG_FOUND)
    include(${SWIG_USE_FILE})
    find_package(Python3 3.6 COMPONENTS Interpreter Development REQUIRED)
    include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
endif()

#---------------------------------------------------------------------------------------------------
# Dependencies pulled in by visionaray
#

if(VISIONARAY_FOUND AND VISIONARAY_COMMON_LIBRARY)
    find_package(GLEW REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(OpenGL REQUIRED)

    vkt_use_package(GLEW)
    vkt_use_package(GLUT)
    vkt_use_package(OpenGL)

    # Private visionaray_common headers
    include_directories(${VISIONARAY_INCLUDE_DIR}/visionaray/private)

    # TODO
    set(3DCONNEXIONCLIENT_FOUND NOTFOUND)

    if (APPLE)
        include(CMakeFindFrameworks)

        CMAKE_FIND_FRAMEWORKS(3DconnexionClient)

        if (3DconnexionClient_FRAMEWORKS)
            set(__VKT_LINK_LIBRARIES ${__VKT_LINK_LIBRARIES} ${3DconnexionClient_FRAMEWORKS})
            set(__VKT_USED_PACKAGES ${__VKT_USED_PACKAGES} 3DCONNEXIONCLIENT)
            set(3DCONNEXIONCLIENT_FOUND FOUND)
        endif()
    endif()
endif()


#---------------------------------------------------------------------------------------------------
# ImGui submodule
#


if(VKT_ENABLE_IMGUI)
    # Make visible for config file creation

    # That's a submodule after all, so just set to FOUND
    set(IMGUI_FOUND FOUND)
    set(__VKT_USED_PACKAGES ${__VKT_USED_PACKAGES} IMGUI)


    # ImGui is compiled directly into the library
    set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/imgui)
    set(IMGUI_INCLUDE_DIR ${IMGUI_DIR})

    include_directories(${IMGUI_INCLUDE_DIR})

    set(IMGUI_HEADERS
        ${IMGUI_INCLUDE_DIR}/imconfig.h
        ${IMGUI_INCLUDE_DIR}/imgui.h
        ${IMGUI_INCLUDE_DIR}/imgui_internal.h
        ${IMGUI_INCLUDE_DIR}/imstb_rectpack.h
        ${IMGUI_INCLUDE_DIR}/imstb_textedit.h
        ${IMGUI_INCLUDE_DIR}/imstb_truetype.h
    )

    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
    )
endif()


#---------------------------------------------------------------------------------------------------
# Create config file
#

foreach(p ${__VKT_USED_PACKAGES})
    set(VKT_HAVE_${p} 1)
endforeach()

set(CONFIG_DIR ${__VKT_CONFIG_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CONFIG_DIR}/vkt/config.h)

include_directories(${__VKT_CONFIG_DIR})


#---------------------------------------------------------------------------------------------------
# Volkit library
#


include_directories(${PROJECT_SOURCE_DIR}/include/c)
include_directories(${PROJECT_SOURCE_DIR}/include/cpp)
include_directories(${PROJECT_SOURCE_DIR}/include/shared)

set(C_HEADER_DIR ${PROJECT_SOURCE_DIR}/include/c/vkt)
set(CPP_HEADER_DIR ${PROJECT_SOURCE_DIR}/include/cpp/vkt)

set(HEADERS
    ${C_HEADER_DIR}/StructuredVolume.h
)

set(SOURCES
    Aggregates.cpp
    Arithmetic.cpp
    Copy.cpp
    CudaTimer.cpp
    Decompose.cpp
    ExecutionPolicy.cpp
    Fill.cpp
    Flip.cpp
    InputStream.cpp
    LookupTable.cpp
    ManagedResource.cpp
    Memory.cpp
    RawFile.cpp
    Render.cpp
    Rotate.cpp
    Scan.cpp
    StructuredVolume.cpp
    TransfuncEditor.cpp
    Transform.cpp
    VirvoFile.cpp
    Voxel.cpp
)

vkt_cuda_compile(CUDA_SOURCES
    Arithmetic_cuda.cu
    Copy_cuda.cu
    Decompose_cuda.cu
    Fill_cuda.cu
)

vkt_add_library(volkit
    ${HEADERS}
    ${SOURCES}
    ${IMGUI_HEADERS}
    ${IMGUI_SOURCES}
    ${CUDA_SOURCES}
)


# Explicitly link with visionaray_common
if(VISIONARAY_COMMON_LIBRARY)
    target_link_libraries(volkit ${VISIONARAY_COMMON_LIBRARY})
endif()

if(SWIG_FOUND)
    set_property(SOURCE volkit.i PROPERTY CPLUSPLUS ON)
    swig_add_library(volkitpy LANGUAGE python SOURCES volkit.i)
    swig_link_libraries(volkitpy ${Python3_LIBRARIES})
    swig_link_libraries(volkitpy volkit)
endif()
