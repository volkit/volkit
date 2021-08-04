# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(Boost COMPONENTS filesystem system REQUIRED)
find_package(CUDA)
find_package(HDF5 COMPONENTS CXX)
find_package(Nifti)
find_package(SDL2)
find_package(SWIG)
find_package(Threads REQUIRED)
find_package(Virvo)

vkt_use_package(Boost)
vkt_use_package(CUDA)
vkt_use_package(HDF5)
if(HDF5_FOUND)
    include_directories(${HDF5_CXX_INCLUDE_DIRS})
endif()
vkt_use_package(Nifti)
vkt_use_package(SDL2)
vkt_use_package(Threads)
vkt_use_package(SWIG)
vkt_use_package(Virvo)
if(VIRVO_FOUND)
    set(__VKT_LINK_LIBRARIES ${__VKT_LINK_LIBRARIES} ${VIRVO_FILEIO_LIBRARY})
endif()

if(POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
endif()

if(SWIG_FOUND)
    include(${SWIG_USE_FILE})
    if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
        find_package(PythonInterp 3.6 REQUIRED)
        find_package(PythonLibs 3.6 REQUIRED)
        include_directories(${PYTHON_INCLUDE_DIRS})
    else()
        find_package(Python3 3.6 COMPONENTS Interpreter Development REQUIRED)
        include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
    endif()
endif()

#---------------------------------------------------------------------------------------------------
# Visionaray submodule and dependencies
#

find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

vkt_use_package(GLEW)
vkt_use_package(GLUT)
vkt_use_package(OpenGL)

include_directories(${VSNRAY_DIR}/include)
# For private visionaray_common headers
include_directories(${VSNRAY_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../3rdparty/visionaray/config)

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


#---------------------------------------------------------------------------------------------------
# ImGui submodule
#


if(VKT_ENABLE_IMGUI)
    # Make visible for config file creation

    # That's a submodule after all, so just set to FOUND
    set(IMGUI_FOUND FOUND)
    set(__VKT_USED_PACKAGES ${__VKT_USED_PACKAGES} IMGUI)


    # ImGui is compiled directly into the library
    set(IMGUI_DIR ${VSNRAY_DIR}/src/3rdparty/imgui)
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
    Crop.cpp
    Decompose.cpp
    ExecutionPolicy.cpp
    Fill.cpp
    FLASHFile.cpp
    Flip.cpp
    HierarchicalVolume.cpp
    Histogram.cpp
    InputStream.cpp
    Logging.cpp
    LookupTable.cpp
    ManagedResource.cpp
    Memory.cpp
    NiftiFile.cpp
    OutputStream.cpp
    RawFile.cpp
    Resample.cpp
    Rotate.cpp
    Scale.cpp
    Scan.cpp
    StructuredVolume.cpp
    Timer.cpp
    TransfuncEditor.cpp
    Transform.cpp
    VirvoFile.cpp
    VolumeFile.cpp
    Voxel.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    vkt_cuda_compile(CUDA_SOURCES
        Arithmetic_cuda.cu
        Copy_cuda.cu
        Decompose_cuda.cu
        Fill_cuda.cu
        Flip_cuda.cu
        Histogram_cuda.cu
        Memory_cuda.cu
        Render.cu
        Resample_cuda.cu
    )

    set(SOURCES ${SOURCES}
        CudaTimer.cpp
    )

    vkt_add_cuda_library(volkit
        ${HEADERS}
        ${SOURCES}
        ${IMGUI_HEADERS}
        ${IMGUI_SOURCES}
        ${CUDA_SOURCES}
    )
else()
    set(SOURCES ${SOURCES}
        Render.cpp
    )

    vkt_add_library(volkit
        ${HEADERS}
        ${SOURCES}
        ${IMGUI_HEADERS}
        ${IMGUI_SOURCES}
    )
endif()



# Explicitly link with visionaray
target_link_libraries(volkit visionaray)
target_link_libraries(volkit visionaray_common)

if(SWIG_FOUND)
    set_property(SOURCE volkit.i PROPERTY CPLUSPLUS ON)
    #if(BUILD_SHARED_LIBS)
        swig_add_library(volkitpy LANGUAGE python SOURCES volkit.i)
    #else()
    #    swig_add_library(volkitpy TYPE STATIC LANGUAGE python SOURCES volkit.i)
    #endif()
    swig_link_libraries(volkitpy ${Python3_LIBRARIES})
    swig_link_libraries(volkitpy volkit)
endif()
