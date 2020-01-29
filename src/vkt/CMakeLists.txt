find_package(CUDA)
find_package(SDL2)
find_package(Visionaray)

vkt_use_package(CUDA)
vkt_use_package(SDL2)
#vkt_use_package(Visionaray COMPONENTS common) # TODO!!
vkt_use_package(Visionaray)

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

set(C_HEADER_DIR ${PROJECT_SOURCE_DIR}/include/c/vkt)
set(CPP_HEADER_DIR ${PROJECT_SOURCE_DIR}/include/cpp/vkt)

set(HEADERS
    ${C_HEADER_DIR}/StructuredVolume.h
)

set(SOURCES
    Copy.cpp
    CudaTimer.cpp
    Decompose.cpp
    ExecutionPolicy.cpp
    Fill.cpp
    ManagedBuffer.cpp
    Memory.cpp
    Render.cpp
    StructuredVolume.cpp
    Transform.cpp
)

vkt_cuda_compile(CUDA_SOURCES
    Decompose_cuda.cu
    Fill_cuda.cu
)

vkt_add_library(volkit ${HEADERS} ${SOURCES} ${CUDA_SOURCES})


# Explicitly link with visionaray_common which includes
if(VISIONARAY_COMMON_LIBRARY)
    target_link_libraries(volkit ${VISIONARAY_COMMON_LIBRARY})
    include_directories(${VISIONARAY_INCLUDE_DIR}/visionaray/private)
endif()
