# This file is distributed under the MIT license.
# See the LICENSE file for details.

macro(vkt_link_libraries)
    set(__VKT_LINK_LIBRARIES ${__VKT_LINK_LIBRARIES} ${ARGN})
endmacro()

function(vkt_cuda_compile outfiles)
    if(NOT CUDA_FOUND OR NOT VKT_ENABLE_CUDA)
        return()
    endif()

    foreach(f ${ARGN})
        get_filename_component(suffix ${f} EXT)

        if(NOT ${suffix} STREQUAL ".cu")
            message(FATAL_ERROR "Cannot cuda_compile file with extension ${suffix}")
            return()
        endif()

        set(nvcc_flags_old__ ${CUDA_NVCC_FLAGS})
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-extended-lambda")

        if(BUILD_SHARED_LIBS)
            cuda_compile(cuda_compile_obj ${f} SHARED)
        else()
            cuda_compile(cuda_compile_obj ${f})
        endif()
        set(out ${out} ${f} ${cuda_compile_obj})

        set (CUDA_NVC_FLAGS ${nvcc_flags_old__})
    endforeach()

    set(${outfiles} ${out} PARENT_SCOPE)
endfunction()

function(vkt_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} ${__VKT_LINK_LIBRARIES})

    if(VKT_MACOSX_BUNDLE)
        set_target_properties(${name} PROPERTIES MACOSX_BUNDLE TRUE)
    endif()
endfunction()

function(vkt_add_cuda_executable name)
    if(NOT CUDA_FOUND OR NOT VKT_ENABLE_CUDA)
        return()
    endif()

    set(nvcc_flags_old__ ${CUDA_NVCC_FLAGS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-extended-lambda")

    cuda_add_executable(${name} ${ARGN})

    set (CUDA_NVCC_FLAGS ${nvcc_flags_old__})

    target_link_libraries(${name} ${__VKT_LINK_LIBRARIES})

    if(VKT_MACOSX_BUNDLE)
        set_target_properties(${name} PROPERTIES MACOSX_BUNDLE TRUE)
    endif()
endfunction()

function(vkt_add_library name)
    add_library(${name} ${ARGN})
    target_link_libraries(${name} ${__VKT_LINK_LIBRARIES})
endfunction()

function(vkt_add_cuda_library name)
    if(NOT CUDA_FOUND OR NOT VKT_ENABLE_CUDA)
        return()
    endif()

    set(nvcc_flags_old__ ${CUDA_NVCC_FLAGS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-extended-lambda")

    cuda_add_library(${name} ${ARGN})

    if(NOT BUILD_SHARED_LIBS)
        set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set_property(TARGET ${name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()

    set (CUDA_NVCC_FLAGS ${nvcc_flags_old__})

    target_link_libraries(${name} ${__VKT_LINK_LIBRARIES})
endfunction()
