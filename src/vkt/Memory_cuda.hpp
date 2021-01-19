// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <cstddef>

#if VKT_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <vkt/Memory.hpp>

#include "macros.hpp"

namespace vkt
{
    inline void Allocate_cuda(void** ptr, std::size_t size)
    {
#if VKT_HAVE_CUDA
        VKT_CUDA_SAFE_CALL__(cudaMalloc(ptr, size));
#endif
    }

    inline void Free_cuda(void* ptr)
    {
#if VKT_HAVE_CUDA
        VKT_CUDA_SAFE_CALL__(cudaFree(ptr));
#endif
    }

    void MemsetRange_cuda(
            void* dst,
            void const* src,
            std::size_t dstSize,
            std::size_t srcSize
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

} // vkt
