// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include <vkt/Memory.hpp>

#include "macros.hpp"

namespace vkt
{
    inline void Allocate_cuda(void** ptr, std::size_t size)
    {
        VKT_CUDA_SAFE_CALL__(cudaMalloc(ptr, size));
    }

    inline void Free_cuda(void* ptr)
    {
        VKT_CUDA_SAFE_CALL__(cudaFree(ptr));
    }

    void MemsetRange_cuda(
            void* dst,
            void const* src,
            std::size_t dstSize,
            std::size_t srcSize
            );

} // vkt
