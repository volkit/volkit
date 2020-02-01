// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <cstddef>

#if VKT_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "macros.hpp"
#include "Memory.hpp"

namespace vkt
{
    void Allocate_cuda(void** ptr, std::size_t size)
    {
#if VKT_HAVE_CUDA
        VKT_CUDA_SAFE_CALL__(cudaMalloc(ptr, size));
#endif
    }

    void Free_cuda(void* ptr)
    {
#if VKT_HAVE_CUDA
        VKT_CUDA_SAFE_CALL__(cudaFree(ptr));
#endif
    }

    void Copy_cuda(void* dst, void const* src, std::size_t size, CopyKind ck)
    {
#if VKT_HAVE_CUDA
        cudaMemcpyKind cck;

        switch (ck)
        {
        case CopyKind::HostToHost:
            cck = cudaMemcpyHostToHost;
            break;

        case CopyKind::HostToDevice:
            cck = cudaMemcpyHostToDevice;
            break;

        case CopyKind::DeviceToHost:
            cck = cudaMemcpyDeviceToHost;
            break;

        case CopyKind::DeviceToDevice:
            cck = cudaMemcpyDeviceToDevice;
            break;
        }

        VKT_CUDA_SAFE_CALL__(cudaMemcpy(dst, src, size, cck));
#endif
    }

} // vkt
