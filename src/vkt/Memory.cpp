// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cstring>

#if VKT_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <vkt/Memory.hpp>

#include "macros.hpp"
#include "Memory_cuda.hpp"
#include "Memory_serial.hpp"

namespace vkt
{
    void Allocate(void** ptr, std::size_t size)
    {
        VKT_CALL__(Allocate, ptr, size);
    }

    void Free(void* ptr)
    {
        VKT_CALL__(Free, ptr);
    }

    void Copy(void* dst, void const* src, std::size_t size, CopyKind ck)
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
#else
        assert(ck == CopyKind::HostToHost);

        std::memcpy(dst, src, size);
#endif
    }

} // vkt
