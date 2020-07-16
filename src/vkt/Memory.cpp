// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cassert>
#include <cstring>

#if VKT_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <vkt/Memory.hpp>

#include <vkt/Memory.h>

#include "macros.hpp"
#include "Memory_cuda.hpp"
#include "Memory_serial.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

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

    void Memcpy(void* dst, void const* src, std::size_t size, CopyKind ck)
    {
        if (ck == CopyKind::HostToHost)
        {
            std::memcpy(dst, src, size);
        }
        else
        {
#if VKT_HAVE_CUDA
            cudaMemcpyKind cck;

            switch (ck)
            {
            case CopyKind::HostToHost:
                assert(0);
                return;

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
        }
#endif
    }

    void MemsetRange(void* dst, void const* src, std::size_t dstSize, std::size_t srcSize)
    {
        VKT_CALL__(MemsetRange, dst, src, dstSize, srcSize);
    }
} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

void vktAllocate(void** ptr, size_t size)
{
    VKT_CALL__(vkt::Allocate, ptr, size);
}

void vktFree(void* ptr)
{
    VKT_CALL__(vkt::Free, ptr);
}

void vktMemcpy(void* dst, void const* src, size_t size, vktCopyKind ck)
{
    vkt::Memcpy(dst, src, size, (vkt::CopyKind)ck);
}
