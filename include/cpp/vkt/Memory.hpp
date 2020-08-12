// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include "common.hpp"

/*! @file  Memory.hpp
 * @brief  Memory manipulation functions that interoperate between CPU and GPU
 */
namespace vkt
{
    enum class CopyKind
    {
        HostToHost,
        HostToDevice,
        DeviceToHost,
        DeviceToDevice,
    };

    VKTAPI void Allocate(void** ptr, std::size_t size);

    VKTAPI void Free(void* ptr);

    VKTAPI void Memcpy(void* dst, void const* src, std::size_t size, CopyKind ck);

    //! Fill operation - not called Fill() to avoid name clashes with the volkit Fill algorithm
    VKTAPI void MemsetRange(void* dst, void const* src, std::size_t dstSize, std::size_t srcSize);

} // vkt
