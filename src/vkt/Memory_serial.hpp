// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <vkt/Memory.hpp>

namespace vkt
{
    inline void Allocate_serial(void** ptr, std::size_t size)
    {
        *ptr = malloc(size);
    }

    inline void Free_serial(void* ptr)
    {
        free(ptr);
    }

    inline void MemsetRange_serial(
            void* dst,
            void const* src,
            std::size_t dstSize,
            std::size_t srcSize
            )
    {
        std::size_t numElem = dstSize / srcSize;

        for (std::size_t i = 0; i < numElem; ++i)
        {
            std::memcpy((char*)dst + i * srcSize, src, srcSize);
        }
    }
} // vkt
