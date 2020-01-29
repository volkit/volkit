#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "Memory.hpp"

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

    inline void Copy_serial(void* dst, void const* src, std::size_t size, CopyKind)
    {
        std::memcpy(dst, src, size);
    }
} // vkt
