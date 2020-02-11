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
} // vkt
