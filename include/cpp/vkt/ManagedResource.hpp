// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"

namespace vkt
{
    typedef void* ManagedResource;
    typedef uint32_t ResourceHandle;

    VKTAPI ResourceHandle RegisterManagedResource(ManagedResource resource);

    VKTAPI void UnregisterManagedResource(ResourceHandle handle);

    VKTAPI ManagedResource GetManagedResource(ResourceHandle handle);

} // vkt
