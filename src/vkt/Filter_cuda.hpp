// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void ApplyFilterRange_cuda(
        StructuredVolume& dest,
        StructuredVolume& source,
        Vec3i first,
        Vec3i last,
        Filter filter,
        AddressMode am
        )
    {}

} // vkt
