// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/HierarchicalVolume.hpp>
#include <vkt/linalg.hpp>

namespace vkt
{
    void Crop_cuda(
            HierarchicalVolume& dst,
            HierarchicalVolume& src,
            Vec3i first,
            Vec3i last
            );
} // vkt
