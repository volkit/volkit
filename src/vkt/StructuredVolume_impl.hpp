// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include <vkt/StructuredVolume.hpp>

struct vktStructuredVolume_impl
{
    vktStructuredVolume_impl(
            int32_t dimx,
            int32_t dimy,
            int32_t dimz,
            uint16_t bytesPerVoxel,
            float distX,
            float distY,
            float distZ,
            float mappingLo,
            float mappingHi)
        : volume(dimx, dimy, dimz, bytesPerVoxel, distX, distY, distZ, mappingLo, mappingHi)
    {
    }

    vktStructuredVolume_impl(vkt::StructuredVolume& rhs)
        : volume(rhs)
    {
    }

    vkt::StructuredVolume volume;
};
