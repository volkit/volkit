// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/Voxel.hpp>

namespace vkt
{
    inline void FillRange_serial(StructuredVolume& volume, Vec3i first, Vec3i last, float value)
    {
        uint8_t mappedVoxel[StructuredVolume::GetMaxBytesPerVoxel()];
        MapVoxel(
            mappedVoxel,
            value,
            volume.getBytesPerVoxel(),
            volume.getVoxelMapping().x,
            volume.getVoxelMapping().y
            );

        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    volume.setBytes(x, y, z, mappedVoxel);
                }
            }
        }
    }
} // vkt
