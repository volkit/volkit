// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    inline void FlipRange_serial(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Axis axis
            )
    {
        int32_t rangeX = last.x - first.x;
        int32_t rangeY = last.y - first.y;
        int32_t rangeZ = last.z - first.z;

        switch (axis)
        {
        case Axis::X:
            rangeX >>= 1;
            break;

        case Axis::Y:
            rangeY >>= 1;
            break;

        case Axis::Z:
            rangeZ >>= 1;
            break;
        }

        Vec3i dims = dest.getDims();

        for (int32_t z = 0; z < rangeZ; ++z)
        {
            for (int32_t y = 0; y < rangeY; ++y)
            {
                for (int32_t x = 0; x < rangeX; ++x)
                {
                    int32_t xx = (axis == Axis::X) ? last.x - 1 - x : x;
                    int32_t yy = (axis == Axis::Y) ? last.y - 1 - y : y;
                    int32_t zz = (axis == Axis::Y) ? last.z - 1 - z : z;

                    uint8_t voxel1[StructuredVolume::GetMaxBytesPerVoxel()];
                    uint8_t voxel2[StructuredVolume::GetMaxBytesPerVoxel()];

                    // Exchange in a way that would even work if dest eq source
                    source.getBytes(x, y, z, voxel1);
                    source.getBytes(xx, yy, zz, voxel2);

                    dest.setBytes(x, y, z, voxel2);
                    dest.setBytes(xx, yy, zz, voxel1);
                }
            }
        }
    }
} // vkt
