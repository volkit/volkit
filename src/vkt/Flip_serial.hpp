#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    inline void FlipRange_serial(StructuredVolume& volume, vec3i first, vec3i last, Axis axis)
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

        vec3i dims = volume.getDims();

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

                    volume.getVoxel(x, y, z, voxel1);
                    volume.getVoxel(xx, yy, zz, voxel2);

                    volume.setVoxel(x, y, z, voxel2);
                    volume.setVoxel(xx, yy, zz, voxel1);
                }
            }
        }
    }
} // vkt
