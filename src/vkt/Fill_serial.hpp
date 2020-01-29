#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    inline void FillRange_serial(StructuredVolume& volume, vec3i first, vec3i last, float value)
    {
        uint8_t mappedVoxel[StructuredVolume::GetMaxBytesPerVoxel()];
        volume.mapVoxel(mappedVoxel, value);

        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    volume.setVoxel(x, y, z, mappedVoxel);
                }
            }
        }
    }
} // vkt
