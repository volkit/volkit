// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdlib>

#include <vkt/HierarchicalVolume.hpp>
#include <vkt/StructuredVolume.hpp>
#include <vkt/Voxel.hpp>

#include "DataFormatInfo.hpp"
#include "linalg.hpp"

namespace vkt
{
    inline void FillRange_serial(StructuredVolume& volume, Vec3i first, Vec3i last, float value)
    {
        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    volume.setValue(x, y, z, value);
                }
            }
        }
    }

    inline void FillRange_serial(HierarchicalVolume& volume, Vec3i first, Vec3i last, float value)
    {
        uint8_t mappedVoxel[HierarchicalVolume::GetMaxBytesPerVoxel()];
        MapVoxel(
            mappedVoxel,
            value,
            volume.getDataFormat(),
            volume.getVoxelMapping().x,
            volume.getVoxelMapping().y
            );

        for (std::size_t i = 0; i < volume.getNumBricks(); ++i)
        {
            Brick const& brick = volume.getBricks()[i];

            Vec3i lo = brick.lower;
            Vec3i hi = brick.lower + brick.dims * (1<<brick.level);

            Vec3i start = max(lo, first);
            Vec3i end   = min(hi, last);
            Vec3i inc   = {1<<brick.level,1<<brick.level,1<<brick.level};

            // That's "nearest" interpolation (TODO!)
            for (int z = start.z; z < end.z; z += inc.z) // important that this is < and not != !!!
            {
                for (int y = start.y; y < end.y; y += inc.y)
                {
                    for (int x = start.x; x < end.x; x += inc.x)
                    {
                        int ix = (x - brick.lower.x) / inc.x;
                        int iy = (y - brick.lower.y) / inc.y;
                        int iz = (z - brick.lower.z) / inc.z;

                        static const std::size_t bytesPerVoxel = getSizeInBytes(volume.getDataFormat());

                        const std::size_t idx
                          = brick.offsetInBytes
                          + (ix
                          + iy * brick.dims.x
                          + iz * brick.dims.x * brick.dims.y) * bytesPerVoxel;

                        std::memcpy(volume.getData() + idx, mappedVoxel, bytesPerVoxel);
                    }
                }
            }
        }
    }
} // vkt
