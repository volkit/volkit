// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>

#include <vkt/HierarchicalVolume.hpp>

#include "DataFormatInfo.hpp"
#include "linalg.hpp"

namespace vkt
{
    inline void Crop_serial(
            HierarchicalVolume& dst,
            HierarchicalVolume& src,
            Vec3i first,
            Vec3i last
            )
    {
        // Assumes that dst was resized with CropResize()

        // ID of the *new* brick
        int brickID = 0;

        for (std::size_t i = 0; i < src.getNumBricks(); ++i)
        {
            Vec3i lo{0,0,0};
            Vec3i hi = src.getBricks()[i].dims;
            lo *= (int)(1<<src.getBricks()[i].level);
            hi *= (int)(1<<src.getBricks()[i].level);
            lo += src.getBricks()[i].lower;
            hi += src.getBricks()[i].lower;

            lo = max(lo, first);
            hi = min(hi, last);

            Vec3i newDims = hi - lo;
            if (newDims.x > 0 && newDims.y > 0 && newDims.z > 0)
            {
                Brick oldBrick = src.getBricks()[i];
                Brick newBrick = dst.getBricks()[brickID++];

                unsigned levelDiff = oldBrick.level-newBrick.level;
                // That's by how much we have to multiply the
                // new cell size to obtain the old cell size
                unsigned sizeDiff = 1<<levelDiff;

                Vec3i loDiff = (newBrick.lower+first)-oldBrick.lower;

                std::size_t offNew = newBrick.offsetInBytes;

                for (int z = 0; z < newBrick.dims.z; z += sizeDiff)
                {
                    for (int y = 0; y < newBrick.dims.y; y += sizeDiff)
                    {
                        for (int x = 0; x < newBrick.dims.x; x += sizeDiff)
                        {
                            int oldX = (loDiff.x>>oldBrick.level) + x / sizeDiff;
                            int oldY = (loDiff.y>>oldBrick.level) + y / sizeDiff;
                            int oldZ = (loDiff.z>>oldBrick.level) + z / sizeDiff;

                            std::size_t off = oldBrick.offsetInBytes
                                + (oldZ * oldBrick.dims.x * oldBrick.dims.y
                                 + oldY * oldBrick.dims.x
                                 + oldX) * vkt::getSizeInBytes(src.getDataFormat());

                            uint8_t bytes[HierarchicalVolume::GetMaxBytesPerVoxel()];

                            std::memcpy(bytes, src.getData() + off,
                                        vkt::getSizeInBytes(src.getDataFormat()));

                            for (unsigned d = 0; d < sizeDiff; ++d)
                            {
                                std::memcpy(dst.getData() + offNew, bytes,
                                            vkt::getSizeInBytes(src.getDataFormat()));
                                offNew += vkt::getSizeInBytes(src.getDataFormat());
                            }
                        }
                    }
                }
            }
        }
    }
} // vkt
