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
        std::vector<int> brickIDs(dst.getNumBricks());
        std::memcpy(brickIDs.data(), dst.getData(), sizeof(int) * brickIDs.size());

        for (std::size_t i = 0; i < dst.getNumBricks(); ++i)
        {
            Brick oldBrick = src.getBricks()[brickIDs[i]];
            Brick newBrick = dst.getBricks()[i];

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
} // vkt
