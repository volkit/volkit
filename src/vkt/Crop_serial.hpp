// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>
#include <cstring>

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
        // Make a copy of these so we're not querying the execution policy
        // all the time (which is quite expensive)
        uint8_t* dstData = dst.getData();
        uint8_t* srcData = src.getData();

        // Assumes that dst was resized with CropResize()
        std::vector<int> brickIDs(dst.getNumBricks());
        std::memcpy(brickIDs.data(), dstData, sizeof(int) * brickIDs.size());

        for (std::size_t i = 0; i < dst.getNumBricks(); ++i)
        {
            Brick oldBrick = src.getBricks()[brickIDs[i]];
            Brick newBrick = dst.getBricks()[i];

            std::memcpy(dstData + newBrick.offsetInBytes, srcData + oldBrick.offsetInBytes,
                            oldBrick.dims.x * oldBrick.dims.y * oldBrick.dims.z
                               * vkt::getSizeInBytes(src.getDataFormat()));
        }
    }
} // vkt
