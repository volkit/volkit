// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cassert>
#include <vector>

#include <vkt/Crop.hpp>

#include "Crop_serial.hpp"
#include "DataFormatInfo.hpp"
#include "linalg.hpp"
#include "macros.hpp"

#if VKT_HAVE_CUDA
#include "Crop_cuda.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error CropResize(
            HierarchicalVolume& dst,
            HierarchicalVolume& src,
            Vec3i first,
            Vec3i last
            )
    {
        // TODO: that only works when on the CPU!
        std::vector<Brick> newBricks;
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
                unsigned level = src.getBricks()[i].level;

                unsigned levelX = level;
                unsigned cellW = 1<<levelX;
                while (newDims.x % cellW != 0)
                {
                    levelX >>= 1;
                    cellW = 1<<levelX;
                }

                unsigned levelY = level;
                unsigned cellH = 1<<levelY;
                while (newDims.y % cellH != 0)
                {
                    levelY >>= 1;
                    cellH = 1<<levelY;
                }

                unsigned levelZ = level;
                unsigned cellD = 1<<levelZ;
                while (newDims.z % cellD != 0)
                {
                    levelZ >>= 1;
                    cellD = 1<<levelZ;
                }

                unsigned newLevel = min(min(levelX, levelY), levelZ);

                std::size_t offsetInBytes = 0;
                if (!newBricks.empty())
                {
                    Brick prev = newBricks.back();
                    offsetInBytes = prev.offsetInBytes
                        + prev.dims.x * prev.dims.y * prev.dims.z
                        * getSizeInBytes(src.getDataFormat());
                }

                newDims.x >>= newLevel;
                newDims.y >>= newLevel;
                newDims.z >>= newLevel;
                assert(newLevel <= level);

                newBricks.push_back({
                    lo-first, newDims, offsetInBytes, newLevel
                    });
            }
        }

        dst.setBricks(newBricks.data(), newBricks.size());
    }

    Error Crop(
            HierarchicalVolume& dst,
            HierarchicalVolume& src,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ
            )
    {
        VKT_CALL__(
            Crop,
            dst,
            src,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error Crop(
            HierarchicalVolume& dst,
            HierarchicalVolume& src,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_CALL__(Crop, dst, src, first, last);

        return NoError;
    }
} // vkt
