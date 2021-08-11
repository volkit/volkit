// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cassert>
#include <vector>

#include <vkt/Crop.hpp>
#include <vkt/ExecutionPolicy.hpp>
#include <vkt/Memory.hpp>

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
        Brick* srcBricks = nullptr;

        ExecutionPolicy ep = GetThreadExecutionPolicy();
        if (ep.device == vkt::ExecutionPolicy::Device::GPU)
        {
            srcBricks = new Brick[src.getNumBricks()];
            Memcpy(srcBricks, src.getBricks(), src.getNumBricks() * sizeof(Brick),
                   CopyKind::DeviceToHost);
        }
        else
            srcBricks = src.getBricks();


        std::vector<Brick> newBricks;
        std::vector<int> newBrickIDs;
        std::size_t offsetInBytes = 0;
        for (std::size_t i = 0; i < src.getNumBricks(); ++i)
        {
            Vec3i lo{0,0,0};
            Vec3i hi = srcBricks[i].dims;
            lo *= (int)(1<<srcBricks[i].level);
            hi *= (int)(1<<srcBricks[i].level);
            lo += srcBricks[i].lower;
            hi += srcBricks[i].lower;

            Box3i isect = { max(lo, first), min(hi, last) };

            if (isect.max.x - isect.min.x > 0 && isect.max.y - isect.min.y > 0 && isect.max.z - isect.min.z > 0)
            {
                newBricks.push_back({
                    lo-first, srcBricks[i].dims, offsetInBytes, srcBricks[i].level
                    });

                offsetInBytes += srcBricks[i].dims.x * srcBricks[i].dims.y * srcBricks[i].dims.z
                                    * vkt::getSizeInBytes(src.getDataFormat());

                newBrickIDs.push_back((int)i);
            }
        }

        dst.setBricks(newBricks.data(), newBricks.size());

        dst.setMaxDims(last-first);

        // Trick: store the old to new brick correspondences in the
        // scalar field array that we just allocated via setBricks.
        // That array is filled with invalid values anyway, might
        // as well just put that memory to good use..

        ep = GetThreadExecutionPolicy();
        CopyKind ck = CopyKind::HostToHost;
        if (ep.device == vkt::ExecutionPolicy::Device::GPU)
        {
            delete[] srcBricks;
            ck = CopyKind::HostToDevice;
        }

        Memcpy(dst.getData(), newBrickIDs.data(), sizeof(int) * newBricks.size(), ck);

        return NoError;
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
