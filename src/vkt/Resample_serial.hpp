// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>

#include "linalg.hpp"

namespace vkt
{
    void Resample_serial(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            )
    {
        if (dst.getDims() == src.getDims())
        {
            // In that case don't resample spatially!

            Vec3i dims = dst.getDims();

            for (int32_t z = 0; z != dims.z; ++z)
            {
                for (int32_t y = 0; y != dims.y; ++y)
                {
                    for (int32_t x = 0; x != dims.x; ++x)
                    {
                        Vec3i index{x,y,z};
                        dst.setValue(index, src.getValue(index));
                    }
                }
            }
        }
        else
        {

            Vec3i dstDims = dst.getDims();
            Vec3i srcDims = src.getDims();

            for (int32_t z = 0; z != dstDims.z; ++z)
            {
                for (int32_t y = 0; y != dstDims.y; ++y)
                {
                    for (int32_t x = 0; x != dstDims.x; ++x)
                    {
                        float srcX = x / float(dstDims.x) * srcDims.x;
                        float srcY = y / float(dstDims.y) * srcDims.y;
                        float srcZ = z / float(dstDims.z) * srcDims.z;
                        float value = src.sampleLinear(srcX, srcY, srcZ);
                        dst.setValue({x,y,z}, value);
                    }
                }
            }
        }
    }

} // vkt
