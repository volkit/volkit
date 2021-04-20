// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

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

    void ResampleCLAHE_serial(
            StructuredVolume& dst,
            StructuredVolume& src
            )
    {
        auto imageLoad = [&](int x, int y, int z)
            -> unsigned
        {
                    uint8_t data[4];
                    src.getBytes(x, y, z, data);

                    unsigned ival;
                    switch (dst.getDataFormat())
                    {
                    case DataFormat::UInt8:
                        ival = (unsigned)data[0];
                        break;

                    case DataFormat::UInt16:
#ifdef VKT_LITTLE_ENDIAN
                        ival = static_cast<unsigned>(data[0])
                                      | static_cast<unsigned>(data[1] << 8);
#else
                        ival = static_cast<unsigned>(data[0] << 8)
                                      | static_cast<unsigned>(data[1]);
#endif
                        break;
                    }

                    return ival;
        };

        Vec3i dstDims = dst.getDims();
        Vec3i srcDims = src.getDims();

        assert(dstDims == srcDims);
        assert(dst.getDataFormat() == src.getDataFormat());

        // Min-max "shader"
        unsigned globalMin = UINT_MAX;
        unsigned globalMax = 0U;
        for (int32_t z = 0; z != dstDims.z; ++z)
        {
            for (int32_t y = 0; y != dstDims.y; ++y)
            {
                for (int32_t x = 0; x != dstDims.x; ++x)
                {
                    unsigned ival = imageLoad(x,y,z);
                    globalMin = std::min(globalMin, ival);
                    globalMax = std::max(globalMax, ival);
                }
            }
        }

        // LUT shader
        constexpr static unsigned NumBins = 1<<16;
        unsigned LUT[NumBins];
        std::fill(LUT, LUT+NumBins, 0);

        unsigned binSize = 1 + uint((globalMax - globalMin) / NumBins);
        for (unsigned index = 0; index < NumBins; ++index)
        {
            LUT[index] = ( index - globalMin ) / binSize;
        }

        Vec3i numSB{4,4,2};

        unsigned offsetX = 0;
        unsigned offsetY = 0;
        unsigned offsetZ = 0;
        bool useLUT = true;

        std::vector<unsigned> hist(numSB.x*numSB.y*numSB.z*NumBins);
        std::fill(hist.begin(), hist.end(), 0);

        std::vector<unsigned> histMax(numSB.x*numSB.y*numSB.z);
        std::fill(histMax.begin(), histMax.end(), 0);

        // Hist shader
        for (int32_t z = 0; z != dstDims.z; ++z)
        {
            for (int32_t y = 0; y != dstDims.y; ++y)
            {
                for (int32_t x = 0; x != dstDims.x; ++x)
                {
                    Vec3i index{x,y,z};
                    Vec3i sizeSB = dst.getDims() / numSB;
                    Vec3i currSB = index / sizeSB;
                    // if we are not within the volume of interest -> return 

                    // get the gray value of the Volume
                    unsigned volSample = imageLoad(x+offsetX, y+offsetY, z+offsetZ);
                    unsigned histIndex = (currSB.z * numSB.x * numSB.y + currSB.y * numSB.x + currSB.x);
                    
                    // Increment the appropriate histogram
                    unsigned grayIndex = (NumBins * histIndex) + volSample;
                    if (useLUT){
                        grayIndex = (NumBins * histIndex) + LUT[ volSample ];
                    }
                    // atomicAdd( hist[ grayIndex ], 1 );
                    hist[grayIndex]++;

                    // update the histograms max value
                    // atomicMax( histMax[ histIndex ], hist[ grayIndex ] );
                    histMax[ histIndex ] += hist[ grayIndex ];
                }
            }
        }
    }
} // vkt
