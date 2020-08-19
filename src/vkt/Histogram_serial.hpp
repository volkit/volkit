// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>
#include <cstring>

#include <vkt/config.h>

#include <vkt/Histogram.hpp>
#include <vkt/Memory.hpp>
#include <vkt/StructuredVolume.hpp>

#include "linalg.hpp"

namespace vkt
{
    void ComputeHistogramRange_serial(
            StructuredVolume& volume,
            Histogram& histogram,
            Vec3i first,
            Vec3i last
            )
    {
        float lo = volume.getVoxelMapping().x;
        float hi = volume.getVoxelMapping().y;

        std::size_t numBins = histogram.getNumBins();

        std::size_t* bins = histogram.getBinCounts();

        std::size_t zero = 0;

        MemsetRange(bins, &zero, sizeof(std::size_t) * numBins, sizeof(zero));

        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    float val = volume.getValue(x, y, z);

                    std::size_t binID = (std::size_t)((val - lo) * (numBins / (hi - lo)));

                    bins[binID]++;
                }
            }
        }
    }
} // vkt
