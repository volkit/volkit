// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <vkt/config.h>

#include <vkt/Aggregates.hpp>
#include <vkt/StructuredVolume.hpp>

#include "linalg.hpp"

namespace vkt
{
    void ComputeAggregatesRange_serial(
            StructuredVolume& volume,
            Aggregates& aggregates,
            Vec3i first,
            Vec3i last
            )
    {
        std::memset(&aggregates, 0, sizeof(aggregates));

        aggregates.min  =  FLT_MAX;
        aggregates.max  = -FLT_MAX;
        aggregates.prod = 1.f;

        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    float val = 0.f;
                    volume.getValue(x, y, z, val);

                    if (val < aggregates.min)
                    {
                        aggregates.min = val;
                        aggregates.argmin = { x, y, z };
                    }

                    if (val > aggregates.max)
                    {
                        aggregates.max = val;
                        aggregates.argmax = { x, y, z };
                    }

                    aggregates.mean += val;

                    aggregates.sum += val;
                    aggregates.prod *= val;
                }
            }
        }

        size_t numElems = volume.getDims().x * size_t(volume.getDims().y) * volume.getDims().z;

        aggregates.mean /= (double)numElems;


        // 2nd pass to compute standard deviation and variance
        // TODO: do we want to separate this from the others??
        for (int32_t z = first.z; z != last.z; ++z)
        {
            for (int32_t y = first.y; y != last.y; ++y)
            {
                for (int32_t x = first.x; x != last.x; ++x)
                {
                    float val = 0.f;
                    volume.getValue(x, y, z, val);

                    aggregates.var += (val - aggregates.mean) * (val - aggregates.mean);
                }
            }
        }

        aggregates.var /= (double)numElems;
        aggregates.stddev = sqrtf(aggregates.var);
    }

} // vkt
