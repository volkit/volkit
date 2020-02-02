// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    struct Aggregates
    {
        float min;
        float max;
        float mean;
        float stddev;
        float var;
        float sum;
        float prod;
        Vec3i argmin;
        Vec3i argmax;
    };

    VKTAPI Error ComputeAggregates(StructuredVolume& volume,
                                   Aggregates& aggregates);

    VKTAPI Error ComputeAggregatesRange(StructuredVolume& volume,
                                        Aggregates& aggregates,
                                        int32_t firstX,
                                        int32_t firstY,
                                        int32_t firstZ,
                                        int32_t lastX,
                                        int32_t lastY,
                                        int32_t lastZ);

    VKTAPI Error ComputeAggregatesRange(StructuredVolume& volume,
                                        Aggregates& aggregates,
                                        Vec3i first,
                                        Vec3i last);

} // vkt
