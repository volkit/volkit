// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include <vkt/ManagedBuffer.hpp>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    class Histogram : public ManagedBuffer<std::size_t>
    {
    public:
        Histogram(std::size_t numBins);

        std::size_t getNumBins() const;

        std::size_t* getBinCounts();
    };

    VKTAPI Error ComputeHistogram(StructuredVolume& volume,
                                  Histogram& histogram);

    VKTAPI Error ComputeHistogramRange(StructuredVolume& volume,
                                       Histogram& histogram,
                                       int32_t firstX,
                                       int32_t firstY,
                                       int32_t firstZ,
                                       int32_t lastX,
                                       int32_t lastY,
                                       int32_t lastZ);

    VKTAPI Error ComputeHistogramRange(StructuredVolume& volume,
                                       Histogram& histogram,
                                       Vec3i first,
                                       Vec3i last);
} // vkt
