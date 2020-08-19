// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/Histogram.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void ComputeHistogramRange_cuda(
            StructuredVolume& volume,
            Histogram& histogram,
            Vec3i first,
            Vec3i last
            )
    {
    }

} // vkt
