// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/Aggregates.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void ComputeAggregatesRange_cuda(
            StructuredVolume& volume,
            Aggregates& aggregates,
            Vec3i first,
            Vec3i last
            )
    {
    }

} // vkt
