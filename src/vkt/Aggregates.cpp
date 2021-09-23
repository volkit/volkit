// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>

#include <vkt/Aggregates.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Aggregates.h>
#include <vkt/StructuredVolume.h>

#include "Aggregates_cuda.hpp"
#include "Aggregates_serial.hpp"
#include "Callable.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error ComputeAggregates(StructuredVolume& volume, Aggregates& aggregates)
    {
        VKT_LEGACY_CALL__(ComputeAggregatesRange, volume, aggregates, { 0, 0, 0 }, volume.getDims());

        return NoError;
    }

    Error ComputeAggregatesRange(
            StructuredVolume& volume,
            Aggregates& aggregates,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ
            )
    {
        VKT_LEGACY_CALL__(
            ComputeAggregatesRange,
            volume,
            aggregates,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ }
            );

        return NoError;
    }

    Error ComputeAggregatesRange(
            StructuredVolume& volume,
            Aggregates& aggregates,
            Vec3i first,
            Vec3i last
            )
    {
        VKT_LEGACY_CALL__(
            ComputeAggregatesRange,
            volume,
            aggregates,
            first,
            last
            );

        return NoError;
    }
} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktComputeAggregatesSV(vktStructuredVolume volume, vktAggregates_t* aggregates)
{
    vkt::Aggregates aggrCPP;

    vkt::ComputeAggregates(volume->volume, aggrCPP);

    std::memcpy(aggregates, &aggrCPP, sizeof(aggrCPP));

    return vktNoError;
}

vktError vktComputeAggregatesRangeSV(
        vktStructuredVolume volume,
        vktAggregates_t* aggregates,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ
        )
{
    vkt::Aggregates aggrCPP;

    vkt::ComputeAggregatesRange(
        volume->volume,
        aggrCPP,
        firstX,
        firstY,
        firstZ,
        lastX,
        lastY,
        lastZ
        );

    std::memcpy(aggregates, &aggrCPP, sizeof(aggrCPP));

    return vktNoError;
}
