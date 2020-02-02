// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>

#include <vkt/Aggregates.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Aggregates.h>
#include <vkt/StructuredVolume.h>

#include "Aggregates_cuda.hpp"
#include "Aggregates_serial.hpp"
#include "macros.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    VKTAPI Error ComputeAggregates(StructuredVolume& volume, Aggregates& aggregates)
    {
        VKT_CALL__(ComputeAggregatesRange, volume, aggregates, Vec3i(0, 0, 0), volume.getDims());

        return NoError;
    }
} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

VKTAPI vktError vktComputeAggregatesSV(vktStructuredVolume volume, vktAggregates_t* aggregates)
{
    vkt::Aggregates aggrCPP;

    vkt::ComputeAggregates(volume->volume, aggrCPP);

    std::memcpy(aggregates, &aggrCPP, sizeof(aggrCPP));

    return vktNoError;
}
