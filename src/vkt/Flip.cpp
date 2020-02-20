// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Flip.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Flip.h>
#include <vkt/StructuredVolume.h>

#include "Flip_cuda.hpp"
#include "Flip_serial.hpp"
#include "macros.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Flip(StructuredVolume& dest, StructuredVolume& source, Axis axis)
    {
        VKT_CALL__(
            FlipRange,
            dest,
            source,
            { 0, 0, 0 },
            dest.getDims(),
            axis
            );

        return NoError;
    }

    Error FlipRange(
            StructuredVolume& dest,
            StructuredVolume& source,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ,
            Axis axis
            )
    {
        VKT_CALL__(
            FlipRange,
            dest,
            source,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            axis
            );

        return NoError;
    }

    Error FlipRange(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Axis axis
            )
    {
        VKT_CALL__(FlipRange, dest, source, first, last, axis);

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktFlipSV(vktStructuredVolume dest, vktStructuredVolume source, vktAxis axis)
{
    VKT_CALL__(
        FlipRange,
        dest->volume,
        source->volume,
        { 0, 0, 0 },
        dest->volume.getDims(),
        (vkt::Axis)axis
        );

    return vktNoError;
}

vktError vktFlipRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ,
        vktAxis axis
        )
{
    VKT_CALL__(
        FlipRange,
        dest->volume,
        source->volume,
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        (vkt::Axis)axis
        );

    return vktNoError;
}
