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
            axis,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 }
            );

        return NoError;
    }

    Error FlipRange(
            StructuredVolume& dest,
            StructuredVolume& source,
            Axis axis,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ,
            int32_t dstOffsetX,
            int32_t dstOffsetY,
            int32_t dstOffsetZ
            )
    {
        VKT_CALL__(
            FlipRange,
            dest,
            source,
            axis,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ }
            );

        return NoError;
    }

    Error FlipRange(
            StructuredVolume& dest,
            StructuredVolume& source,
            Axis axis,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
        VKT_CALL__(FlipRange, dest, source, axis, first, last, dstOffset);

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
        (vkt::Axis)axis,
        { 0, 0, 0 },
        dest->volume.getDims(),
        { 0, 0, 0 }
        );

    return vktNoError;
}

vktError vktFlipRangeSV(
        vktStructuredVolume dest,
        vktStructuredVolume source,
        vktAxis axis,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ,
        int32_t dstOffsetX,
        int32_t dstOffsetY,
        int32_t dstOffsetZ
        )
{
    VKT_CALL__(
        FlipRange,
        dest->volume,
        source->volume,
        (vkt::Axis)axis,
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        { dstOffsetX, dstOffsetY, dstOffsetZ }
        );

    return vktNoError;
}
