// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <vkt/Flip.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Flip.h>
#include <vkt/StructuredVolume.h>

#include "Callable.hpp"
#include "Flip_serial.hpp"
#include "StructuredVolume_impl.hpp"

#if VKT_HAVE_CUDA
#include "Flip_cuda.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Flip(StructuredVolume& dest, StructuredVolume& source, Axis axis)
    {
        VKT_LEGACY_CALL__(
            FlipRange,
            dest,
            source,
            { 0, 0, 0 },
            dest.getDims(),
            { 0, 0, 0 },
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
        VKT_LEGACY_CALL__(
            FlipRange,
            dest,
            source,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { 0, 0, 0 },
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
            int32_t dstOffsetX,
            int32_t dstOffsetY,
            int32_t dstOffsetZ,
            Axis axis
            )
    {
        VKT_LEGACY_CALL__(
            FlipRange,
            dest,
            source,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            { dstOffsetX, dstOffsetY, dstOffsetZ },
            axis
            );

        return NoError;
    }

    Error FlipRange(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset,
            Axis axis
            )
    {
        VKT_LEGACY_CALL__(FlipRange, dest, source, first, last, dstOffset, axis);

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
        VKT_LEGACY_CALL__(FlipRange, dest, source, first, last, { 0, 0, 0 }, axis);

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktFlipSV(vktStructuredVolume dest, vktStructuredVolume source, vktAxis axis)
{
    VKT_LEGACY_CALL__(
        FlipRange,
        dest->volume,
        source->volume,
        { 0, 0, 0 },
        dest->volume.getDims(),
        { 0, 0, 0 },
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
        int32_t dstOffsetX,
        int32_t dstOffsetY,
        int32_t dstOffsetZ,
        vktAxis axis
        )
{
    VKT_LEGACY_CALL__(
        FlipRange,
        dest->volume,
        source->volume,
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        { dstOffsetX, dstOffsetY, dstOffsetZ },
        (vkt::Axis)axis
        );

    return vktNoError;
}
