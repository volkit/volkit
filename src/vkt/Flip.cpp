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
    Error Flip(StructuredVolume& volume, Axis axis)
    {
        VKT_CALL__(
            FlipRange,
            volume,
            Vec3i(0, 0, 0),
            volume.getDims(),
            axis
            );

        return NoError;
    }

    Error FlipRange(
            StructuredVolume& volume,
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
            volume,
            Vec3i(firstX, firstY, firstZ),
            Vec3i(lastX, lastY, lastZ),
            axis
            );

        return NoError;
    }

    Error FlipRange(StructuredVolume& volume, Vec3i first, Vec3i last, Axis axis)
    {
        VKT_CALL__(FlipRange, volume, first, last, axis);

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktFlipSV(vktStructuredVolume volume, vktAxis axis)
{
    VKT_CALL__(
        FlipRange,
        volume->volume,
        vkt::Vec3i(0, 0, 0),
        volume->volume.getDims(),
        (vkt::Axis)axis
        );

    return vktNoError;
}

vktError vktFlipRangeSV(
        vktStructuredVolume volume,
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
        volume->volume,
        vkt::Vec3i(firstX, firstY, firstZ),
        vkt::Vec3i(lastX, lastY, lastZ),
        (vkt::Axis)axis
        );

    return vktNoError;
}
