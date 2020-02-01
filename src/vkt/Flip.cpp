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
            vec3i(0, 0, 0),
            volume.getDims(),
            axis
            );

        return NO_ERROR;
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
            vec3i(firstX, firstY, firstZ),
            vec3i(lastX, lastY, lastZ),
            axis
            );

        return NO_ERROR;
    }

    Error FlipRange(StructuredVolume& volume, vec3i first, vec3i last, Axis axis)
    {
        VKT_CALL__(FlipRange, volume, first, last, axis);

        return NO_ERROR;
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
        vkt::vec3i(0, 0, 0),
        volume->volume.getDims(),
        (vkt::Axis)axis
        );

    return VKT_NO_ERROR;
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
        vkt::vec3i(firstX, firstY, firstZ),
        vkt::vec3i(lastX, lastY, lastZ),
        (vkt::Axis)axis
        );

    return VKT_NO_ERROR;
}