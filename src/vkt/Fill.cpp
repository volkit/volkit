#include <vkt/Fill.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Fill.h>
#include <vkt/StructuredVolume.h>

#include "Fill_cuda.hpp"
#include "Fill_serial.hpp"
#include "macros.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Fill(StructuredVolume& volume, float value)
    {
        VKT_CALL__(
            FillRange,
            volume,
            Vec3i(0, 0, 0),
            volume.getDims(),
            value);

        return NO_ERROR;
    }

    Error FillRange(
            StructuredVolume& volume,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ,
            float value
            )
    {
        VKT_CALL__(
            FillRange,
            volume,
            Vec3i(firstX, firstY, firstZ),
            Vec3i(lastX, lastY, lastZ),
            value
            );

        return NO_ERROR;
    }

    Error FillRange(StructuredVolume& volume, Vec3i first, Vec3i last, float value)
    {
        VKT_CALL__(FillRange, volume, first, last, value);

        return NO_ERROR;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktFillSV(vktStructuredVolume volume, float value)
{
    VKT_CALL__(
        FillRange,
        volume->volume,
        vkt::Vec3i(0, 0, 0),
        volume->volume.getDims(),
        value);

    return VKT_NO_ERROR;
}

vktError vktFillRangeSV(
        vktStructuredVolume volume,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ,
        float value)
{
    VKT_CALL__(
        FillRange,
        volume->volume,
        vkt::Vec3i(firstX, firstY, firstZ),
        vkt::Vec3i(lastX, lastY, lastZ),
        value);

    return VKT_NO_ERROR;
}
