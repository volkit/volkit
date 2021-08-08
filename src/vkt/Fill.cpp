// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cstring>

#include <vkt/Fill.hpp>
#include <vkt/HierarchicalVolume.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Fill.h>
#include <vkt/HierarchicalVolume.h>
#include <vkt/StructuredVolume.h>

#include "Fill_serial.hpp"
#include "HierarchicalVolume_impl.hpp"
#include "macros.hpp"
#include "StructuredVolume_impl.hpp"

#if VKT_HAVE_CUDA
#include "Fill_cuda.hpp"
#endif

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
            { 0, 0, 0 },
            volume.getDims(),
            value);

        return NoError;
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
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            value
            );

        return NoError;
    }

    Error FillRange(StructuredVolume& volume, Vec3i first, Vec3i last, float value)
    {
        VKT_CALL__(FillRange, volume, first, last, value);

        return NoError;
    }

    Error Fill(HierarchicalVolume& volume, float value)
    {
        VKT_CALL__(
            FillRange,
            volume,
            { 0, 0, 0 },
            volume.getDims(),
            value);

        return NoError;
    }

    Error FillRange(
            HierarchicalVolume& volume,
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
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            value
            );

        return NoError;
    }

    Error FillRange(HierarchicalVolume& volume, Vec3i first, Vec3i last, float value)
    {
        VKT_CALL__(FillRange, volume, first, last, value);

        return NoError;
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
        { 0, 0, 0 },
        volume->volume.getDims(),
        value
        );

    return vktNoError;
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
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        value);

    return vktNoError;
}

vktError vktFillHV(vktHierarchicalVolume volume, float value)
{
    VKT_CALL__(
        FillRange,
        volume->volume,
        { 0, 0, 0 },
        volume->volume.getDims(),
        value
        );

    return vktNoError;
}

vktError vktFillRangeHV(
        vktHierarchicalVolume volume,
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
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        value);

    return vktNoError;
}
