// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <vkt/Transform.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Transform.h>
#include <vkt/StructuredVolume.h>

#include "Callable.hpp"
#include "StructuredVolume_impl.hpp"
#include "Transform_serial.hpp"

#if VKT_HAVE_CUDA
#include "Transform_cuda.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Transform(StructuredVolume& volume, TransformUnaryOp unaryOp)
    {
        VKT_LEGACY_CALL__(
            TransformRange,
            volume,
            { 0, 0, 0 },
            volume.getDims(),
            unaryOp
            );

        return NoError;
    }

    Error TransformRange(
            StructuredVolume& volume,
            int32_t firstX,
            int32_t firstY,
            int32_t firstZ,
            int32_t lastX,
            int32_t lastY,
            int32_t lastZ,
            TransformUnaryOp unaryOp
            )
    {
        VKT_LEGACY_CALL__(
            TransformRange,
            volume,
            { firstX, firstY, firstZ },
            { lastX, lastY, lastZ },
            unaryOp
            );

        return NoError;
    }

    Error TransformRange(
            StructuredVolume& volume,
            Vec3i first,
            Vec3i last,
            TransformUnaryOp unaryOp
            )
    {
        VKT_LEGACY_CALL__(TransformRange, volume, first, last, unaryOp);

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktTransformSV1(vktStructuredVolume volume, vktTransformUnaryOp unaryOp)
{
    VKT_LEGACY_CALL__(
        TransformRange,
        volume->volume,
        { 0, 0, 0 },
        volume->volume.getDims(),
        (vkt::TransformUnaryOp)unaryOp
        );

    return vktNoError;
}

vktError vktTransformSV2(
        vktStructuredVolume volume1,
        vktStructuredVolume volume2,
        vktTransformBinaryOp binaryOp)
{
    VKT_LEGACY_CALL__(
        TransformRange,
        volume1->volume,
        volume2->volume,
        { 0, 0, 0 },
        volume1->volume.getDims(),
        (vkt::TransformBinaryOp)binaryOp
        );

    return vktNoError;
}

vktError vktTransformRangeSV1(
        vktStructuredVolume volume,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ,
        vktTransformUnaryOp unaryOp
        )
{
    VKT_LEGACY_CALL__(
        TransformRange,
        volume->volume,
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        (vkt::TransformUnaryOp)unaryOp
        );

    return vktNoError;
}

vktError vktTransformRangeSV2(
        vktStructuredVolume volume1,
        vktStructuredVolume volume2,
        int32_t firstX,
        int32_t firstY,
        int32_t firstZ,
        int32_t lastX,
        int32_t lastY,
        int32_t lastZ,
        vktTransformBinaryOp binaryOp
        )
{
    VKT_LEGACY_CALL__(
        TransformRange,
        volume1->volume,
        volume2->volume,
        { firstX, firstY, firstZ },
        { lastX, lastY, lastZ },
        (vkt::TransformBinaryOp)binaryOp
        );

    return vktNoError;
}
