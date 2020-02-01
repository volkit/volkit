#include <vkt/Transform.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Transform.h>
#include <vkt/StructuredVolume.h>

#include "macros.hpp"
#include "StructuredVolume_impl.hpp"
#include "Transform_cuda.hpp"
#include "Transform_serial.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
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
        VKT_CALL__(
            TransformRange,
            volume,
            Vec3i(firstX, firstY, firstZ),
            Vec3i(lastX, lastY, lastZ),
            unaryOp
            );

        return NO_ERROR;
    }

    Error TransformRange(
            StructuredVolume& volume,
            Vec3i first,
            Vec3i last,
            TransformUnaryOp unaryOp
            )
    {
        VKT_CALL__(TransformRange, volume, first, last, unaryOp);

        return NO_ERROR;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktTransformSV1(vktStructuredVolume volume, vktTransformUnaryOp unaryOp)
{
    VKT_CALL__(
        TransformRange,
        volume->volume,
        vkt::Vec3i(0, 0, 0),
        volume->volume.getDims(),
        unaryOp
        );

    return VKT_NO_ERROR;
}

vktError vktTransformSV2(
        vktStructuredVolume volume1,
        vktStructuredVolume volume2,
        vktTransformBinaryOp binaryOp)
{
    VKT_CALL__(
        TransformRange,
        volume1->volume,
        volume2->volume,
        vkt::Vec3i(0, 0, 0),
        volume1->volume.getDims(),
        binaryOp
        );

    return VKT_NO_ERROR;
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
    VKT_CALL__(
        TransformRange,
        volume->volume,
        vkt::Vec3i(firstX, firstY, firstZ),
        vkt::Vec3i(lastX, lastY, lastZ),
        (vkt::TransformUnaryOp)unaryOp
        );

    return VKT_NO_ERROR;
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
    VKT_CALL__(
        TransformRange,
        volume1->volume,
        volume2->volume,
        vkt::Vec3i(firstX, firstY, firstZ),
        vkt::Vec3i(lastX, lastY, lastZ),
        (vktTransformBinaryOp)binaryOp
        );

    return VKT_NO_ERROR;
}
