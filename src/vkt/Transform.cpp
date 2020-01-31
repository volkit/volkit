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
    void TransformRange(
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
            vec3i(firstX, firstY, firstZ),
            vec3i(lastX, lastY, lastZ),
            unaryOp
            );
    }

    void TransformRange(
            StructuredVolume& volume,
            vec3i first,
            vec3i last,
            TransformUnaryOp unaryOp
            );

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktTransformSV1(vktStructuredVolume volume, vktTransformUnaryOp unaryOp)
{
    VKT_CALL__(
        TransformRange,
        volume->volume,
        vkt::vec3i(0, 0, 0),
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
        vkt::vec3i(0, 0, 0),
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
        vkt::vec3i(firstX, firstY, firstZ),
        vkt::vec3i(lastX, lastY, lastZ),
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
        vkt::vec3i(firstX, firstY, firstZ),
        vkt::vec3i(lastX, lastY, lastZ),
        (vktTransformBinaryOp)binaryOp
        );

    return VKT_NO_ERROR;
}
