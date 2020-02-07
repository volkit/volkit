// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Rotate.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Rotate.h>
#include <vkt/StructuredVolume.h>

#include "macros.hpp"
#include "Rotate_cuda.hpp"
#include "Rotate_serial.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Rotate(
        StructuredVolume& dest,
        StructuredVolume& source,
        Vec3f axis,
        float angleInRadians,
        Vec3f centerOfRotation
        )
    {
        VKT_CALL__(
            RotateRange,
            dest,
            source,
            { 0, 0, 0 },
            source.getDims(),
            axis,
            angleInRadians,
            centerOfRotation
            );

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktRotateSV(
        vktStructuredVolume dest,
        vktStructuredVolume source,
        vktVec3f_t axis,
        float angleInRadians,
        vktVec3f_t centerOfRotation
        )
{
    VKT_CALL__(
        RotateRange,
        dest->volume,
        source->volume,
        { 0, 0, 0 },
        source->volume.getDims(),
        { axis.x, axis.y, axis.z },
        angleInRadians,
        { centerOfRotation.x, centerOfRotation.y, centerOfRotation.z }
        );

    return vktNoError;
}
