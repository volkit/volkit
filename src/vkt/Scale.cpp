// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Scale.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Scale.h>
#include <vkt/StructuredVolume.h>

#include "macros.hpp"
#include "Scale_cuda.hpp"
#include "Scale_serial.hpp"
#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    Error Scale(
        StructuredVolume& dest,
        StructuredVolume& source,
        Vec3f scalingFactor,
        Vec3f centerOfScaling
        )
    {
        VKT_CALL__(
            ScaleRange,
            dest,
            source,
            { 0, 0, 0 },
            source.getDims(),
            scalingFactor,
            centerOfScaling
            );

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktScaleSV(
        vktStructuredVolume dest,
        vktStructuredVolume source,
        vktVec3f_t scalingFactor,
        vktVec3f_t centerOfScaling
        )
{
    VKT_CALL__(
        ScaleRange,
        dest->volume,
        source->volume,
        { 0, 0, 0 },
        source->volume.getDims(),
        { scalingFactor.x, scalingFactor.y, scalingFactor.z },
        { centerOfScaling.x, centerOfScaling.y, centerOfScaling.z }
        );

    return vktNoError;
}
