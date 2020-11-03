// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/Resample.hpp>

#include "macros.hpp"
#include "Resample_cuda.hpp"
#include "Resample_serial.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    VKTAPI Error Resample(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            )
    {
        VKT_CALL__(Resample, dst, src, filter);

        return NoError;
    }
}
