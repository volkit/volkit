// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <vkt/Resample.hpp>

#include "macros.hpp"
#include "Resample_serial.hpp"

#if VKT_HAVE_CUDA
#include "Resample_cuda.hpp"
#endif

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    VKTAPI Error Resample(
            StructuredVolume& dst,
            StructuredVolume& src,
            FilterMode fm
            )
    {
        VKT_CALL__(Resample, dst, src, fm);

        return NoError;
    }

    VKTAPI Error Resample(
            StructuredVolume& dst,
            HierarchicalVolume& src,
            FilterMode fm
            )
    {
        VKT_CALL__(Resample, dst, src, fm);

        return NoError;
    }

    VKTAPI Error ResampleCLAHE(
            StructuredVolume& dst,
            StructuredVolume& src
            )
    {
        VKT_CALL__(ResampleCLAHE, dst, src);
    }
}
