// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <vkt/Resample.hpp>

#include "Callable.hpp"
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
    Error Resample(
            StructuredVolume& dst,
            StructuredVolume& src,
            FilterMode fm
            )
    {
        VKT_LEGACY_CALL__(Resample, dst, src, fm);

        return NoError;
    }

    Error Resample(
            StructuredVolume& dst,
            HierarchicalVolume& src,
            FilterMode fm
            )
    {
        VKT_LEGACY_CALL__(Resample, dst, src, fm);

        return NoError;
    }

    Error ResampleCLAHE(
            StructuredVolume& dst,
            StructuredVolume& src
            )
    {
        VKT_LEGACY_CALL__(ResampleCLAHE, dst, src);
    }
}
