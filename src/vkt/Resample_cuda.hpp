// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void Resample_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            )
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif

} // vkt
