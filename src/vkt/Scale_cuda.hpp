// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/Scale.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void ScaleRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Vec3f scalingFactor,
            Vec3f centerOfScaling
            )
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
