// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/Rotate.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void RotateRange_cuda(
            StructuredVolume& dest,
            StructuredVolume& source,
            Vec3i first,
            Vec3i last,
            Vec3f axis,
            float angleInRadians,
            Vec3f centerOfRotation
            )
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
