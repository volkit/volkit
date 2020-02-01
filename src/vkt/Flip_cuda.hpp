// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void FlipRange_cuda(StructuredVolume& volume, Vec3i first, Vec3i last, Axis axis)
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
