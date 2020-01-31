#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void FlipRange_cuda(StructuredVolume& volume, vec3i first, vec3i last, Axis axis)
#if 0//VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
