#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void FillRange_cuda(StructuredVolume& volume, vec3i first, vec3i last, float value)
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
