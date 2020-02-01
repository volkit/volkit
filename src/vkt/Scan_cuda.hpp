#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void ScanRange_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            vec3i first,
            vec3i last,
            vec3i dstOffset
            )
    {
    }

} // vkt
