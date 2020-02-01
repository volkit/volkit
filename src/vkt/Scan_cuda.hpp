#pragma once

#include <vkt/config.h>

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void ScanRange_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            Vec3i first,
            Vec3i last,
            Vec3i dstOffset
            )
    {
    }

} // vkt
