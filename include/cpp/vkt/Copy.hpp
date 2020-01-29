#pragma once

#include <cstdint>

#include "linalg.hpp"

namespace vkt
{
    class StructuredVolume;

    void CopyRange(StructuredVolume& dst,
                   StructuredVolume& src,
                   int32_t firstX,
                   int32_t firstY,
                   int32_t firstZ,
                   int32_t lastX,
                   int32_t lastY,
                   int32_t lastZ,
                   int32_t dstOffsetX = 0,
                   int32_t dstOffsetY = 0,
                   int32_t dstOffsetZ = 0);

    void CopyRange(StructuredVolume& dst,
                   StructuredVolume& src,
                   vec3i first,
                   vec3i last,
                   vec3i dstOffset = vec3i(0));
} // vkt
