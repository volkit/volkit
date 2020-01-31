#pragma once

#include <cstdint>

#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    void Flip(StructuredVolume& volume,
              Axis axis);

    void FlipRange(StructuredVolume& volume,
                   int32_t firstX,
                   int32_t firstY,
                   int32_t firstZ,
                   int32_t lastX,
                   int32_t lastY,
                   int32_t lastZ,
                   Axis axis);

    void FlipRange(StructuredVolume& volume,
                   vec3i first,
                   vec3i last,
                   Axis axis);

} // vkt
