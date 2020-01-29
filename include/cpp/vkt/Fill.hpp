#pragma once

#include <cstdint>

#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    void Fill(StructuredVolume& volume,
              float value);

    void FillRange(StructuredVolume& volume,
                   int32_t firstX,
                   int32_t firstY,
                   int32_t firstZ,
                   int32_t lastX,
                   int32_t lastY,
                   int32_t lastZ,
                   float value);

    void FillRange(StructuredVolume& volume,
                   vec3i first,
                   vec3i last,
                   float value);

} // vkt
