// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKTAPI Error Flip(StructuredVolume& dest,
                      StructuredVolume& source,
                      Axis axis);

    VKTAPI Error FlipRange(StructuredVolume& dest,
                           StructuredVolume& source,
                           int32_t firstX,
                           int32_t firstY,
                           int32_t firstZ,
                           int32_t lastX,
                           int32_t lastY,
                           int32_t lastZ,
                           Axis axis);

    VKTAPI Error FlipRange(StructuredVolume& dest,
                           StructuredVolume& source,
                           Vec3i first,
                           Vec3i last,
                           Axis axis);

} // vkt
