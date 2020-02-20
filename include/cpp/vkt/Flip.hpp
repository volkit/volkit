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
                           Axis axis,
                           int32_t firstX,
                           int32_t firstY,
                           int32_t firstZ,
                           int32_t lastX,
                           int32_t lastY,
                           int32_t lastZ,
                           int32_t dstOffsetX = 0,
                           int32_t dstOffsetY = 0,
                           int32_t dstOffsetZ = 0);

    VKTAPI Error FlipRange(StructuredVolume& dest,
                           StructuredVolume& source,
                           Axis axis,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

} // vkt
