// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKTAPI Error Copy(StructuredVolume& dst,
                      StructuredVolume& src);

    VKTAPI Error CopyRange(StructuredVolume& dst,
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

    VKTAPI Error CopyRange(StructuredVolume& dst,
                           StructuredVolume& src,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = Vec3i(0));
} // vkt
