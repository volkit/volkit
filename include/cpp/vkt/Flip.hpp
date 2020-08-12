// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

/*! @file Flip.hpp
 * @brief  Flip volume along a cartesian axis
 *
 * Algorithm that flips the volume or a user-specified range along a specified
 * axis and stores the result in a destination volume. It is allowable to
 * perform this operation in-place by letting the `dest` and `source`
 * references point to the same volume instance. However, when a destination
 * offset `dstOffset != 0` is specified, the behavior of the operation is
 * undefined. Specifically, with a parallel execution policy, it is not
 * guaranteed that the operations are performed atomically.
 */
namespace vkt
{
    //! Flip whole volume along a cartesian axis
    VKTAPI Error Flip(StructuredVolume& dest,
                      StructuredVolume& source,
                      Axis axis);

    //! Flip the range `[firstXXX..lastXXX)` along a cartesian axis
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

    //! Flip the range `[first..last)` along a cartesian axis
    VKTAPI Error FlipRange(StructuredVolume& dest,
                           StructuredVolume& source,
                           Axis axis,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

} // vkt
