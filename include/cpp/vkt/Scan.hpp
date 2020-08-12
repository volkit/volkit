// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

/*! @file  Scan.hpp
 * @brief  Compute 3D prefix sum volume
 *
 * Range-based prefix sum operations, summation will start at the corner
 * specified by `first` and advance up until `last`. Note that it is allowable
 * that, e.g., `first.x > last.x`; in that case, the prefix sum will be
 * computed from right to left (whereas `first.x < last.x` implies summation
 * from left to right).
 */
namespace vkt
{
    //! Compute 3D prefix sum of whole volume
    VKTAPI Error Scan(StructuredVolume& dst,
                      StructuredVolume& src);

    //! Compute 3D prefix sum over the range `[firstXXX..lastXXX)`
    VKTAPI Error ScanRange(StructuredVolume& dst,
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

    //! Compute 3D prefix sum over the range `[first..last)`
    VKTAPI Error ScanRange(StructuredVolume& dst,
                           StructuredVolume& src,
                           Vec3i first,
                           Vec3i last,
                           Vec3i dstOffset = { 0, 0, 0 });

} // vkt
