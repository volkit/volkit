// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

/*! @file  Fill.hpp
 * @brief  Fill algorithm that fills a range of the volume with a constant value
 */
namespace vkt
{
    //! Fill the whole volume with a constant value
    VKTAPI Error Fill(StructuredVolume& volume,
                      float value);

    //! Fill the range `[firstXXX..lastXXX)` with a constant value
    VKTAPI Error FillRange(StructuredVolume& volume,
                           int32_t firstX,
                           int32_t firstY,
                           int32_t firstZ,
                           int32_t lastX,
                           int32_t lastY,
                           int32_t lastZ,
                           float value);

    //! Fill the range `[first..last)` with a constant value
    VKTAPI Error FillRange(StructuredVolume& volume,
                           Vec3i first,
                           Vec3i last,
                           float value);

} // vkt
