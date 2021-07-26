// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.hpp"
#include "forward.hpp"

/*! @file  Resample.hpp
 * @brief  Resample volumes in space and regarding their value ranges
 */
namespace vkt
{
    enum class Filter
    {
        Nearest,
        Linear,
    };

    VKTAPI Error Resample(StructuredVolume& dst,
                          StructuredVolume& src,
                          Filter filter);

    VKTAPI Error Resample(StructuredVolume& dst,
                          HierarchicalVolume& src,
                          Filter filter);
} // vkt
