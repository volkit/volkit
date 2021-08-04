// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.hpp"
#include "forward.hpp"

/*! @file  Resample.hpp
 * @brief  Resample volumes in space and regarding their data format
 */
namespace vkt
{
    enum class Filter
    {
        Nearest,
        Linear,
    };

    //! Resample in space and/or regarding data format
    VKTAPI Error Resample(StructuredVolume& dst,
                          StructuredVolume& src,
                          Filter filter);

    VKTAPI Error Resample(StructuredVolume& dst,
                          HierarchicalVolume& src,
                          Filter filter);

    //! Resample using contrast limited adaptive histogram equalization
    VKTAPI Error ResampleCLAHE(StructuredVolume& dst,
                               StructuredVolume& src);
} // vkt
