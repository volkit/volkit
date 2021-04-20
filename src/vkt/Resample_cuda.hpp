// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void Resample_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            );

    void ResampleCLAHE_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            );
} // vkt
