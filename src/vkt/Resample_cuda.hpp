// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/HierarchicalVolume.hpp>
#include <vkt/Resample.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void Resample_cuda(
            StructuredVolume& dst,
            StructuredVolume& src,
            Filter filter
            );

    void Resample_cuda(
            StructuredVolume& dst,
            HierarchicalVolume& src,
            Filter filter
            );

} // vkt
