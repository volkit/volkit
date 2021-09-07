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
            FilterMode fm
            );

    void Resample_cuda(
            StructuredVolume& dst,
            HierarchicalVolume& src,
            FilterMode fm
            );

    void ResampleCLAHE_cuda(
            StructuredVolume& dst,
            StructuredVolume& src
            );
} // vkt
