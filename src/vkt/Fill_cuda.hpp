// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>

#include "HierarchicalVolumeView.hpp"
#include "StructuredVolumeView.hpp"

namespace vkt
{
    void FillRange_cuda(StructuredVolumeView volume, Vec3i first, Vec3i last, float value);

    void FillRange_cuda(HierarchicalVolumeView volume, Vec3i first, Vec3i last, float value);

} // vkt
