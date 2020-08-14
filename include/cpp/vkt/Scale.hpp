// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKTAPI Error Scale(StructuredVolume& dest,
                       StructuredVolume& source,
                       Vec3f scalingFactor,
                       Vec3f centerOfScaling);

} // vkt
