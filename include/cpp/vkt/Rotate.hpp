// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKTAPI Error Rotate(StructuredVolume& dest,
                        StructuredVolume& source,
                        Vec3f axis,
                        float angleInRadians,
                        Vec3f centerOfRotation);

} // vkt
