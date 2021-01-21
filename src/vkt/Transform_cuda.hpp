// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    // Unary op
    void TransformRange_cuda(
            StructuredVolume& volume,
            Vec3i first,
            Vec3i last,
            TransformUnaryOp unaryOp
            )
    {
    }

    // Binary op
    void TransformRange_cuda(
            StructuredVolume& volume1,
            StructuredVolume& volume2,
            Vec3i first,
            Vec3i last,
            TransformBinaryOp binaryOp
            )
    {
    }

} // vkt
