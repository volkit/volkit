// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Decompose_cuda.hpp"

namespace vkt
{
    void BrickDecompose_cuda(
            Array3D<StructuredVolume>& decomp,
            StructuredVolume& volume,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
    {
    }

    void BrickDecomposeC_cuda(
            vktArray3D_vktStructuredVolume dest,
            vktStructuredVolume source,
            vktVec3i_t brickSize,
            vktVec3i_t haloSizeNeg,
            vktVec3i_t haloSizePos
            )
    {
    }
} // vkt
