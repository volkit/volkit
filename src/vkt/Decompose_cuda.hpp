// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/config.h>

#include <vkt/Array3D.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Array3D.h>
#include <vkt/linalg.h>
#include <vkt/StructuredVolume.h>

namespace vkt
{
    void BrickDecompose_cuda(
            Array3D<StructuredVolume>& decomp,
            StructuredVolume& volume,
            Vec3i brickSize,
            Vec3i haloSizeNeg,
            Vec3i haloSizePos
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif

    void BrickDecomposeC_cuda(
            vktArray3D_vktStructuredVolume dest,
            vktStructuredVolume source,
            vktVec3i_t brickSize,
            vktVec3i_t haloSizeNeg,
            vktVec3i_t haloSizePos
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
