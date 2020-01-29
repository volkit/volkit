#pragma once

#include <vkt/config.h>

#include <vkt/Array3D.hpp>
#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

namespace vkt
{
    void BrickDecompose_cuda(
            Array3D<StructuredVolume>& decomp,
            StructuredVolume& volume,
            vec3i brickSize,
            vec3i haloSizeNeg,
            vec3i haloSizePos
            )
#if VKT_HAVE_CUDA
    ;
#else
    {}
#endif
} // vkt
