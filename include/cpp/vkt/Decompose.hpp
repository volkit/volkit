// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include <vkt/Array3D.hpp>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    VKTAPI Error BrickDecompose(Array3D<StructuredVolume>& decomp,
                                StructuredVolume& source,
                                int32_t brickSizeX,
                                int32_t brickSizeY,
                                int32_t brickSizeZ,
                                int32_t haloSizeNegX,
                                int32_t haloSizeNegY,
                                int32_t haloSizeNegZ,
                                int32_t haloSizePosX,
                                int32_t haloSizePosY,
                                int32_t haloSizePosZ);

    VKTAPI Error BrickDecompose(Array3D<StructuredVolume>& decomp,
                                StructuredVolume& source,
                                Vec3i brickSize,
                                Vec3i haloSizeNeg,
                                Vec3i haloSizePos);

    VKTAPI Error BrickDecomposeGetNumBricks(int32_t& numBricksX,
                                            int32_t& numBricksY,
                                            int32_t& numBricksZ,
                                            int32_t dimX,
                                            int32_t dimY,
                                            int32_t dimZ,
                                            int32_t brickSizeX,
                                            int32_t brickSizeY,
                                            int32_t brickSizeZ,
                                            int32_t haloSizeNegX,
                                            int32_t haloSizeNegY,
                                            int32_t haloSizeNegZ,
                                            int32_t haloSizePosX,
                                            int32_t haloSizePosY,
                                            int32_t haloSizePosZ);

    VKTAPI Error BrickDecomposeGetNumBricks(Vec3i& numBricks,
                                            Vec3i dims,
                                            Vec3i brickSize,
                                            Vec3i haloSizeNeg,
                                            Vec3i haloSizePos);

} // vkt
