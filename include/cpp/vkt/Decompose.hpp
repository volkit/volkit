#pragma once

#include <cstdint>

#include <vkt/Array3D.hpp>

#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    void BrickDecompose(Array3D<StructuredVolume>& decomp,
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

    void BrickDecompose(Array3D<StructuredVolume>& decomp,
                        StructuredVolume& source,
                        vec3i brickSize,
                        vec3i haloSizeNeg,
                        vec3i haloSizePos);

    void BrickDecomposeGetNumBricks(int32_t& numBricksX,
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

    void BrickDecomposeGetNumBricks(vec3i& numBricks,
                                    vec3i dims,
                                    vec3i brickSize,
                                    vec3i haloSizeNeg,
                                    vec3i haloSizePos);

} // vkt
