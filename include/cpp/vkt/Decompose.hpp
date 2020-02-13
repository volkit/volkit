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
    VKTAPI Error BrickDecompose(Array3D<StructuredVolume>& dest,
                                StructuredVolume& source,
                                int32_t brickSizeX,
                                int32_t brickSizeY,
                                int32_t brickSizeZ,
                                int32_t haloSizeNegX = 0,
                                int32_t haloSizeNegY = 0,
                                int32_t haloSizeNegZ = 0,
                                int32_t haloSizePosX = 0,
                                int32_t haloSizePosY = 0,
                                int32_t haloSizePosZ = 0);

    VKTAPI Error BrickDecompose(Array3D<StructuredVolume>& dest,
                                StructuredVolume& source,
                                Vec3i brickSize,
                                Vec3i haloSizeNeg = { 0, 0, 0 },
                                Vec3i haloSizePos = { 0, 0, 0 });

    VKTAPI Error BrickDecomposeResize(Array3D<StructuredVolume>& dest,
                                      StructuredVolume& source,
                                      int32_t brickSizeX,
                                      int32_t brickSizeY,
                                      int32_t brickSizeZ,
                                      int32_t haloSizeNegX = 0,
                                      int32_t haloSizeNegY = 0,
                                      int32_t haloSizeNegZ = 0,
                                      int32_t haloSizePosX = 0,
                                      int32_t haloSizePosY = 0,
                                      int32_t haloSizePosZ = 0);

    VKTAPI Error BrickDecomposeResize(Array3D<StructuredVolume>& dest,
                                      StructuredVolume& source,
                                      Vec3i brickSize,
                                      Vec3i haloSizeNeg = { 0, 0, 0 },
                                      Vec3i haloSizePos = { 0, 0, 0 });

} // vkt
