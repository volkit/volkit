// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdint>

#include <vkt/Voxel.hpp>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    typedef void (*TransformUnaryOp)(int32_t x,
                                     int32_t y,
                                     int32_t z,
                                     VoxelView voxel);

    typedef void (*TransformBinaryOp)(int32_t x1,
                                      int32_t y1,
                                      int32_t z1,
                                      VoxelView voxel1,
                                      VoxelView voxel2);

    VKTAPI Error Transform(StructuredVolume& volume,
                           TransformUnaryOp unaryOp);

    VKTAPI Error Transform(StructuredVolume& volume1,
                           StructuredVolume& volume2,
                           TransformBinaryOp binaryOp);

    VKTAPI Error TransformRange(StructuredVolume& volume,
                                int32_t firstX,
                                int32_t firstY,
                                int32_t firstZ,
                                int32_t lastX,
                                int32_t lastY,
                                int32_t lastZ,
                                TransformUnaryOp unaryOp);

    VKTAPI Error TransformRange(StructuredVolume& volume,
                                Vec3i first,
                                Vec3i last,
                                TransformUnaryOp unaryOp);

    VKTAPI Error TransformRange(StructuredVolume& volume1,
                                StructuredVolume& volume2,
                                int32_t firstX,
                                int32_t firstY,
                                int32_t firstZ,
                                int32_t lastX,
                                int32_t lastY,
                                int32_t lastZ,
                                TransformBinaryOp binaryOp);

    VKTAPI Error TransformRange(StructuredVolume& volume1,
                                StructuredVolume& volume2,
                                Vec3i first,
                                Vec3i last,
                                TransformBinaryOp binaryOp);
} // vkt
