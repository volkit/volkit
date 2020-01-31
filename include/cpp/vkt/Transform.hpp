#pragma once

#include <cstdint>

#include "forward.hpp"
#include "linalg.hpp"

namespace vkt
{
    typedef void (*TransformUnaryOp)(int32_t x,
                                     int32_t y,
                                     int32_t z,
                                     uint8_t* voxel);

    typedef void (*TransformBinaryOp)(int32_t x1,
                                      int32_t y1,
                                      int32_t z1,
                                      uint8_t* voxel1,
                                      uint8_t* voxel2);

    void Transform(StructuredVolume& volume,
                   TransformUnaryOp unaryOp);

    void Transform(StructuredVolume& volume1,
                   StructuredVolume& volume2,
                   TransformBinaryOp binaryOp);

    void TransformRange(StructuredVolume& volume,
                        int32_t firstX,
                        int32_t firstY,
                        int32_t firstZ,
                        int32_t lastX,
                        int32_t lastY,
                        int32_t lastZ,
                        TransformUnaryOp unaryOp);

    void TransformRange(StructuredVolume& volume,
                        vec3i first,
                        vec3i last,
                        TransformUnaryOp unaryOp);

    void TransformRange(StructuredVolume& volume1,
                        StructuredVolume& volume2,
                        int32_t firstX,
                        int32_t firstY,
                        int32_t firstZ,
                        int32_t lastX,
                        int32_t lastY,
                        int32_t lastZ,
                        TransformBinaryOp binaryOp);

    void TransformRange(StructuredVolume& volume1,
                        StructuredVolume& volume2,
                        vec3i first,
                        vec3i last,
                        TransformBinaryOp binaryOp);
} // vkt
