// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include <vkt/Voxel.h>

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef void (*vktTransformUnaryOp)(int32_t x,
                                    int32_t y,
                                    int32_t z,
                                    vktVoxelView_t voxel);

typedef void (*vktTransformBinaryOp)(int32_t x1,
                                     int32_t y1,
                                     int32_t z1,
                                     vktVoxelView_t voxel1,
                                     vktVoxelView_t voxel2);

VKTAPI vktError vktTransformSV1(vktStructuredVolume volume,
                                vktTransformUnaryOp unaryOp);

VKTAPI vktError vktTransformSV2(vktStructuredVolume volume1,
                                vktStructuredVolume volume2,
                                vktTransformBinaryOp binaryOp);

VKTAPI vktError vktTransformRangeSV1(vktStructuredVolume volume,
                                     int32_t firstX,
                                     int32_t firstY,
                                     int32_t firstZ,
                                     int32_t lastX,
                                     int32_t lastY,
                                     int32_t lastZ,
                                     vktTransformUnaryOp unaryOp);

VKTAPI vktError vktTransformRangeSV2(vktStructuredVolume volume1,
                                     vktStructuredVolume volume2,
                                     int32_t firstX,
                                     int32_t firstY,
                                     int32_t firstZ,
                                     int32_t lastX,
                                     int32_t lastY,
                                     int32_t lastZ,
                                     int32_t volume2OffsetX,
                                     int32_t volume2OffsetY,
                                     int32_t volume2OffsetZ,
                                     vktTransformBinaryOp binaryOp);

VKTAPI vktError vktTransformSubVoxelRangeSV1(vktStructuredVolume volume,
                                             float firstX,
                                             float firstY,
                                             float firstZ,
                                             float lastX,
                                             float lastY,
                                             float lastZ,
                                             vktTransformUnaryOp unaryOp);

VKTAPI vktError vktTransformSubVoxelRangeSV2(vktStructuredVolume volume1,
                                             vktStructuredVolume volume2,
                                             float firstX,
                                             float firstY,
                                             float firstZ,
                                             float lastX,
                                             float lastY,
                                             float lastZ,
                                             float volume2OffsetX,
                                             float volume2OffsetY,
                                             float volume2OffsetZ,
                                             vktTransformBinaryOp binaryOp);

#ifdef __cplusplus
}
#endif
