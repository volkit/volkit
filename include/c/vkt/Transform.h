#pragma once

#include <stdint.h>

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef float (*vktTransformUnaryOp)(float);
typedef float (*vktTransformBinaryOp)(float, float);

VKTAPI vktError vktTransformSV1(vktStructuredVolume volume1,
                                vktTransformUnaryOp unaryOp);

VKTAPI vktError vktTransformSV2(vktStructuredVolume volume1,
                                vktStructuredVolume volume2,
                                vktTransformBinaryOp binaryOp);

VKTAPI vktError vktTransformRangeSV1(vktStructuredVolume volume1,
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

VKTAPI vktError vktTransformSubVoxelRangeSV1(vktStructuredVolume volume1,
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
