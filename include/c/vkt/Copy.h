#pragma once

#include <stdint.h>

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI vktError vktCopySV(vktStructuredVolume dst,
                          vktStructuredVolume src);

VKTAPI vktError vktCopyRangeSV(vktStructuredVolume dst,
                               vktStructuredVolume src,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               int32_t dstOffsetX,
                               int32_t dstOffsetY,
                               int32_t dstOffsetZ);

VKTAPI vktError vktCopySubVoxelRangeSV(vktStructuredVolume dst,
                                       vktStructuredVolume src,
                                       float firstX,
                                       float firstY,
                                       float firstZ,
                                       float lastX,
                                       float lastY,
                                       float lastZ,
                                       float dstOffsetX,
                                       float dstOffsetY,
                                       float dstOffsetZ);

#ifdef __cplusplus
}
#endif
