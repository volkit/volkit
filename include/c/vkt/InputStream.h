// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI void vktInputStreamCreate(vktInputStream* stream,
                                 vktDataSource source);

VKTAPI void vktInputStreamDestroy(vktInputStream stream);

VKTAPI vktError vktInputStreamReadSV(vktInputStream stream,
                                     vktStructuredVolume volume);

VKTAPI vktError vktInputStreamReadRangeSV(vktInputStream stream,
                                          vktStructuredVolume volume,
                                          int32_t firstX,
                                          int32_t firstY,
                                          int32_t firstZ,
                                          int32_t lastX,
                                          int32_t lastY,
                                          int32_t lastZ,
                                          int32_t dstOffsetX,
                                          int32_t dstOffsetY,
                                          int32_t dstOffsetZ);

#ifdef __cplusplus
}
#endif
