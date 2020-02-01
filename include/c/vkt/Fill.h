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

VKTAPI vktError vktFillSV(vktStructuredVolume volume,
                          float value);

VKTAPI vktError vktFillRangeSV(vktStructuredVolume volume,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ,
                               float value);

VKTAPI vktError vktFillSubVoxelRangeSV(vktStructuredVolume volume,
                                       float firstX,
                                       float firstY,
                                       float firstZ,
                                       float lastX,
                                       float lastY,
                                       float lastZ,
                                       float value);

#ifdef __cplusplus
}
#endif
