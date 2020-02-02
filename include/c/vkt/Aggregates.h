// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include "common.h"
#include "forward.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    float min;
    float max;
    float mean;
    float median;
    float std;
    float var;
    float sum;
    float prod;
    vktVec3i_t argmin;
    vktVec3i_t argmax;
} vktAggregates_t;

VKTAPI vktError vktComputeAggregatesSV(vktStructuredVolume volume,
                                        vktAggregates_t* aggregates);

VKTAPI vktError vktComputeAggregatesRangeSV(vktStructuredVolume volume,
                                            vktAggregates_t* aggregates,
                                            int32_t firstX,
                                            int32_t firstY,
                                            int32_t firstZ,
                                            int32_t lastX,
                                            int32_t lastY,
                                            int32_t lastZ);

#ifdef __cplusplus
}
#endif
