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

VKTAPI vktError vktFlipSV(vktStructuredVolume dest,
                          vktStructuredVolume source,
                          vktAxis axis);

VKTAPI vktError vktFlipRangeSV(vktStructuredVolume dest,
                               vktStructuredVolume source,
                               vktAxis axis,
                               int32_t firstX,
                               int32_t firstY,
                               int32_t firstZ,
                               int32_t lastX,
                               int32_t lastY,
                               int32_t lastZ);

#ifdef __cplusplus
}
#endif
