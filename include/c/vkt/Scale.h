// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.h"
#include "forward.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI vktError vktScaleSV(vktStructuredVolume dest,
                           vktStructuredVolume source,
                           vktVec3f_t scalingFactor,
                           vktVec3f_t centerOfScaling);

#ifdef __cplusplus
}
#endif
