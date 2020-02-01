// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
} vktRenderState_t;

VKTAPI vktError vktRenderSV(vktStructuredVolume volume,
                            vktRenderState_t renderState,
                            vktRenderState_t* newRenderState);

#ifdef __cplusplus
}
#endif
