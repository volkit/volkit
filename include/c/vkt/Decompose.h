// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include <vkt/Array3D.h>

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct vktStructuredVolume_impl;

typedef struct vktStructuredVolume_impl* vktStructuredVolume;

VKTAPI vktError vktBrickDecomposeSV(vktArray3D_StructuredVolume decomp,
                                    vktStructuredVolume source,
                                    int32_t brickSizeX,
                                    int32_t brickSizeY,
                                    int32_t brickSizeZ,
                                    int32_t haloSizeNegX,
                                    int32_t haloSizeNegY,
                                    int32_t haloSizeNegZ,
                                    int32_t haloSizePosX,
                                    int32_t haloSizePosY,
                                    int32_t haloSizePosZ);

#ifdef __cplusplus
}
#endif
