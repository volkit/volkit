// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*!
 * @brief  Voxel view type to pass voxel data around
 */
typedef struct
{
    uint8_t* bytes;
    uint16_t bytesPerVoxel;
    float mappingLo;
    float mappingHi;
} vktVoxelView_t;

/*!
 * @brief  provide in-memory representation of mapped voxel
 */
VKTAPI vktError vktMapVoxel(uint8_t* dst,
                            float value,
                            uint16_t bytesPerVoxel,
                            float mappingLo,
                            float mappingHi);

/*!
 * @brief  convert from mapped in-memory representation to voxel value
 */
VKTAPI vktError vktUnmapVoxel(float* value,
                              uint8_t const* src,
                              uint16_t bytesPerVoxel,
                              float mappingLo,
                              float mappingHi);

#ifdef __cplusplus
}
#endif
