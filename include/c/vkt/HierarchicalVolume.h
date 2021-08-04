// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stddef.h>
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
    vktVec3i_t lower;
    vktVec3i_t dims;
    size_t offsetInBytes;
    unsigned level;
} vktBrick_t;

VKTAPI uint8_t vktHierarchicalVolumeGetMaxBytesPerVoxel();

VKTAPI void vktHierarchicalVolumeCreate(vktHierarchicalVolume* volume,
                                        vktBrick_t* bricks,
                                        size_t numBricks,
                                        vktDataFormat dataFormat,
                                        float mappingLo,
                                        float mappingHi);

// VKTAPI void vktHierarchicalVolumeCreateCopy(vktHierarchicalVolume* volume,
//                                             vktHierarchicalVolume rhs);

VKTAPI void vktHierarchicalVolumeDestroy(vktHierarchicalVolume volume);

VKTAPI vktVec3i_t vktHierarchicalVolumeGetDims3iv(vktHierarchicalVolume volume);

VKTAPI void vktHierarchicalVolumeGetDims3i(vktHierarchicalVolume volume,
                                           int32_t* dimX,
                                           int32_t* dimY,
                                           int32_t* dimZ);

VKTAPI size_t vktHierarchicalVolumeGetNumBricks(vktHierarchicalVolume volume);

VKTAPI vktBrick_t* vktHierarchicalVolumeGetBricks(vktHierarchicalVolume volume);

VKTAPI vktDataFormat vktHierarchicalVolumeGetDataFormat(vktHierarchicalVolume volume);

VKTAPI void vktHierarchicalVolumeSetVoxelMapping2f(vktHierarchicalVolume volume,
                                                   float lo,
                                                   float hi);

VKTAPI void vktHierarchicalVolumeGetVoxelMapping2f(vktHierarchicalVolume volume,
                                                   float* lo,
                                                   float* hi);

VKTAPI void vktHierarchicalVolumeSetVoxelMapping2fv(vktHierarchicalVolume volume,
                                                    vktVec2f_t mapping);

VKTAPI vktVec2f_t vktHierarchicalVolumeGetVoxelMapping2fv(vktHierarchicalVolume volume);

VKTAPI uint8_t* vktHierarchicalVolumeGetData(vktHierarchicalVolume volume);

#ifdef __cplusplus
}
#endif
