// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vkt/ManagedResource.h>

#include "common.h"
#include "forward.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI uint16_t vktStructuredVolumeGetMaxBytesPerVoxel();

VKTAPI void vktStructuredVolumeCreate(vktStructuredVolume* volume,
                                      int32_t dimX,
                                      int32_t dimY,
                                      int32_t dimZ,
                                      uint16_t bytesPerVoxel,
                                      float distX,
                                      float distY,
                                      float distZ,
                                      float mappingLo,
                                      float mappingHi);

VKTAPI void vktStructuredVolumeCreateCopy(vktStructuredVolume* volume,
                                          vktStructuredVolume rhs);

VKTAPI void vktStructuredVolumeDestroy(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetDims3i(vktStructuredVolume volume,
                                         int32_t dimX,
                                         int32_t dimY,
                                         int32_t dimZ);

VKTAPI void vktStructuredVolumeGetDims3i(vktStructuredVolume volume,
                                         int32_t* dimX,
                                         int32_t* dimY,
                                         int32_t* dimZ);

VKTAPI void vktStructuredVolumeSetDims3iv(vktStructuredVolume volume,
                                          vktVec3i_t dims);

VKTAPI vktVec3i_t vktStructuredVolumeGetDims3iv(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetBytesPerVoxel(vktStructuredVolume volume,
                                                uint16_t bpv);

VKTAPI uint16_t vktStructuredVolumeGetBytesPerVoxel(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetDist3f(vktStructuredVolume volume,
                                         float distX,
                                         float distY,
                                         float distZ);

VKTAPI void vktStructuredVolumeGetDist3f(vktStructuredVolume volume,
                                         float* distX,
                                         float* distY,
                                         float* distZ);

VKTAPI void vktStructuredVolumeSetDist3fv(vktStructuredVolume volume,
                                          vktVec3f_t dist);

VKTAPI vktVec3f_t vktStructuredVolumeGetDist3fv(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetVoxelMapping2f(vktStructuredVolume volume,
                                                 float lo,
                                                 float hi);

VKTAPI void vktStructuredVolumeGetVoxelMapping2f(vktStructuredVolume volume,
                                                 float* lo,
                                                 float* hi);

VKTAPI void vktStructuredVolumeSetVoxelMapping2fv(vktStructuredVolume volume,
                                                  vktVec2f_t mapping);

VKTAPI vktVec2f_t vktStructuredVolumeGetVoxelMapping2fv(vktStructuredVolume volume);

VKTAPI vktBox3f_t vktStructuredVolumeGetWorldBounds(vktStructuredVolume volume);

VKTAPI uint8_t* vktStructuredVolumeGetData(vktStructuredVolume volume);

VKTAPI float vtkStructuredVolumeSampleLinear(int32_t x, int32_t y, int32_t z);

VKTAPI void vktStructuredVolumeSetValue(vktStructuredVolume volume,
                                        int32_t x,
                                        int32_t y,
                                        int32_t z,
                                        float value);

VKTAPI void vktStructuredVolumeGetValue(vktStructuredVolume volume,
                                        int32_t x,
                                        int32_t y,
                                        int32_t z,
                                        float* value);

VKTAPI void vktStructuredVolumeSetBytes(vktStructuredVolume volume,
                                        int32_t x,
                                        int32_t y,
                                        int32_t z,
                                        uint8_t const* data);

VKTAPI void vktStructuredVolumeGetBytes(vktStructuredVolume volume,
                                        int32_t x,
                                        int32_t y,
                                        int32_t z,
                                        uint8_t* data);

VKTAPI size_t vktStructuredVolumeGetSizeInBytes(vktStructuredVolume volume);

VKTAPI vktResourceHandle vktStructuredVolumeGetResourceHandle(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeMigrate(vktStructuredVolume volume);

#ifdef __cplusplus
}
#endif
