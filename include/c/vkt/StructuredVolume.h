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

VKTAPI uint16_t vktStructuredVolumeGetMaxBytesPerVoxel();

VKTAPI void vktStructuredVolumeCreate(vktStructuredVolume* volume,
                                      int32_t dimx,
                                      int32_t dimy,
                                      int32_t dimz,
                                      uint16_t bytesPerVoxel,
                                      float mappingLo,
                                      float mappingHi);

VKTAPI void vktStructuredVolumeCreateCopy(vktStructuredVolume* volume,
                                          vktStructuredVolume rhs);

VKTAPI void vktStructuredVolumeDestroy(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetDims3i(vktStructuredVolume volume,
                                         int32_t dimx,
                                         int32_t dimy,
                                         int32_t dimz);

VKTAPI void vktStructuredVolumeGetDims3i(vktStructuredVolume volume,
                                         int32_t* dimx,
                                         int32_t* dimy,
                                         int32_t* dimz);

VKTAPI void vktStructuredVolumeSetDims3iv(vktStructuredVolume volume,
                                          vktVec3i_t dims);

VKTAPI vktVec3i_t vktStructuredVolumeGetDims3iv(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetBytesPerVoxel(vktStructuredVolume volume,
                                                uint16_t bpv);

VKTAPI uint16_t vktStructuredVolumeGetBytesPerVoxel(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeSetVoxelMapping2f(vktStructuredVolume volume,
                                                 float lo,
                                                 float hi);

VKTAPI void vktStructuredVolumeGetVoxelMapping2f(vktStructuredVolume volume,
                                                 float* lo,
                                                 float* hi);

VKTAPI void vktStructuredVolumeSetVoxelMapping2fv(vktStructuredVolume volume,
                                                  vktVec2f_t mapping);

VKTAPI vktVec2f_t vktStructuredVolumeGetVoxelMapping2fv(vktStructuredVolume volume);

VKTAPI uint8_t* vktStructuredVolumeGetData(vktStructuredVolume volume);

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

VKTAPI void vktStructuredVolumeSetVoxel(vktStructuredVolume volume,
                                        int32_t x,
                                        int32_t y,
                                        int32_t z,
                                        uint8_t const* data);

VKTAPI void vktStructuredVolumeGetVoxel(vktStructuredVolume volume,
                                        int32_t x,
                                        int32_t y,
                                        int32_t z,
                                        uint8_t* data);

VKTAPI void vktStructuredVolumeMapVoxel(vktStructuredVolume volume,
                                        uint8_t* dst,
                                        float src);

VKTAPI void vktStructuredVolumeUnmapVoxel(vktStructuredVolume volume,
                                          float* dst,
                                          uint8_t const* src);

VKTAPI size_t vktStructuredVolumeGetSizeInBytes(vktStructuredVolume volume);

VKTAPI void vktStructuredVolumeMigrate(vktStructuredVolume);

#ifdef __cplusplus
}
#endif
