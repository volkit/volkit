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

VKTAPI void vktLookupTableCreate(vktLookupTable* lut,
                                 int32_t dimX,
                                 int32_t dimY,
                                 int32_t dimZ,
                                 vktColorFormat format);

VKTAPI void vktLookupTableDestroy(vktLookupTable lut);

VKTAPI void vktLookupTableSetDims3i(vktLookupTable lut,
                                    int32_t dimX,
                                    int32_t dimY,
                                    int32_t dimZ);

VKTAPI void vktLookupTableGetDims3i(vktLookupTable lut,
                                    int32_t* dimX,
                                    int32_t* dimY,
                                    int32_t* dimZ);

VKTAPI void vktLookupTableSetDims3iv(vktLookupTable lut,
                                     vktVec3i_t dims);

VKTAPI vktVec3i_t vktLookupTableGetDims3iv(vktLookupTable lut);

VKTAPI void vktLookupTableSetColorFormat(vktLookupTable lut,
                                         vktColorFormat format);

VKTAPI vktColorFormat vktLookupTableGetColorFormat(vktLookupTable lut);

VKTAPI uint8_t* vktLookupTableGetData(vktLookupTable lut);

VKTAPI size_t vktLookupTableGetSizeInBytes(vktLookupTable lut);

VKTAPI vktResourceHandle vktLookupTableGetResourceHandle(vktLookupTable lut);

VKTAPI void vktLookupTableMigrate(vktLookupTable lut);

#ifdef __cplusplus
}
#endif
