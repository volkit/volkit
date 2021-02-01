// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "common.h"
#include "forward.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI void vktRawFileCreateS(vktRawFile* file,
                              char const* fileName,
                              char const* mode);

VKTAPI void vktRawFileCreateFD(vktRawFile* file,
                               FILE* fd);

VKTAPI vktDataSource vktRawFileGetBase(vktRawFile file);

VKTAPI void vktRawFileDestroy(vktRawFile file);

VKTAPI size_t vktRawFileRead(vktRawFile file,
                             char* buf,
                             size_t len);

VKTAPI vktBool_t vktRawFileGood(vktRawFile file);

VKTAPI vktVec3i_t vktRawFileGetDims3iv(vktRawFile file);

VKTAPI vktDataFormat vktRawFileGetDataFormat(vktRawFile file);

#ifdef __cplusplus
}
#endif
