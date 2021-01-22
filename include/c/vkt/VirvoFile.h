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

VKTAPI void vktVirvoFileCreateS(vktVirvoFile* file,
                                char const* fileName);

VKTAPI vktDataSource vktVirvoFileGetBase(vktVirvoFile file);

VKTAPI void vktVirvoFileDestroy(vktVirvoFile file);

VKTAPI size_t vktVirvoFileRead(vktVirvoFile file,
                               char* buf,
                               size_t len);

VKTAPI vktBool_t vktVirvoFileGood(vktVirvoFile file);

VKTAPI vktVec3i_t vktVirvoFileGetDims3iv(vktVirvoFile file);

VKTAPI vktDataFormat vktVirvoFileGetDataFormat(vktVirvoFile file);

#ifdef __cplusplus
}
#endif
