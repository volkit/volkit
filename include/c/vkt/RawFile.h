// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

VKTAPI void vktRawFileCreateS(vktRawFile* file,
                              char const* fileName,
                              char const* mode);

VKTAPI void vktRawFileCreateFD(vktRawFile* file,
                               FILE* fd);

VKTAPI void vktRawFileRead(vktRawFile file,
                           char* buf,
                           size_t len);

VKTAPI vktBool vktRawFileGood(vktRawFile file);

#ifdef __cplusplus
}
#endif
