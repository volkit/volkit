// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
#include "forward.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    vktBool_t isStructured;
    vktVec3i_t dims;
    vktDataFormat dataFormat;
} vktVolumeFileHeader_t;

static void vktVolumeFileHeaderDefaultInit(vktVolumeFileHeader_t* header)
{
    memset(header, 0, sizeof(*header));
}

VKTAPI void vktVolumeFileCreateS(vktVolumeFile* file,
                                 char const* fileName,
                                 vktOpenMode om);

VKTAPI vktDataSource vktVolumeFileGetBase(vktVolumeFile file);

VKTAPI void vktVolumeFileDestroy(vktVolumeFile file);

VKTAPI size_t vktVolumeFileRead(vktVolumeFile file,
                                char* buf,
                                size_t len);

VKTAPI vktBool_t vktVolumeFileGood(vktVolumeFile file);

VKTAPI vktVolumeFileHeader_t vktVolumeFileGetHeader(vktVolumeFile file);

#ifdef __cplusplus
}
#endif
