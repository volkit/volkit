// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    float x;
    float y;
} vktVec2f_t;

typedef struct
{
    float x;
    float y;
    float z;
} vktVec3f_t;

typedef struct
{
    int x;
    int y;
    int z;
} vktVec3i_t;

typedef struct
{
    vktVec3i_t min;
    vktVec3i_t max;
} vktBox3f_t;

typedef enum
{
    vktAxisX,
    vktAxisY,
    vktAxisZ,
} vktAxis;

#ifdef __cplusplus
}
#endif
