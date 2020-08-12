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
    float x;
    float y;
    float z;
    float w;
} vktVec4f_t;

typedef struct
{
    int x;
    int y;
} vktVec2i_t;

typedef struct
{
    int x;
    int y;
    int z;
} vktVec3i_t;

typedef struct
{
    int x;
    int y;
    int z;
    int w;
} vktVec4i_t;

typedef struct
{
    vktVec2f_t min;
    vktVec2f_t max;
} vktBox2f_t;

typedef struct
{
    vktVec3f_t min;
    vktVec3f_t max;
} vktBox3f_t;

typedef struct
{
    vktVec2i_t min;
    vktVec2i_t max;
} vktBox2i_t;

typedef struct
{
    vktVec3i_t min;
    vktVec3i_t max;
} vktBox3i_t;

typedef struct
{
    vktVec3f_t col0;
    vktVec3f_t col1;
    vktVec3f_t col2;
} vktMat3f_t;

typedef struct
{
    vktVec4f_t col0;
    vktVec4f_t col1;
    vktVec4f_t col2;
    vktVec4f_t col3;
} vktMat4f_t;

typedef enum
{
    vktAxisX,
    vktAxisY,
    vktAxisZ,
} vktAxis;

#ifdef __cplusplus
}
#endif
