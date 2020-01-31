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
    int x;
    int y;
    int z;
} vktVec3i_t;

typedef enum
{
    vktAxisX,
    vktAxisY,
    vktAxisZ,
} vktAxis;

#ifdef __cplusplus
}
#endif
