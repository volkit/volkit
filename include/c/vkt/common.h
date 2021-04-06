// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

/*! VKTAPI */
#ifndef VKTAPI

#define VKTAPI

#endif 



/*! Boolean */
typedef uint8_t vktBool_t;
#define VKT_FALSE 0
#define VKT_TRUE  1

/*! Error constants */
typedef enum
{
    vktInvalidValue      = -1,
    vktNoError           =  0,

    vktInvalidDataSource =  1,
    vktReadError         =  2,
    vktWriteError        =  3,
}  vktError;

typedef enum
{
    vktColorFormatUnspecified,

    vktColorFormatR8,
    vktColorFormatRG8,
    vktColorFormatRGB8,
    vktColorFormatRGBA8,
    vktColorFormatR16UI,
    vktColorFormatRG16UI,
    vktColorFormatRGB16UI,
    vktColorFormatRGBA16UI,
    vktColorFormatR32UI,
    vktColorFormatRG32UI,
    vktColorFormatRGB32UI,
    vktColorFormatRGBA32UI,
    vktColorFormatR32F,
    vktColorFormatRG32F,
    vktColorFormatRGB32F,
    vktColorFormatRGBA32F,

    // Keep last!
    vktColorFormatCount,

} vktColorFormat;

typedef enum
{
    vktDataFormatUnspecified,

    vktDataFormatInt8,
    vktDataFormatInt16,
    vktDataFormatInt32,
    vktDataFormatUInt8,
    vktDataFormatUInt16,
    vktDataFormatUInt32,
    vktDataFormatFloat32,

    // Keep last!
    vktVoxelFormatCount,

} vktDataFormat;

typedef enum
{
    vktOpenModeRead,
    vktOpenModeWrite,
    vktOpenModeReadWrite,
} vktOpenMode;
