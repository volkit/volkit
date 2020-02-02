// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

/*! VKTAPI */
#define VKTAPI

/*! Boolean */
typedef uint8_t vktBool;

/*! Error constants */
typedef enum
{
    vktInvalidValue      = -1,
    vktNoError           =  0,

    vktInvalidDataSource =  1,
    vktReadError         =  2,
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
