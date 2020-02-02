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
