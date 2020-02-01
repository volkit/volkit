// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    vktExecutionPolicyDeviceCPU,
    vktExecutionPolicyDeviceGPU,
} vktExecutionPolicyDevice;

typedef enum
{
    vktExecutionPolicyHostAPISerial,
    //vktExecutionPolicyAPIOpenMP,
    //vktExecutionPolicyAPITBB,
} vktExecutionPolicyHostAPI;

typedef enum
{
    vktExecutionPolicyDeviceAPICUDA,
    //vktExecutionPolicyDeviceAPIOpenCL,
} vktExecutionPolicyDeviceAPI;

typedef struct
{
    vktExecutionPolicyDevice device;
    vktExecutionPolicyHostAPI hostApi;
    vktExecutionPolicyDeviceAPI deviceApi;
} vktExecutionPolicy_t;

VKTAPI void vktSetThreadExecutionPolicy(vktExecutionPolicy_t policy);

VKTAPI vktExecutionPolicy_t vktGetThreadExecutionPolicy();

#ifdef __cplusplus
}
#endif
