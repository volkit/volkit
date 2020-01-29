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

VKTAPI vktError vktSetThreadExecutionPolicy(vktExecutionPolicy_t policy);

VKTAPI vktError vktGetThreadExecutionPolicy(vktExecutionPolicy_t* policy);

#ifdef __cplusplus
}
#endif
