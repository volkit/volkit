#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct vktCudaTimer_impl;

typedef struct vktCudaTimer_impl* vktCudaTimer;

VKTAPI void vktCudaTimerCreate(vktCudaTimer* timer);

VKTAPI void vktCudaTimerDestroy(vktCudaTimer timer);

VKTAPI void vktCudaTimerReset(vktCudaTimer timer);

VKTAPI double vktCudaTimerGetElapsedSeconds(vktCudaTimer timer);

#ifdef __cplusplus
}
#endif
