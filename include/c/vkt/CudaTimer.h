#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct vktCudaTimer_impl;

typedef struct vktCudaTimer_impl* vktCudaTimer;

VKTAPI vktError vktCudaTimerCreate(vktCudaTimer* timer);

VKTAPI vktError vktCudaTimerDestroy(vktCudaTimer timer);

VKTAPI vktError vktCudaTimerReset(vktCudaTimer timer);

VKTAPI vktError vktCudaTimerGetElapsedSeconds(vktCudaTimer timer,
                                              double* seconds);

#ifdef __cplusplus
}
#endif
