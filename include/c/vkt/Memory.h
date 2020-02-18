// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stddef.h>

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    vktCopyKindHostToHost,
    vktCopyKindHostToDevice,
    vktCopyKindDeviceToHost,
    vktCopyKindDeviceToDevice,
} vktCopyKind;

VKTAPI void vktAllocate(void** ptr, size_t size);

VKTAPI void vktFree(void* ptr);

VKTAPI void vktMemcpy(void* dst, void const* src, size_t size, vktCopyKind ck);

#ifdef __cplusplus
}
#endif
