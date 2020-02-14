// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdint.h>

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef void* vktManagedResource;
typedef uint32_t vktResourceHandle;

VKTAPI vktResourceHandle vktRegisterManagedResource(vktManagedResource resource);

VKTAPI void vktUnregisterManagedResource(vktResourceHandle handle);

VKTAPI vktManagedResource vktGetManagedResource(vktResourceHandle handle);

#ifdef __cplusplus
}
#endif
