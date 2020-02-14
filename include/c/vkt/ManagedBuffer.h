// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vkt/ExecutionPolicy.h>
#include <vkt/ManagedResource.h>
#include <vkt/Memory.h>

#include "common.h"
#include "forward.h"

#define VKT_MANAGED_BUFFER_DECL__(Type)                                         \
    typedef struct                                                              \
    {                                                                           \
        Type* data_;                                                            \
        size_t size_;                                                           \
                                                                                \
        vktExecutionPolicy_t lastAllocationPolicy_;                             \
        vktResourceHandle resourceHandle_;                                      \
    } vktManagedBuffer_##Type##_impl;                                           \
                                                                                \
    typedef vktManagedBuffer_##Type##_impl* vktManagedBuffer_##Type;            \
                                                                                \
                                                                                \
    /* public */                                                                \
    void vktManagedBuffer_##Type##_CreateEmpty(vktManagedBuffer_##Type* buffer);\
                                                                                \
    void vktManagedBuffer_##Type##_Create(vktManagedBuffer_##Type* buffer,      \
                                          size_t size);                         \
                                                                                \
    void vktManagedBuffer_##Type##_CreateCopy(vktManagedBuffer_##Type* buffer,  \
                                              vktManagedBuffer_##Type rhs);     \
                                                                                \
    void vktManagedBuffer_##Type##_Destroy(vktManagedBuffer_##Type buffer);     \
                                                                                \
    vktResourceHandle vktManagedBuffer_##Type##_GetResourceHandle(              \
                                            vktManagedBuffer_##Type buffer);    \
                                                                                \
    void vktManagedBuffer_##Type##_Migrate(vktManagedBuffer_##Type buffer);     \
                                                                                \
    /* private */                                                               \
    void vktManagedBuffer_##Type##_Allocate__(vktManagedBuffer_##Type buffer,   \
                                              size_t size);                     \
                                                                                \
    void vktManagedBuffer_##Type##_Deallocate__(vktManagedBuffer_##Type buffer);\
                                                                                \
    void vktManagedBuffer_##Type##_Resize__(vktManagedBuffer_##Type buffer,     \
                                            size_t size);                       \
                                                                                \
    void vktManagedBuffer_##Type##_Copy__(vktManagedBuffer_##Type buffer,       \
                                          vktManagedBuffer_##Type rhs);



#define VKT_MANAGED_BUFFER_DEF__(Type)                                          \
    inline void vktManagedBuffer_##Type##_CreateEmpty(                          \
                                            vktManagedBuffer_##Type* buffer)    \
    {                                                                           \
        assert(buffer != NULL);                                                 \
                                                                                \
        *buffer = (vktManagedBuffer_##Type##_impl*)malloc(                      \
                            sizeof(vktManagedBuffer_##Type##_impl));            \
                                                                                \
        memset(                                                                 \
            &((*buffer)->lastAllocationPolicy_),                                \
            0,                                                                  \
            sizeof(vktExecutionPolicy_t)                                        \
            );                                                                  \
                                                                                \
        (*buffer)->size_ = 0;                                                   \
        (*buffer)->data_ = NULL;                                                \
                                                                                \
        (*buffer)->resourceHandle_ = vktRegisterManagedResource(buffer);        \
    }                                                                           \
                                                                                \
    inline void vktManagedBuffer_##Type##_Create(                               \
                                            vktManagedBuffer_##Type* buffer,    \
                                            size_t size)                        \
    {                                                                           \
        assert(buffer != NULL);                                                 \
                                                                                \
        *buffer = (vktManagedBuffer_##Type##_impl*)malloc(                      \
                            sizeof(vktManagedBuffer_##Type##_impl));            \
                                                                                \
        memset(                                                                 \
            &((*buffer)->lastAllocationPolicy_),                                \
            0,                                                                  \
            sizeof(vktExecutionPolicy_t)                                        \
            );                                                                  \
                                                                                \
        vktManagedBuffer_##Type##_Allocate__(*buffer, size);                    \
                                                                                \
        (*buffer)->resourceHandle_ = vktRegisterManagedResource(buffer);        \
    }                                                                           \
                                                                                \
    inline void vktManagedBuffer_##Type##_CreateCopy(                           \
                                            vktManagedBuffer_##Type* buffer,    \
                                            vktManagedBuffer_##Type rhs)        \
    {                                                                           \
        assert(buffer != NULL);                                                 \
                                                                                \
        *buffer = (vktManagedBuffer_##Type##_impl*)malloc(sizeof(Type));        \
                                                                                \
        vktManagedBuffer_##Type##_Migrate(rhs);                                 \
                                                                                \
        vktManagedBuffer_##Type##_Allocate__(*buffer, rhs->size_);              \
                                                                                \
        vktManagedBuffer_##Type##_Copy__(*buffer, rhs);                         \
                                                                                \
        (*buffer)->resourceHandle_ = vktRegisterManagedResource(buffer);        \
    }                                                                           \
                                                                                \
    inline void vktManagedBuffer_##Type##_Destroy(                              \
                                            vktManagedBuffer_##Type buffer)     \
    {                                                                           \
        vktUnregisterManagedResource(buffer->resourceHandle_);                  \
                                                                                \
        vktManagedBuffer_##Type##_Deallocate__(buffer);                         \
                                                                                \
        free(buffer);                                                           \
    }                                                                           \
                                                                                \
    inline vktResourceHandle vktManagedBuffer_##Type##_GetResourceHandle(       \
                                            vktManagedBuffer_##Type buffer)     \
    {                                                                           \
        return buffer->resourceHandle_;                                         \
    }                                                                           \
                                                                                \
                                                                                \
    inline void vktManagedBuffer_##Type##_Migrate(                              \
                                            vktManagedBuffer_##Type buffer)     \
    {                                                                           \
        vktExecutionPolicy_t ep = vktGetThreadExecutionPolicy();                \
                                                                                \
        if (ep.device != buffer->lastAllocationPolicy_.device)                  \
        {                                                                       \
            Type* newData = NULL;                                               \
                                                                                \
            vktAllocate((void**)&newData, buffer->size_ * sizeof(Type));        \
                                                                                \
            vktCopyKind ck = ep.device == vktExecutionPolicyDeviceGPU           \
                            ? vktCopyKindHostToDevice                           \
                            : vktCopyKindDeviceToHost;                          \
                                                                                \
            vktMemcpy(newData, buffer->data_, buffer->size_ * sizeof(Type), ck);\
                                                                                \
            /* Free the data array with the old allocation policy */            \
            vktSetThreadExecutionPolicy(buffer->lastAllocationPolicy_);         \
                                                                                \
            vktFree(buffer->data_);                                             \
                                                                                \
            /* Restore the most recent execution policy */                      \
            vktSetThreadExecutionPolicy(ep);                                    \
                                                                                \
            /* This is now also the policy we used for the last allocation */   \
            buffer->lastAllocationPolicy_ = ep;                                 \
                                                                                \
            /* Migration complete */                                            \
            buffer->data_ = newData;                                            \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* private */                                                               \
    inline void vktManagedBuffer_##Type##_Allocate__(                           \
                                            vktManagedBuffer_##Type buffer,     \
                                            size_t size)                        \
    {                                                                           \
        buffer->lastAllocationPolicy_ = vktGetThreadExecutionPolicy();          \
                                                                                \
        buffer->size_ = size;                                                   \
                                                                                \
        vktAllocate((void**)&buffer->data_, buffer->size_ * sizeof(Type));      \
    }                                                                           \
                                                                                \
    inline void vktManagedBuffer_##Type##_Deallocate__(                         \
                                            vktManagedBuffer_##Type buffer)     \
    {                                                                           \
        /* Free with last allocation policy */                                  \
        vktExecutionPolicy_t curr = vktGetThreadExecutionPolicy();              \
                                                                                \
        vktSetThreadExecutionPolicy(buffer->lastAllocationPolicy_);             \
                                                                                \
        vktFree(buffer->data_);                                                 \
                                                                                \
        /* Make policy from before vktFree() call current */                    \
        buffer->lastAllocationPolicy_ = curr;                                   \
        vktSetThreadExecutionPolicy(curr);                                      \
    }                                                                           \
                                                                                \
    inline void vktManagedBuffer_##Type##_Resize__(                             \
                                            vktManagedBuffer_##Type buffer,     \
                                            size_t size)                        \
    {                                                                           \
        size_t oldSize = buffer->size_;                                         \
        size_t newSize = size;                                                  \
                                                                                \
        vktManagedBuffer_##Type##_Migrate(buffer);                              \
                                                                                \
        vktExecutionPolicy_t ep = vktGetThreadExecutionPolicy();                \
                                                                                \
        vktCopyKind ck = ep.device == vktExecutionPolicyDeviceGPU               \
                        ? vktCopyKindHostToDevice                               \
                        : vktCopyKindDeviceToHost;                              \
                                                                                \
        Type* temp = NULL;                                                      \
        vktAllocate((void**)&temp, oldSize * sizeof(Type));                     \
        vktMemcpy(                                                              \
            temp,                                                               \
            buffer->data_,                                                      \
            oldSize < newSize ? oldSize * sizeof(Type) : newSize * sizeof(Type),\
            ck                                                                  \
            );                                                                  \
                                                                                \
        vktFree(buffer->data_);                                                 \
        vktAllocate((void**)&buffer->data_, newSize * sizeof(Type));            \
        vktMemcpy(                                                              \
            buffer->data_,                                                      \
            temp,                                                               \
            oldSize < newSize ? oldSize * sizeof(Type) : newSize * sizeof(Type),\
            ck                                                                  \
            );                                                                  \
                                                                                \
        vktFree(temp);                                                          \
                                                                                \
        buffer->size_ = newSize;                                                \
    }                                                                           \
                                                                                \
    inline void vktManagedBuffer_##Type##_Copy__(                               \
                                            vktManagedBuffer_##Type buffer,     \
                                            vktManagedBuffer_##Type rhs)        \
    {                                                                           \
        vktManagedBuffer_##Type##_Migrate(rhs);                                 \
                                                                                \
        vktCopyKind ck = buffer->lastAllocationPolicy_.device                   \
                            == vktExecutionPolicyDeviceGPU                      \
                            ? vktCopyKindDeviceToDevice                         \
                            : vktCopyKindHostToHost;                            \
                                                                                \
        size_t size = buffer->size_ < rhs->size_                                \
                        ? buffer->size_ * sizeof(Type)                          \
                        : rhs->size_ * sizeof(Type);                            \
        vktMemcpy(buffer->data_, rhs->data_, size, ck);                         \
    }


#define VKT_MANAGED_BUFFER_INIT(Type)                                           \
    VKT_MANAGED_BUFFER_DECL__(Type)                                             \
    VKT_MANAGED_BUFFER_DEF__(Type)

/*!
 * @brief  vktManagedBuffer instantiations for some commonly used types
 */
///@{
VKT_MANAGED_BUFFER_INIT(uint8_t)
VKT_MANAGED_BUFFER_INIT(vktStructuredVolume)
///@}
