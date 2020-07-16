// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include <vkt/ManagedBuffer.h>

#include "common.h"
#include "forward.h"
#include "linalg.h"

/*!
 * @brief  instantiate vktArray3D with Type
 */
#define VKT_ARRAY3D_DECL__(Type)                                                \
    typedef struct                                                              \
    {                                                                           \
        vktManagedBuffer_##Type##_impl* base_;                                  \
        vktVec3i_t dims_;                                                       \
    } vktArray3D_##Type##_impl;                                                 \
                                                                                \
    typedef vktArray3D_##Type##_impl* vktArray3D_##Type;                        \
    typedef Type vktArray3D_##Type##_ValueType;                                 \
    typedef Type* vktArray3D_##Type##_Iterator;                                 \
    typedef Type const* vktArray3D_##Type##_ConstIterator;                      \
                                                                                \
                                                                                \
    /* public */                                                                \
    void vktArray3D_##Type##_CreateEmpty(vktArray3D_##Type* arr);               \
                                                                                \
    void vktArray3D_##Type##_Create(vktArray3D_##Type* arr,                     \
                                    vktVec3i_t dims);                           \
                                                                                \
    void vktArray3D_##Type##_CreateCopy(vktArray3D_##Type* arr,                 \
                                        vktArray3D_##Type rhs);                 \
                                                                                \
    void vktArray3D_##Type##_Destroy(vktArray3D_##Type arr);                    \
                                                                                \
    void vktArray3D_##Type##_Resize(vktArray3D_##Type arr,                      \
                                    vktVec3i_t dims);                           \
                                                                                \
    void vktArray3D_##Type##_Fill(vktArray3D_##Type arr,                        \
                                  Type value);                                  \
                                                                                \
    vktArray3D_##Type##_Iterator vktArray3D_##Type##_Begin(                     \
                                        vktArray3D_##Type arr);                 \
                                                                                \
    vktArray3D_##Type##_ConstIterator vktArray3D_##Type##_CBegin(               \
                                        vktArray3D_##Type arr);                 \
                                                                                \
    vktArray3D_##Type##_Iterator vktArray3D_##Type##_End(                       \
                                        vktArray3D_##Type arr);                 \
                                                                                \
    vktArray3D_##Type##_ConstIterator vktArray3D_##Type##_CEnd(                 \
                                        vktArray3D_##Type arr);                 \
                                                                                \
    vktArray3D_##Type##_ValueType* vktArray3D_##Type##_Access(                  \
                                        vktArray3D_##Type arr,                  \
                                        vktVec3i_t index);                      \
                                                                                \
    vktArray3D_##Type##_ValueType const* vktArray3D_##Type##_CAccess(           \
                                        vktArray3D_##Type arr,                  \
                                        vktVec3i_t index);                      \
                                                                                \
    vktBool_t vktArray3D_##Type##_Empty(vktArray3D_##Type arr);                 \
                                                                                \
    Type* vktArray3D_##Type##_Data(vktArray3D_##Type arr);                      \
                                                                                \
    Type const* vktArray3D_##Type##_CData(vktArray3D_##Type arr);               \
                                                                                \
    vktVec3i_t vktArray3D_##Type##_Dims(vktArray3D_##Type arr);                 \
                                                                                \
    size_t vktArray3D_##Type##_NumElements(vktArray3D_##Type arr);



#define VKT_ARRAY3D_DEF__(Type)                                                 \
    inline void vktArray3D_##Type##_CreateEmpty(vktArray3D_##Type* arr)         \
    {                                                                           \
        assert(arr != NULL);                                                    \
                                                                                \
        *arr = (vktArray3D_##Type##_impl*)malloc(                               \
                        sizeof(vktArray3D_##Type##_impl));                      \
                                                                                \
        vktManagedBuffer_##Type##_CreateEmpty(&(*arr)->base_);                  \
        (*arr)->dims_.x = 0;                                                    \
        (*arr)->dims_.y = 0;                                                    \
        (*arr)->dims_.z = 0;                                                    \
    }                                                                           \
                                                                                \
    inline void vktArray3D_##Type##_Create(vktArray3D_##Type* arr,              \
                                           vktVec3i_t dims)                     \
    {                                                                           \
        assert(arr != NULL);                                                    \
                                                                                \
        *arr = (vktArray3D_##Type##_impl*)malloc(                               \
                        sizeof(vktArray3D_##Type##_impl));                      \
                                                                                \
        vktManagedBuffer_##Type##_Create(                                       \
                &(*arr)->base_,                                                 \
                dims.x * size_t(dims.y) * dims.z * sizeof(Type)                 \
                );                                                              \
        (*arr)->dims_ = dims;                                                   \
    }                                                                           \
                                                                                \
    inline void vktArray3D_##Type##_CreateCopy(vktArray3D_##Type* arr,          \
                                               vktArray3D_##Type rhs)           \
    {                                                                           \
        assert(arr != NULL);                                                    \
                                                                                \
        *arr = (vktArray3D_##Type##_impl*)malloc(sizeof(vktArray3D_##Type));    \
                                                                                \
        vktManagedBuffer_##Type##_CreateCopy(&(*arr)->base_, rhs->base_);       \
        (*arr)->dims_ = rhs->dims_;                                             \
    }                                                                           \
                                                                                \
    inline void vktArray3D_##Type##_Destroy(vktArray3D_##Type arr)              \
    {                                                                           \
        vktManagedBuffer_##Type##_Destroy(arr->base_);                          \
                                                                                \
        free(arr);                                                              \
    }                                                                           \
                                                                                \
    inline void vktArray3D_##Type##_Resize(vktArray3D_##Type arr,               \
                                           vktVec3i_t dims)                     \
    {                                                                           \
        vktManagedBuffer_##Type##_Resize__(                                     \
                arr->base_,                                                     \
                dims.x * size_t(dims.y) * dims.z * sizeof(Type)                 \
                );                                                              \
        arr->dims_ = dims;                                                      \
    }                                                                           \
                                                                                \
    inline void vktArray3D_##Type##_Fill(vktArray3D_##Type arr,                 \
                                         Type value)                            \
    {                                                                           \
        vktManagedBuffer_##Type##_Fill__(arr->base_, value);                    \
    }                                                                           \
                                                                                \
    inline vktArray3D_##Type##_Iterator vktArray3D_##Type##_Begin(              \
                                                    vktArray3D_##Type arr)      \
    {                                                                           \
        return vktArray3D_##Type##_Data(arr);                                   \
    }                                                                           \
                                                                                \
    inline vktArray3D_##Type##_ConstIterator vktArray3D_##Type##_CBegin(        \
                                                    vktArray3D_##Type arr)      \
    {                                                                           \
        return vktArray3D_##Type##_CData(arr);                                  \
    }                                                                           \
                                                                                \
    inline vktArray3D_##Type##_Iterator vktArray3D_##Type##_End(                \
                                                    vktArray3D_##Type arr)      \
    {                                                                           \
        return vktArray3D_##Type##_Data(arr)                                    \
             + vktArray3D_##Type##_NumElements(arr);                            \
    }                                                                           \
                                                                                \
    inline vktArray3D_##Type##_ConstIterator vktArray3D_##Type##_CEnd(          \
                                                    vktArray3D_##Type arr)      \
    {                                                                           \
        return vktArray3D_##Type##_CData(arr)                                   \
             + vktArray3D_##Type##_NumElements(arr);                            \
    }                                                                           \
                                                                                \
    inline vktArray3D_##Type##_ValueType* vktArray3D_##Type##_Access(           \
                                                    vktArray3D_##Type arr,      \
                                                    vktVec3i_t index)           \
    {                                                                           \
        size_t linearIndex = index.z * arr->dims_.x * (size_t)arr->dims_.y      \
                           + index.y * arr->dims_.x                             \
                           + index.x;                                           \
                                                                                \
        return &vktArray3D_##Type##_Data(arr)[linearIndex];                     \
    }                                                                           \
                                                                                \
    inline vktArray3D_##Type##_ValueType const* vktArray3D_##Type##_CAccess(    \
                                                    vktArray3D_##Type arr,      \
                                                    vktVec3i_t index)           \
    {                                                                           \
        size_t linearIndex = index.z * arr->dims_.x * (size_t)arr->dims_.y      \
                           + index.y * arr->dims_.x                             \
                           + index.x;                                           \
                                                                                \
        return &vktArray3D_##Type##_CData(arr)[linearIndex];                    \
    }                                                                           \
                                                                                \
    inline vktBool_t vktArray3D_##Type##_Empty(vktArray3D_##Type arr)           \
    {                                                                           \
        return vktArray3D_##Type##_NumElements(arr) == 0;                       \
    }                                                                           \
                                                                                \
    inline Type* vktArray3D_##Type##_Data(vktArray3D_##Type arr)                \
    {                                                                           \
        vktManagedBuffer_##Type##_Migrate(arr->base_);                          \
                                                                                \
        return arr->base_->data_;                                               \
    }                                                                           \
                                                                                \
    inline Type const* vktArray3D_##Type##_CData(vktArray3D_##Type arr)         \
    {                                                                           \
        vktManagedBuffer_##Type##_Migrate(arr->base_);                          \
                                                                                \
        return arr->base_->data_;                                               \
    }                                                                           \
                                                                                \
    inline vktVec3i_t vktArray3D_##Type##_Dims(vktArray3D_##Type arr)           \
    {                                                                           \
        return arr->dims_;                                                      \
    }                                                                           \
                                                                                \
    inline size_t vktArray3D_##Type##_NumElements(vktArray3D_##Type arr)        \
    {                                                                           \
        return arr->dims_.x * (size_t)arr->dims_.y * arr->dims_.z;              \
    }


#define VKT_ARRAY3D_INIT(Type)                                                  \
    VKT_ARRAY3D_DECL__(Type)                                                    \
    VKT_ARRAY3D_DEF__(Type)

/*!
 * @brief  vktArray3D instantiations for some commonly used types
 */
///@{
VKT_ARRAY3D_INIT(vktStructuredVolume)
///@}
