#pragma once

#include "forward.h"
#include "linalg.h"

/*!
 * @brief  instantiate vktArray3D with Type
 */
#define ARRAY3D_INIT(Type)                                                      \
    typedef struct                                                              \
    {                                                                           \
        vkt##Type* data_;                                                            \
        vktVec3i_t dims_;                                                       \
    } vktArray3D_##Type;                                                        \


/*!
 * @brief  vktArray3D instantiations for some commonly used types
 */
///@{
ARRAY3D_INIT(StructuredVolume)
///@}
