// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "forward.h"
#include "linalg.h"

/*!
 * @brief  instantiate vktArray3D with Type
 */
#define ARRAY3D_INIT(Type)                                                      \
    typedef struct                                                              \
    {                                                                           \
        Type* data_;                                                            \
        vktVec3i_t dims_;                                                       \
    } vktArray3D_##Type;                                                        \


/*!
 * @brief  vktArray3D instantiations for some commonly used types
 */
///@{
ARRAY3D_INIT(vktStructuredVolume)
///@}
