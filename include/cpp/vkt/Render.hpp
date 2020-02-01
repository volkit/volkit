// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.hpp"
#include "forward.hpp"

namespace vkt
{
    struct RenderState
    {
    };

    VKTAPI Error Render(StructuredVolume& volume,
                        RenderState const& renderState,
                        RenderState* newRenderState = 0);

} // vkt
