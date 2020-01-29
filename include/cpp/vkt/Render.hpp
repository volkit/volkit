#pragma once

#include "forward.hpp"

namespace vkt
{
    struct RenderState
    {
    };

    void Render(StructuredVolume& volume,
                RenderState const& renderState,
                RenderState* newRenderState = 0);

} // vkt
