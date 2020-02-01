// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "common.hpp"
#include "forward.hpp"

namespace vkt
{
    enum class RenderAlgo
    {
        RayMarching,
        MultiScattering,
    };

    struct RenderState
    {
        //! Rendering algorithm
        RenderAlgo renderAlgo = RenderAlgo::RayMarching;

        //! Parameters related to ray marching algorithm
        ///@{

        //! Ray marching step size in object coordinates
        float dt = 1.f;

        ///@}

        //! Parameters related to multi-scattering algorithm
        ///@{

        //! Majorant extinction coefficient
        float majorant = 1.f;

        ///@}

        //! General parameters
        ///@{

        //! Width of the rendering viewport
        int viewportWidth = 512;

        //! Height of the rendering viewport
        int viewportHeight = 512;

        ///@}

    };

    VKTAPI Error Render(StructuredVolume& volume,
                        RenderState const& renderState,
                        RenderState* newRenderState = 0);

} // vkt
