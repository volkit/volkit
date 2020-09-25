// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/ManagedResource.hpp>

#include "common.hpp"
#include "forward.hpp"

namespace vkt
{
    enum class RenderAlgo
    {
        RayMarching,
        ImplicitIso,
        MultiScattering,
    };

    struct RenderState
    {
        //! Rendering algorithm
        RenderAlgo renderAlgo = RenderAlgo::RayMarching;

        //! Parameters related to ray marching algorithm
        ///@{

        //! Ray marching step size in object coordinates
        float dtRayMarching = 1.f;

        ///@}

        //! Parameters related to implicit iso algorithm
        ///@{

        //! The number of activated iso values
        uint16_t numIsoSurfaces = 1;

        //! The maximum number of iso surfaces
        enum { MaxIsoSurfaces = 10 };

        //! The iso surfaces
        float isoSurfaces[MaxIsoSurfaces] = { .5f };

        //! Implicit iso step size in object coordinates
        float dtImplicitIso = 1.f;

        ///@}

        //! Parameters related to multi-scattering algorithm
        ///@{

        //! Majorant extinction coefficient
        float majorant = 1.f;

        ///@}

        //! General parameters
        ///@{

        //! RGBA32F lookup table for absorption, emission and scattering albedo
        ResourceHandle rgbaLookupTable = ResourceHandle(-1);

        //! Histogram, will be displayed in a widget inside the viewport
        ResourceHandle histogram = ResourceHandle(-1);

        //! Width of the rendering viewport
        int viewportWidth = 512;

        //! Height of the rendering viewport
        int viewportHeight = 512;

        //! Convert final colors from linear to sRGB
        vkt::Bool sRGB = 1;

        ///@}

    };

    VKTAPI Error Render(StructuredVolume& volume,
                        RenderState const& renderState = {},
                        RenderState* newRenderState = 0);

} // vkt
