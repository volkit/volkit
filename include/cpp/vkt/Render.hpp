// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>

#include <vkt/ManagedResource.hpp>

#include "common.hpp"
#include "forward.hpp"
#include "linalg.hpp"

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

        //! Parameters related to animation
        ///@{

        unsigned animationFrame = 0;

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

        //! Initial camera, optionally set by the user
        struct
        {
            //! By default, this isn't set but determined using viewAll()
            vkt::Bool isSet = 0;

            //! Position where the camera is at
            vkt::Vec3f eye = { 0.f, 0.f, 0.f };

            //! Position that we're looking at
            vkt::Vec3f center = { 0.f, 0.f, -1.f };

            //! Camera up vector
            vkt::Vec3f up = { 0.f, 1.f, 0.f };

            //! vertical field of view, specified in degree
            float fovy = 45.f;

            //! Lens radius, used for depth of field
            float lensRadius = .001f;

            //! Distance we're focusing at, used for depth of field
            float focalDistance = 10.f;
        } initialCamera;

        //! Take snapshots using a key or when the viewer was closed
        struct
        {
            //! Not enabled by default
            vkt::Bool enabled = 0;

            //! File to store the snap shot to. Ending determines file type
            char const* fileName = "";

            //! Overrides key press
            vkt::Bool takeOnClose = 0;

            //! If @see takeOnClose not set, this key is used
            char key = 'p';

            //! Optional message that is printed to the console
            char const* message = "";
        } snapshotTool;

        ///@}

    };

    /*! @brief  Render single volume
     *
     * Render a single structured volume. @param volume is a managed volkit
     * structured volume object. Rendering of multiple volumes can be done
     * using @ref RenderFrames().
     * @see RenderFrames()
     */
    VKTAPI Error Render(StructuredVolume& volume,
                        RenderState const& renderState = {},
                        RenderState* newRenderState = 0);

    /*! @brief  Render volumes as animation frames
     *
     * Render numAnimationFrames volumes as a sequence of time steps. @param
     * volume is a CPU-accessible raw pointer pointing to managed volkit
     * structured volume objects.
     * @see Render()
     */
    VKTAPI Error RenderFrames(StructuredVolume* volumes,
                              std::size_t numAnimationFrames,
                              RenderState const& renderState = {},
                              RenderState* newRenderState = 0);

    /*! @brief  Render single volume
     *
     * Render a single hierarchical volume. @param volume is a managed volkit
     * hierarchical volume object. Rendering of multiple volumes can be done
     * using @ref RenderFrames().
     * @see RenderFrames()
     */
    VKTAPI Error Render(HierarchicalVolume& volume,
                        RenderState const& renderState = {},
                        RenderState* newRenderState = 0);

    /*! @brief  Render volumes as animation frames
     *
     * Render numAnimationFrames volumes as a sequence of time steps. @param
     * volume is a CPU-accessible raw pointer pointing to managed volkit
     * hierarchical volume objects.
     * @see Render()
     */
    VKTAPI Error RenderFrames(HierarchicalVolume* volumes,
                              std::size_t numAnimationFrames,
                              RenderState const& renderState = {},
                              RenderState* newRenderState = 0);
} // vkt
