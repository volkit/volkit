// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vkt/ManagedResource.h>

#include "common.h"
#include "forward.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    vktRenderAlgoRayMarching,
    vktRenderAlgoImplicitIso,
    vktRenderAlgoMultiScattering,
} vktRenderAlgo;

typedef struct
{
    //! Rendering algorithm
    vktRenderAlgo renderAlgo;

    //! Parameters related to ray marching algorithm
    ///@{

    //! Ray marching step size in object coordinates
    float dtRayMarching;

    ///@}

    //! Parameters related to implicit iso algorithm
    ///@{

    //! The number of activated iso values
    uint16_t numIsoSurfaces;

    //! The maximum number of iso surfaces
    enum { MaxIsoSurfaces = 10 };

    //! The iso surfaces
    float isoSurfaces[MaxIsoSurfaces];

    //! Implicit iso step size in object coordinates
    float dtImplicitIso;

    ///@}

    //! Parameters related to multi-scattering algorithm
    ///@{

    //! Majorant extinction coefficient
    float majorant;

    ///@}

    //! General parameters
    ///@{

    //! RGBA32F lookup table for absorption, emission and scattering albedo
    vktResourceHandle rgbaLookupTable;

    //! Width of the rendering viewport
    int viewportWidth;

    //! Height of the rendering viewport
    int viewportHeight;

    //! Convert final colors from linear to sRGB
    vktBool_t sRGB;

    ///@}

} vktRenderState_t;

static void vktRenderStateDefaultInit(vktRenderState_t* renderState)
{
#ifdef __cplusplus
    *renderState = {
        vktRenderAlgoRayMarching,
        1.f,
        1,
        { .5f },
        1.f,
        1.f,
        vktResourceHandle(-1),
        512,
        512,
        1
        };
#else
    *renderState = (vktRenderState_t) {
        .renderAlgo = vktRenderAlgoRayMarching,
        .dtRayMarching = 1.f,
        .numIsoSurfaces = 1,
        .isoSurfaces = { .5f },
        .dtImplicitIso = 1.f,
        .majorant = 1.f,
        .rgbaLookupTable = -1,
        .viewportWidth = 512,
        .viewportHeight = 512,
        .sRGB = 1
        };
#endif
}

VKTAPI vktError vktRenderSV(vktStructuredVolume volume,
                            vktRenderState_t renderState,
                            vktRenderState_t* newRenderState);

#ifdef __cplusplus
}
#endif
